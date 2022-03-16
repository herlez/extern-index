/**
 * @file    src/em_succinct_irreducible_src/process_halfsegment_pairs.hpp
 * @section LICENCE
 *
 * This file is part of EM-SuccinctIrreducible v0.2.0
 * See: http://www.cs.helsinki.fi/group/pads/
 *
 * Copyright (C) 2016-2017
 *   Juha Karkkainen <juha.karkkainen (at) cs.helsinki.fi>
 *   Dominik Kempa <dominik.kempa (at) gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 **/

#ifndef __SRC_EM_SUCCINCT_IRREDUCIBLE_SRC_PROCESS_HALFSEGMENT_PAIRS_HPP_INCLUDED
#define __SRC_EM_SUCCINCT_IRREDUCIBLE_SRC_PROCESS_HALFSEGMENT_PAIRS_HPP_INCLUDED

#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>
#include <omp.h>

#include "io/async_stream_reader.hpp"
#include "io/async_stream_writer.hpp"
#include "io/async_multi_stream_writer.hpp"
#include "convert_to_vbyte_slab.hpp"
#include "utils.hpp"


namespace em_succinct_irreducible_private {

template<typename char_type>
std::uint64_t naive_lcp(
    std::uint64_t i,
    std::uint64_t j,
    std::uint64_t lcp,
    std::FILE *f_text,
    std::uint64_t text_length,
    std::uint64_t &io_volume) {

  // Allocate buffers.
  static const std::uint64_t bufsize = (1L << 20);
  char_type *b1 = utils::allocate_array<char_type>(bufsize);
  char_type *b2 = utils::allocate_array<char_type>(bufsize);

  // Initialize the I/O volume.
  std::uint64_t io_vol = 0;

  // Finish the LCP computation.
  while (true) {
    std::uint64_t toread = std::min(bufsize,
        text_length - std::max(i, j) - lcp);

    if (!toread) break;
    std::uint64_t offset1 = (i + lcp) * sizeof(char_type);
    std::uint64_t offset2 = (j + lcp) * sizeof(char_type);
    utils::read_at_offset(b1, offset1, toread, f_text);
    utils::read_at_offset(b2, offset2, toread, f_text);

    io_vol += 2UL * toread * sizeof(char_type);
    std::uint64_t lcp_delta = 0;

    while (lcp_delta < toread &&
        b1[lcp_delta] == b2[lcp_delta])
      ++lcp_delta;

    lcp += lcp_delta;
    if (lcp_delta < toread)
      break;
  }

  // Clean up.
  utils::deallocate(b1);
  utils::deallocate(b2);

  // Update I/O volume.
  io_volume += io_vol;

  // Return the result.
  return lcp;
}

namespace normal_mode {

struct buf_item_ext {
  std::uint64_t m_left_idx;
  std::uint64_t m_right_idx;
  std::uint64_t m_ans;
  std::uint64_t m_block_id;
};

template<typename char_type,
  typename text_offset_type>
std::uint64_t process_halfsegment_pairs_large_B(
    std::string text_filename,
    std::uint64_t text_length,
    std::uint64_t max_block_size_B,
    std::uint64_t max_halfsegment_size,
    std::uint64_t max_overflow_size,
    std::string **pairs_filenames,
    std::string *irreducible_bits_filenames,
    std::uint64_t &total_io_volume) {

  // Print initial message and start the timer.
  fprintf(stderr, "  Compute irreducible LCP values:\n");
  long double start = utils::wclock();

  // Initialize basic parameters.
  std::uint64_t n_blocks_B =
    (2UL * text_length + max_block_size_B - 1) / max_block_size_B;
  std::uint64_t n_halfsegments =
    (text_length + max_halfsegment_size - 1) / max_halfsegment_size;
  std::uint64_t sum_irreducible_lcps = 0;

  // Open file with text.
  std::FILE *f_text = utils::file_open(text_filename, "r");

  // Initialize multiwriter of values 2i + PLCP[i].
  typedef async_multi_stream_writer<text_offset_type> lcp_multiwriter_type;
  lcp_multiwriter_type *lcp_multiwriter = NULL;
  {
    static const std::uint64_t n_free_buffers = 4;
    std::uint64_t buffer_size = (1UL << 20);
    lcp_multiwriter =
      new lcp_multiwriter_type(n_blocks_B, buffer_size, n_free_buffers);
    for (std::uint64_t block_id = 0; block_id < n_blocks_B; ++block_id)
      lcp_multiwriter->add_file(irreducible_bits_filenames[block_id]);
  }

  // Allocate halfsegments.
  std::uint64_t max_ext_halfsegment_size =
    max_halfsegment_size + max_overflow_size;
  char_type *left_halfsegment =
    utils::allocate_array<char_type>(max_ext_halfsegment_size);
  char_type *right_halfsegment =
    utils::allocate_array<char_type>(max_ext_halfsegment_size);

  // Allocate buffers.
  static const std::uint64_t local_buf_size = (1UL << 20);
  text_offset_type *idx_buf =
    utils::allocate_array<text_offset_type>(local_buf_size * 2);
#ifdef _OPENMP
  buf_item_ext *ans_buf =
    utils::allocate_array<buf_item_ext>(local_buf_size);
#endif

  // Processing of halfsegment pairs follows.
  for (std::uint64_t left_halfsegment_id = 0;
      left_halfsegment_id < n_halfsegments; ++left_halfsegment_id) {
    std::uint64_t left_halfsegment_beg =
      left_halfsegment_id * max_halfsegment_size;
    std::uint64_t left_halfsegment_end =
      std::min(left_halfsegment_beg + max_halfsegment_size, text_length);
    std::uint64_t left_halfsegment_ext_end =
      std::min(left_halfsegment_end + max_overflow_size, text_length);
    std::uint64_t left_halfsegment_ext_size =
      left_halfsegment_ext_end - left_halfsegment_beg;
    bool left_halfsegment_loaded = false;

    // Scan all halfsegments to the right of left_halfsegment_id.
    for (std::uint64_t right_halfsegment_id = left_halfsegment_id;
        right_halfsegment_id < n_halfsegments; right_halfsegment_id++) {
      std::uint64_t right_halfsegment_beg =
        right_halfsegment_id * max_halfsegment_size;
      std::uint64_t right_halfsegment_end =
        std::min(right_halfsegment_beg + max_halfsegment_size, text_length);
      std::uint64_t right_halfsegment_ext_end =
        std::min(right_halfsegment_end + max_overflow_size, text_length);
      std::uint64_t right_halfsegment_ext_size =
        right_halfsegment_ext_end - right_halfsegment_beg;

      // Check if that pair of halfsegments has any associated pairs.
      std::string pairs_filename =
        pairs_filenames[left_halfsegment_id][right_halfsegment_id];
      if (utils::file_exists(pairs_filename) == false ||
          utils::file_size(pairs_filename) == 0) {
        if (utils::file_exists(pairs_filename))
          utils::file_delete(pairs_filename);
        continue;
      }

      // Print initial progress message and start the timer.
      fprintf(stderr, "    Process halfsegments %lu and %lu: ",
          left_halfsegment_id, right_halfsegment_id);
      long double halfsegment_process_start = utils::wclock();

      // Initialize basic stats.
      std::uint64_t local_lcp_sum = 0;
      std::uint64_t extra_io = 0;
      std::uint64_t io_vol = 0;

      // Initialize reading from file associated
      // with current pair of halfsegments.
      typedef async_stream_reader<text_offset_type> pair_reader_type;
      pair_reader_type *pair_reader = new pair_reader_type(pairs_filename);

      // Compute the number of pairs to process.
      std::uint64_t n_pairs =
        utils::file_size(pairs_filename) / (2 * sizeof(text_offset_type));

      // Read left halfsegment from disk (if it was not already.
      if (left_halfsegment_loaded == false) {
        std::uint64_t offset = left_halfsegment_beg * sizeof(char_type);
        utils::read_at_offset(left_halfsegment, offset,
            left_halfsegment_ext_size, text_filename);
        left_halfsegment_loaded = true;
        extra_io += left_halfsegment_ext_size * sizeof(char_type);
      }

      // Read right halfsegment from disk.
      char_type *right_halfsegment_ptr = right_halfsegment;
      if (right_halfsegment_id != left_halfsegment_id) {
        std::uint64_t offset = right_halfsegment_beg * sizeof(char_type);
        utils::read_at_offset(right_halfsegment, offset,
            right_halfsegment_ext_size, text_filename);
        extra_io += right_halfsegment_ext_size * sizeof(char_type);
      } else right_halfsegment_ptr = left_halfsegment;

      // Process all pairs.
      std::uint64_t pairs_processed = 0;
      while (pairs_processed < n_pairs) {
        std::uint64_t filled =
          std::min(n_pairs - pairs_processed, local_buf_size);
        pair_reader->read(idx_buf, filled * 2);

#ifdef _OPENMP
        std::vector<std::uint64_t> long_lcps;
        std::uint64_t max_threads = omp_get_max_threads();
        std::uint64_t max_block_size =
          (filled + max_threads - 1) / max_threads;
        std::uint64_t n_threads =
          (filled + max_block_size - 1) / max_block_size;

        #pragma omp parallel num_threads(n_threads)
        {
          std::uint64_t thread_id = omp_get_thread_num();
          std::uint64_t block_beg = thread_id * max_block_size;
          std::uint64_t block_end =
            std::min(block_beg + max_block_size, filled);

          std::vector<std::uint64_t> local_long_lcps;
          std::uint64_t thread_lcp_sum = 0;

          // Process a block assigned to a given thread.
          for (std::uint64_t j = block_beg; j < block_end; ++j) {
            std::uint64_t i = idx_buf[2 * j];
            std::uint64_t phi_i = idx_buf[2 * j + 1];
            std::uint64_t left_idx = i;
            std::uint64_t right_idx = phi_i;

            // Swap positions if necessary.
            if (!(left_halfsegment_beg <= left_idx &&
                  left_idx < left_halfsegment_end &&
                  right_halfsegment_beg <= right_idx &&
                  right_idx < right_halfsegment_end))
              std::swap(left_idx, right_idx);

            // Compute LCP value.
            std::uint64_t lcp = 0;
            while (left_idx + lcp < left_halfsegment_ext_end &&
                right_idx + lcp < right_halfsegment_ext_end &&
                left_halfsegment[left_idx - left_halfsegment_beg + lcp] ==
                right_halfsegment_ptr[right_idx - right_halfsegment_beg + lcp])
              ++lcp;

            // If the LCP computation cannot be completed,
            // add it to the list of unfinished LCPs.
            if ((left_idx + lcp == left_halfsegment_ext_end &&
                  left_halfsegment_ext_end < text_length) ||
                (right_idx + lcp == right_halfsegment_ext_end &&
                 right_halfsegment_ext_end < text_length)) {

              ans_buf[j].m_left_idx = left_idx;
              ans_buf[j].m_right_idx = right_idx;
              ans_buf[j].m_ans = lcp;
              local_long_lcps.push_back(j);
            } else {

              // Compute answer.
              std::uint64_t pos_B = 2UL * i + lcp;
              std::uint64_t block_id_B = pos_B / max_block_size_B;
              std::uint64_t block_beg_B = block_id_B * max_block_size_B;
              std::uint64_t offset_B = pos_B - block_beg_B;

              // Write answer to buffer.
              ans_buf[j].m_ans = offset_B;
              ans_buf[j].m_block_id = block_id_B;
              thread_lcp_sum += lcp;
            }
          }

          #pragma omp critical
          {

            // Concatenate the list of long LCP processed
            // by a given thread with a global list.
            long_lcps.insert(long_lcps.end(),
                local_long_lcps.begin(), local_long_lcps.end());
            local_lcp_sum += thread_lcp_sum;
          }
        }

        // Finish the computation of long LCPs using naive method.
        for (std::uint64_t j = 0; j < long_lcps.size(); ++j) {
          std::uint64_t which = long_lcps[j];

          // Retreive indexes from the buffer.
          std::uint64_t i = idx_buf[2 * which];
          std::uint64_t left_idx = ans_buf[which].m_left_idx;
          std::uint64_t right_idx = ans_buf[which].m_right_idx;
          std::uint64_t lcp = ans_buf[which].m_ans;

          // Compute LCP.
          lcp = naive_lcp<char_type>(left_idx, right_idx,
              lcp, f_text, text_length, io_vol);

          // Compute answer.
          std::uint64_t pos_B = 2UL * i + lcp;
          std::uint64_t block_id_B = pos_B / max_block_size_B;
          std::uint64_t block_beg_B = block_id_B * max_block_size_B;
          std::uint64_t offset_B = pos_B - block_beg_B;

          // Write answer to buffer.
          ans_buf[which].m_ans = offset_B;
          ans_buf[which].m_block_id = block_id_B;

          // Update statistics.
          local_lcp_sum += lcp;
        }

        // Write LCPs to file.
        for (std::uint64_t j = 0; j < filled; ++j)
          lcp_multiwriter->write_to_ith_file(
              ans_buf[j].m_block_id, ans_buf[j].m_ans);

#else
        for (std::uint64_t j = 0; j < filled; ++j) {
          std::uint64_t i = idx_buf[2 * j];
          std::uint64_t phi_i = idx_buf[2 * j + 1];
          std::uint64_t left_idx = i;
          std::uint64_t right_idx = phi_i;

          // Swap positions if necessary.
          if (!(left_halfsegment_beg <= left_idx &&
                left_idx < left_halfsegment_end &&
                right_halfsegment_beg <= right_idx &&
                right_idx < right_halfsegment_end))
            std::swap(left_idx, right_idx);

          // Compute LCP value.
          std::uint64_t lcp = 0;
          while (left_idx + lcp < left_halfsegment_ext_end &&
              right_idx + lcp < right_halfsegment_ext_end &&
              left_halfsegment[left_idx - left_halfsegment_beg + lcp] ==
              right_halfsegment_ptr[right_idx - right_halfsegment_beg + lcp])
            ++lcp;

          // Finish the long LCP using naive method.
          if ((left_idx + lcp == left_halfsegment_ext_end &&
                left_halfsegment_ext_end < text_length) ||
              (right_idx + lcp == right_halfsegment_ext_end &&
               right_halfsegment_ext_end < text_length))
            lcp = naive_lcp<char_type>(left_idx, right_idx,
                lcp, f_text, text_length, io_vol);

          // Compute answer.
          std::uint64_t pos_B = 2 * i + lcp;
          std::uint64_t block_id_B = pos_B / max_block_size_B;
          std::uint64_t block_beg_B = block_id_B * max_block_size_B;
          std::uint64_t offset_B = pos_B - block_beg_B;

          // Write answer to file.
          lcp_multiwriter->write_to_ith_file(block_id_B, offset_B);

          // Update statistics.
          local_lcp_sum += lcp;
        }
#endif

        // Update the pair counter.
        pairs_processed += filled;
      }

      // Stop I/O threads.
      pair_reader->stop_reading();

      // Update I/O volume.
      io_vol +=
        pair_reader->bytes_read() +
        extra_io +
        n_pairs * sizeof(text_offset_type);
      total_io_volume += io_vol;

      // Clean up.
      delete pair_reader;
      utils::file_delete(pairs_filename);

      // Update statistics.
      sum_irreducible_lcps += local_lcp_sum;

      // Print summary.
      long double avg_lcp =
        (long double)local_lcp_sum / (long double)std::max(1UL, n_pairs);
      long double elapsed = utils::wclock() - halfsegment_process_start;
      fprintf(stderr, "time = %.1Lfs, I/O = %.1LfMiB/s, avg. LCP = %.2Lf\n",
          elapsed, (1.L * io_vol / (1L << 20)) / elapsed, avg_lcp);
    }
  }

  // Clean up.
#ifdef _OPENMP
  utils::deallocate(ans_buf);
#endif

  utils::deallocate(idx_buf);
  utils::deallocate(right_halfsegment);
  utils::deallocate(left_halfsegment);
  delete lcp_multiwriter;
  std::fclose(f_text);

  // Print summary.
  long double total_time = utils::wclock() - start;
  fprintf(stderr, "    Total time = %.2Lfs, "
      "total I/O vol = %.2Lfbytes/input symbol\n",
      total_time, (1.L * total_io_volume) / text_length);

  // Return the result.
  return sum_irreducible_lcps;
}

struct buf_item {
  std::uint64_t m_left_idx;
  std::uint64_t m_right_idx;
  std::uint64_t m_ans;
};

template<typename char_type,
  typename text_offset_type>
std::uint64_t process_halfsegment_pairs_small_B(
    std::string text_filename,
    std::uint64_t text_length,
    std::uint64_t max_halfsegment_size,
    std::uint64_t max_overflow_size,
    std::string **pairs_filenames,
    std::string low_pos_filename,
    std::string high_pos_filename,
    std::uint64_t &total_io_volume) {

  // Print initial message and start the timer.
  fprintf(stderr, "  Compute irreducible LCP values:\n");
  long double start = utils::wclock();

  // Initialize basic parameters.
  std::uint64_t n_halfsegments =
    (text_length + max_halfsegment_size - 1) / max_halfsegment_size;
  std::uint64_t sum_irreducible_lcps = 0;
  
  // Open file with text.
  std::FILE *f_text = utils::file_open(text_filename, "r");

  // Initialize writers of positions 2i + PLCP[i].
  typedef async_stream_writer<text_offset_type> pos_writer_type;
  pos_writer_type *low_pos_writer = new pos_writer_type(low_pos_filename);
  pos_writer_type *high_pos_writer = new pos_writer_type(high_pos_filename);

  // Allocate halfsegments.
  std::uint64_t max_ext_halfsegment_size =
    max_halfsegment_size + max_overflow_size;
  char_type *left_halfsegment =
    utils::allocate_array<char_type>(max_ext_halfsegment_size);
  char_type *right_halfsegment =
    utils::allocate_array<char_type>(max_ext_halfsegment_size);

  // Allocate buffers.
  static const std::uint64_t local_buf_size = (1UL << 20);
  text_offset_type *idx_buf =
    utils::allocate_array<text_offset_type>(local_buf_size * 2);

#ifdef _OPENMP
  buf_item *ans_buf =
    utils::allocate_array<buf_item>(local_buf_size);
#endif

  // Processing of halfsegment pairs follows.
  for (std::uint64_t left_halfsegment_id = 0;
      left_halfsegment_id < n_halfsegments; ++left_halfsegment_id) {
    std::uint64_t left_halfsegment_beg =
      left_halfsegment_id * max_halfsegment_size;
    std::uint64_t left_halfsegment_end =
      std::min(left_halfsegment_beg + max_halfsegment_size, text_length);
    std::uint64_t left_halfsegment_ext_end =
      std::min(left_halfsegment_end + max_overflow_size, text_length);
    std::uint64_t left_halfsegment_ext_size =
      left_halfsegment_ext_end - left_halfsegment_beg;
    bool left_halfsegment_loaded = false;

    // Scan all halfsegments to the right of left_halfsegment_id.
    for (std::uint64_t right_halfsegment_id = left_halfsegment_id;
        right_halfsegment_id < n_halfsegments; right_halfsegment_id++) {
      std::uint64_t right_halfsegment_beg =
        right_halfsegment_id * max_halfsegment_size;
      std::uint64_t right_halfsegment_end =
        std::min(right_halfsegment_beg + max_halfsegment_size, text_length);
      std::uint64_t right_halfsegment_ext_end =
        std::min(right_halfsegment_end + max_overflow_size, text_length);
      std::uint64_t right_halfsegment_ext_size =
        right_halfsegment_ext_end - right_halfsegment_beg;

      // Check if that pair of halfsegments has any associated pairs.
      std::string pairs_filename =
        pairs_filenames[left_halfsegment_id][right_halfsegment_id];
      if (utils::file_exists(pairs_filename) == false ||
          utils::file_size(pairs_filename) == 0) {
        if (utils::file_exists(pairs_filename))
          utils::file_delete(pairs_filename);
        continue;
      }

      // Print initial progress message and start the timer.
      fprintf(stderr, "    Process halfsegments %lu and %lu: ",
          left_halfsegment_id, right_halfsegment_id);
      long double halfsegment_process_start = utils::wclock();

      // Initialize basic stats.
      std::uint64_t local_lcp_sum = 0;
      std::uint64_t extra_io = 0;
      std::uint64_t io_volume = 0;

      // Initialize reading from file associated
      // with current pair of halfsegments.
      typedef async_stream_reader<text_offset_type> pair_reader_type;
      pair_reader_type *pair_reader = new pair_reader_type(pairs_filename);

      // Compute the number of pairs to process.
      std::uint64_t n_pairs =
        utils::file_size(pairs_filename) / (2 * sizeof(text_offset_type));

      // Read left halfsegment from disk (if it wasn't already)
      if (left_halfsegment_loaded == false) {
        std::uint64_t offset = left_halfsegment_beg * sizeof(char_type);
        utils::read_at_offset(left_halfsegment, offset,
            left_halfsegment_ext_size, text_filename);
        left_halfsegment_loaded = true;
        extra_io += left_halfsegment_ext_size * sizeof(char_type);
      }

      // Read right halfsegment from disk.
      char_type *right_halfsegment_ptr = right_halfsegment;
      if (right_halfsegment_id != left_halfsegment_id) {
        std::uint64_t offset = right_halfsegment_beg * sizeof(char_type);
        utils::read_at_offset(right_halfsegment, offset,
            right_halfsegment_ext_size, text_filename);
        extra_io += right_halfsegment_ext_size * sizeof(char_type);
      } else right_halfsegment_ptr = left_halfsegment;

      // Process all pairs.
      std::uint64_t pairs_processed = 0;
      while (pairs_processed < n_pairs) {
        std::uint64_t filled =
          std::min(n_pairs - pairs_processed, local_buf_size);
        pair_reader->read(idx_buf, filled * 2);

#ifdef _OPENMP
        std::vector<std::uint64_t> long_lcps;
        std::uint64_t max_threads = omp_get_max_threads();
        std::uint64_t max_block_size =
          (filled + max_threads - 1) / max_threads;
        std::uint64_t n_threads =
          (filled + max_block_size - 1) / max_block_size;

        #pragma omp parallel num_threads(n_threads)
        {
          std::uint64_t thread_id = omp_get_thread_num();
          std::uint64_t block_beg = thread_id * max_block_size;
          std::uint64_t block_end =
            std::min(block_beg + max_block_size, filled);

          std::vector<std::uint64_t> local_long_lcps;
          std::uint64_t thread_lcp_sum = 0;

          // Process a block assigned to a given thread.
          for (std::uint64_t j = block_beg; j < block_end; ++j) {
            std::uint64_t i = idx_buf[2 * j];
            std::uint64_t phi_i = idx_buf[2 * j + 1];
            std::uint64_t left_idx = i;
            std::uint64_t right_idx = phi_i;

            // Swap positions if necessary.
            if (!(left_halfsegment_beg <= left_idx &&
                  left_idx < left_halfsegment_end &&
                  right_halfsegment_beg <= right_idx &&
                  right_idx < right_halfsegment_end))
              std::swap(left_idx, right_idx);

            // Compute LCP value.
            std::uint64_t lcp = 0;
            while (left_idx + lcp < left_halfsegment_ext_end &&
                right_idx + lcp < right_halfsegment_ext_end &&
                left_halfsegment[left_idx - left_halfsegment_beg + lcp] ==
                right_halfsegment_ptr[right_idx - right_halfsegment_beg + lcp])
              ++lcp;

            // If the LCP computation cannot be completed,
            // add it to the list of unfinished LCPs.
            if ((left_idx + lcp == left_halfsegment_ext_end &&
                  left_halfsegment_ext_end < text_length) ||
                (right_idx + lcp == right_halfsegment_ext_end &&
                 right_halfsegment_ext_end < text_length)) {
              ans_buf[j].m_left_idx = left_idx;
              ans_buf[j].m_right_idx = right_idx;
              ans_buf[j].m_ans = lcp;
              local_long_lcps.push_back(j);
            } else {
              std::uint64_t pos_in_B = 2UL * i + lcp;
              ans_buf[j].m_ans = pos_in_B;
              thread_lcp_sum += lcp;
            }
          }

          #pragma omp critical
          {

            // Concatenate the list of long LCP processed
            // by a given thread with a global list.
            long_lcps.insert(long_lcps.end(),
                local_long_lcps.begin(), local_long_lcps.end());
            local_lcp_sum += thread_lcp_sum;
          }
        }

        // Finish the computation of long LCPs using naive method.
        for (std::uint64_t j = 0; j < long_lcps.size(); ++j) {
          std::uint64_t which = long_lcps[j];

          // Retreive indexes from the buffer.
          std::uint64_t i = idx_buf[2 * which];
          std::uint64_t left_idx = ans_buf[which].m_left_idx;
          std::uint64_t right_idx = ans_buf[which].m_right_idx;
          std::uint64_t lcp = ans_buf[which].m_ans;

          // Compute LCP.
          lcp = naive_lcp<char_type>(left_idx, right_idx,
              lcp, f_text, text_length, io_volume);

          // Compute answer.
          std::uint64_t pos_in_B = 2UL * i + lcp;

          // Write answer to buffer.
          ans_buf[which].m_ans = pos_in_B;

          // Update stats.
          local_lcp_sum += lcp;
        }

        // Write LCPs to file.
        for (std::uint64_t j = 0; j < filled; ++j) {
          std::uint64_t pos = ans_buf[j].m_ans;
          if (pos < text_length)
            low_pos_writer->write(pos);
          else high_pos_writer->write(pos - text_length);
        }

#else
        for (std::uint64_t j = 0; j < filled; ++j) {
          std::uint64_t i = idx_buf[2 * j];
          std::uint64_t phi_i = idx_buf[2 * j + 1];
          std::uint64_t left_idx = i;
          std::uint64_t right_idx = phi_i;

          // Swap positions if necessary.
          if (!(left_halfsegment_beg <= left_idx &&
                left_idx < left_halfsegment_end &&
                right_halfsegment_beg <= right_idx &&
                right_idx < right_halfsegment_end))
            std::swap(left_idx, right_idx);

          // Compute LCP value.
          std::uint64_t lcp = 0;
          while (left_idx + lcp < left_halfsegment_ext_end &&
              right_idx + lcp < right_halfsegment_ext_end &&
              left_halfsegment[left_idx - left_halfsegment_beg + lcp] ==
              right_halfsegment_ptr[right_idx - right_halfsegment_beg + lcp])
            ++lcp;

          // Finish the computation of long LCP using naive method.
          if ((left_idx + lcp == left_halfsegment_ext_end &&
                left_halfsegment_ext_end < text_length) ||
            (right_idx + lcp == right_halfsegment_ext_end &&
             right_halfsegment_ext_end < text_length))
            lcp = naive_lcp<char_type>(left_idx, right_idx,
                lcp, f_text, text_length, io_volume);

          // Write LCP to file.
          std::uint64_t pos_in_B = 2 * i + lcp;
          if (pos_in_B < text_length)
            low_pos_writer->write(pos_in_B);
          else high_pos_writer->write(pos_in_B - text_length);
          local_lcp_sum += lcp;
        }
#endif

        pairs_processed += filled;
      }

      // Stop I/O threads.
      pair_reader->stop_reading();

      // Update I/O volume.
      io_volume +=
        pair_reader->bytes_read() +
        extra_io +
        n_pairs * sizeof(text_offset_type);
      total_io_volume += io_volume;

      // Clean up.
      delete pair_reader;
      utils::file_delete(pairs_filename);

      // Update statistics.
      sum_irreducible_lcps += local_lcp_sum;

      // Print summary.
      long double avg_lcp =
        (long double)local_lcp_sum / (long double)std::max(1UL, n_pairs);
      long double elapsed = utils::wclock() - halfsegment_process_start;
      fprintf(stderr, "time = %.1Lfs, I/O = %.1LfMiB/s, avg. LCP = %.2Lf\n",
          elapsed, (1.L * io_volume / (1L << 20)) / elapsed, avg_lcp);
    }
  }

  // Clean up.
#ifdef _OPENMP
  utils::deallocate(ans_buf);
#endif

  utils::deallocate(idx_buf);
  utils::deallocate(right_halfsegment);
  utils::deallocate(left_halfsegment);
  delete high_pos_writer;
  delete low_pos_writer;
  std::fclose(f_text);

  // Print summary.
  long double total_time = utils::wclock() - start;
  fprintf(stderr, "    Total time = %.2Lfs, "
      "total I/O vol = %.2Lfbytes/input symbol\n",
      total_time, (1.L * total_io_volume) / text_length);

  return sum_irreducible_lcps;
}

}  // namespace normal_mode

namespace inplace_mode {

template<typename char_type,
  typename text_offset_type>
std::uint64_t process_halfsegment_pairs_large_B(
    std::string text_filename,
    std::uint64_t text_length,
    std::uint64_t max_halfsegment_size,
    std::uint64_t max_overflow_size,
    std::string **pos_filenames,
    std::string **phi_filenames,
    std::string **lcp_filenames,
    std::uint64_t &total_io_volume) {

  // Print initial message and start the timer.
  fprintf(stderr, "    Compute irreducible LCP values:\n");
  long double start = utils::wclock();

  // Initialize basic parameters.
  std::uint64_t n_halfsegments =
    (text_length + max_halfsegment_size - 1) / max_halfsegment_size;
  std::uint64_t sum_irreducible_lcps = 0;

  // Open file with text.
  std::FILE *f_text = utils::file_open(text_filename, "r");

  // Allocate halfsegments.
  char_type *left_halfsegment = utils::allocate_array<char_type>(
      max_halfsegment_size + max_overflow_size);
  char_type *right_halfsegment = utils::allocate_array<char_type>(
      max_halfsegment_size + max_overflow_size);

  // Allocate buffers.
  static const std::uint64_t local_buf_size = (1UL << 20);
  text_offset_type *pos_buf =
    utils::allocate_array<text_offset_type>(local_buf_size);
  text_offset_type *phi_buf =
    utils::allocate_array<text_offset_type>(local_buf_size);

#ifdef _OPENMP
  text_offset_type *lcp_buf =
    utils::allocate_array<text_offset_type>(local_buf_size);
  std::uint8_t *vbyte_slab =
    utils::allocate_array<std::uint8_t>(local_buf_size * 9);
#endif

  // Processing of halfsegment pairs follows.
  for (std::uint64_t left_halfsegment_id = 0;
      left_halfsegment_id < n_halfsegments; ++left_halfsegment_id) {
    std::uint64_t left_halfsegment_beg =
      left_halfsegment_id * max_halfsegment_size;
    std::uint64_t left_halfsegment_end =
      std::min(left_halfsegment_beg + max_halfsegment_size, text_length);
    std::uint64_t left_halfsegment_ext_end =
      std::min(left_halfsegment_end + max_overflow_size, text_length);
    std::uint64_t left_halfsegment_ext_size =
      left_halfsegment_ext_end - left_halfsegment_beg;
    bool left_halfsegment_loaded = false;

    // Scan all halfsegments to the right of left_halfsegment_id.
    for (std::uint64_t right_halfsegment_id = left_halfsegment_id;
        right_halfsegment_id < n_halfsegments; right_halfsegment_id++) {
      std::uint64_t right_halfsegment_beg =
        right_halfsegment_id * max_halfsegment_size;
      std::uint64_t right_halfsegment_end =
        std::min(right_halfsegment_beg + max_halfsegment_size, text_length);
      std::uint64_t right_halfsegment_ext_end =
        std::min(right_halfsegment_end + max_overflow_size, text_length);
      std::uint64_t right_halfsegment_ext_size =
        right_halfsegment_ext_end - right_halfsegment_beg;

      // Check if that pair of halfsegments has any associated pairs.
      std::string pos_filename =
        pos_filenames[left_halfsegment_id][right_halfsegment_id];
      std::string phi_filename =
        phi_filenames[left_halfsegment_id][right_halfsegment_id];
      std::string lcp_filename =
        lcp_filenames[left_halfsegment_id][right_halfsegment_id];
      if (utils::file_exists(pos_filename) == false ||
          utils::file_size(pos_filename) == 0) {
        if (utils::file_exists(pos_filename))
          utils::file_delete(pos_filename);
        if (utils::file_exists(phi_filename))
          utils::file_delete(phi_filename);
        continue;
      }

      // Print initial progress message.
      fprintf(stderr, "      Process halfsegments %lu and %lu: ",
          left_halfsegment_id, right_halfsegment_id);

      // Initialize the timer.
      long double halfsegment_process_start = utils::wclock();

      // Initialize basic stats.
      std::uint64_t local_lcp_sum = 0;
      std::uint64_t extra_io = 0;
      std::uint64_t io_vol = 0;

      // Initialize the readers of irreducible positions.
      typedef async_stream_reader<text_offset_type> irr_pos_reader_type;
      irr_pos_reader_type *pos_reader = new irr_pos_reader_type(pos_filename);
      irr_pos_reader_type *phi_reader = new irr_pos_reader_type(phi_filename);

      // Initialize the writer of lcp values.
      typedef async_stream_writer<std::uint8_t> lcp_writer_type;
      lcp_writer_type *lcp_writer = new lcp_writer_type(lcp_filename);

      // Get the number of pairs to process.
      std::uint64_t n_pairs =
        utils::file_size(pos_filename) / sizeof(text_offset_type);

      // Read left halfsegment from disk (if it wasn't already)
      if (left_halfsegment_loaded == false) {
        std::uint64_t offset = left_halfsegment_beg * sizeof(char_type);
        utils::read_at_offset(left_halfsegment, offset,
            left_halfsegment_ext_size, text_filename);
        left_halfsegment_loaded = true;
        extra_io += left_halfsegment_ext_size * sizeof(char_type);
      }

      // Read right halfsegment from disk.
      char_type *right_halfsegment_ptr = right_halfsegment;
      if (right_halfsegment_id != left_halfsegment_id) {
        std::uint64_t offset = right_halfsegment_beg * sizeof(char_type);
        utils::read_at_offset(right_halfsegment, offset,
            right_halfsegment_ext_size, text_filename);
        extra_io += right_halfsegment_ext_size * sizeof(char_type);
      } else right_halfsegment_ptr = left_halfsegment;

      // Process all pairs.
      std::uint64_t pairs_processed = 0;
      while (pairs_processed < n_pairs) {
        std::uint64_t filled =
          std::min(n_pairs - pairs_processed, local_buf_size);
        pos_reader->read(pos_buf, filled);
        phi_reader->read(phi_buf, filled);

#ifdef _OPENMP
        std::vector<std::uint64_t> long_lcps;
        std::uint64_t max_threads = omp_get_max_threads();
        std::uint64_t max_block_size =
          (filled + max_threads - 1) / max_threads;
        std::uint64_t n_threads =
          (filled + max_block_size - 1) / max_block_size;

        #pragma omp parallel num_threads(n_threads)
        {
          std::uint64_t thread_id = omp_get_thread_num();
          std::uint64_t block_beg = thread_id * max_block_size;
          std::uint64_t block_end =
            std::min(block_beg + max_block_size, filled);
          std::vector<std::uint64_t> local_long_lcps;
          std::uint64_t thread_lcp_sum = 0;

          for (std::uint64_t j = block_beg; j < block_end; ++j) {
            std::uint64_t i = pos_buf[j];
            std::uint64_t phi_i = phi_buf[j];
            std::uint64_t left_idx = i;
            std::uint64_t right_idx = phi_i;

            // Swap positions if necessary.
            if (!(left_halfsegment_beg <= left_idx &&
                  left_idx < left_halfsegment_end &&
                  right_halfsegment_beg <= right_idx &&
                  right_idx < right_halfsegment_end))
              std::swap(left_idx, right_idx);

            // Compute LCP value.
            std::uint64_t lcp = 0;
            while (left_idx + lcp < left_halfsegment_ext_end &&
                right_idx + lcp < right_halfsegment_ext_end &&
                left_halfsegment[left_idx - left_halfsegment_beg + lcp] ==
                right_halfsegment_ptr[right_idx - right_halfsegment_beg + lcp])
              ++lcp;

            // Write LCP to buffer.
            lcp_buf[j] = lcp;

            // If the LCP computation cannot be completed,
            // add it to the list of unfinished LCPs.
            if ((left_idx + lcp == left_halfsegment_ext_end &&
                  left_halfsegment_ext_end < text_length) ||
                (right_idx + lcp == right_halfsegment_ext_end &&
                 right_halfsegment_ext_end < text_length))
              local_long_lcps.push_back(j);
            else thread_lcp_sum += lcp;
          }

          #pragma omp critical
          {

            // Concatenate the list of long LCP processed
            // by a given thread with a global list.
            long_lcps.insert(long_lcps.end(),
                local_long_lcps.begin(), local_long_lcps.end());
            local_lcp_sum += thread_lcp_sum;
          }
        }

        // Finish the computatino of long LCPs using naive method.
        for (std::uint64_t j = 0; j < long_lcps.size(); ++j) {
          std::uint64_t which = long_lcps[j];
          std::uint64_t i = pos_buf[which];
          std::uint64_t phi_i = phi_buf[which];
          std::uint64_t left_idx = i;
          std::uint64_t right_idx = phi_i;

          // Swap positions if necessary.
          if (!(left_halfsegment_beg <= left_idx &&
                left_idx < left_halfsegment_end &&
                right_halfsegment_beg <= right_idx &&
                right_idx < right_halfsegment_end))
            std::swap(left_idx, right_idx);

          // Finish computing LCP naively.
          std::uint64_t lcp = lcp_buf[which];
          lcp = naive_lcp<char_type>(left_idx, right_idx,
              lcp, f_text, text_length, io_vol);

          // Write LCP to buffer.
          lcp_buf[which] = lcp;

          // Update stats.
          local_lcp_sum += lcp;
        }

        // Convert LCP values to vbyte encoding (in parallel).
        std::uint64_t vbyte_slab_length =
          convert_to_vbyte_slab(lcp_buf, filled, vbyte_slab);

        // Write LCP values to disk.
        lcp_writer->write(vbyte_slab, vbyte_slab_length);

#else
        for (std::uint64_t j = 0; j < filled; ++j) {
          std::uint64_t i = pos_buf[j];
          std::uint64_t phi_i = phi_buf[j];
          std::uint64_t left_idx = i;
          std::uint64_t right_idx = phi_i;

          // Swap positions if necessary.
          if (!(left_halfsegment_beg <= left_idx &&
                left_idx < left_halfsegment_end &&
                right_halfsegment_beg <= right_idx &&
                right_idx < right_halfsegment_end))
            std::swap(left_idx, right_idx);

          // Compute LCP value.
          std::uint64_t lcp = 0;
          while (left_idx + lcp < left_halfsegment_ext_end &&
              right_idx + lcp < right_halfsegment_ext_end &&
              left_halfsegment[left_idx - left_halfsegment_beg + lcp] ==
              right_halfsegment_ptr[right_idx - right_halfsegment_beg + lcp])
            ++lcp;

          // Finish the long LCP using naive method.
          if ((left_idx + lcp == left_halfsegment_ext_end &&
                left_halfsegment_ext_end < text_length) ||
              (right_idx + lcp == right_halfsegment_ext_end &&
               right_halfsegment_ext_end < text_length))
            lcp = naive_lcp<char_type>(left_idx, right_idx,
                lcp, f_text, text_length, io_vol);

          // Write LCP to file. We convert the value and write to file
          // immediatelly to better overlap of I/O and computation.
          std::uint64_t val = lcp;
          while (val > 127) {
            std::uint64_t x = (val & 0x7f) | 0x80;
            lcp_writer->write((std::uint8_t)x);
            val >>= 7;
          }
          lcp_writer->write((std::uint8_t)val);

          // Update statistics.
          local_lcp_sum += lcp;
        }
#endif

        // Update the number of processed pairs.
        pairs_processed += filled;
      }

      // Stop I/O threads.
      pos_reader->stop_reading();
      phi_reader->stop_reading();

      // Update I/O volume.
      io_vol +=
        pos_reader->bytes_read() +
        phi_reader->bytes_read() +
        lcp_writer->bytes_written() +
        extra_io;
      total_io_volume += io_vol;

      // Clean up.
      delete lcp_writer;
      delete phi_reader;
      delete pos_reader;
      utils::file_delete(phi_filename);

      // Update statistics.
      sum_irreducible_lcps += local_lcp_sum;

      // Print summary.
      long double avg_lcp =
        (long double)local_lcp_sum / (long double)std::max(1UL, n_pairs);
      long double elapsed = utils::wclock() - halfsegment_process_start;
      fprintf(stderr, "time = %.1Lfs, I/O = %.1LfMiB/s, avg. LCP = %.2Lf\n",
          elapsed, (1.L * io_vol / (1L << 20)) / elapsed, avg_lcp);
    }
  }

  // Clean up.
#ifdef _OPENMP
  utils::deallocate(vbyte_slab);
  utils::deallocate(lcp_buf);
#endif

  utils::deallocate(phi_buf);
  utils::deallocate(pos_buf);
  utils::deallocate(right_halfsegment);
  utils::deallocate(left_halfsegment);
  std::fclose(f_text);

  // Print summary.
  long double total_time = utils::wclock() - start;
  fprintf(stderr, "      Total time = %.2Lfs, "
      "total I/O vol = %.2Lfbytes/input symbol\n",
      total_time, (1.L * total_io_volume) / text_length);

  // Return the result.
  return sum_irreducible_lcps;
}

template<typename char_type,
  typename text_offset_type>
std::uint64_t process_halfsegment_pairs_small_B(  // namespace for large/small B?
    std::string text_filename,
    std::uint64_t text_length,
    std::uint64_t max_halfsegment_size,
    std::uint64_t max_overflow_size,
    std::string **pos_filenames,
    std::string **phi_filenames,
    std::string **lcp_filenames,
    std::uint64_t &total_io_volume) {

  // Print the initial message and start the timer.
  fprintf(stderr, "    Compute irreducible LCP values:\n");
  long double start = utils::wclock();

  // Initialize basic parameters.    
  std::uint64_t n_halfsegments =
    (text_length + max_halfsegment_size - 1) / max_halfsegment_size;
  std::uint64_t sum_irreducible_lcps = 0;
  
  // Open file with text.
  std::FILE *f_text = utils::file_open(text_filename, "r");

  // Allocate halfsegments.
  std::uint64_t max_ext_halfsegment_size =
    max_halfsegment_size + max_overflow_size;
  char_type *left_halfsegment =
    utils::allocate_array<char_type>(max_ext_halfsegment_size);
  char_type *right_halfsegment =
    utils::allocate_array<char_type>(max_ext_halfsegment_size);

  // Allocate buffers.
  static const std::uint64_t local_buf_size = (1UL << 20);
  text_offset_type *pos_buf =
    utils::allocate_array<text_offset_type>(local_buf_size);
  text_offset_type *phi_buf =
    utils::allocate_array<text_offset_type>(local_buf_size);

#ifdef _OPENMP
  text_offset_type *lcp_buf =
    utils::allocate_array<text_offset_type>(local_buf_size);
  std::uint8_t *vbyte_slab =
    utils::allocate_array<std::uint8_t>(local_buf_size * 9);
#endif

  // Processing of halfsegment pairs follows.
  for (std::uint64_t left_halfsegment_id = 0;
      left_halfsegment_id < n_halfsegments; ++left_halfsegment_id) {
    std::uint64_t left_halfsegment_beg =
      left_halfsegment_id * max_halfsegment_size;
    std::uint64_t left_halfsegment_end =
      std::min(left_halfsegment_beg + max_halfsegment_size, text_length);
    std::uint64_t left_halfsegment_ext_end =
      std::min(left_halfsegment_end + max_overflow_size, text_length);
    std::uint64_t left_halfsegment_ext_size =
      left_halfsegment_ext_end - left_halfsegment_beg;
    bool left_halfsegment_loaded = false;

    // Scan all halfsegments to the right of left_halfsegment_id.
    for (std::uint64_t right_halfsegment_id = left_halfsegment_id;
        right_halfsegment_id < n_halfsegments; right_halfsegment_id++) {
      std::uint64_t right_halfsegment_beg =
        right_halfsegment_id * max_halfsegment_size;
      std::uint64_t right_halfsegment_end =
        std::min(right_halfsegment_beg + max_halfsegment_size, text_length);
      std::uint64_t right_halfsegment_ext_end =
        std::min(right_halfsegment_end + max_overflow_size, text_length);
      std::uint64_t right_halfsegment_ext_size =
        right_halfsegment_ext_end - right_halfsegment_beg;

      // Check if that pair of halfsegments has any associated pairs.
      std::string pos_filename =
        pos_filenames[left_halfsegment_id][right_halfsegment_id];
      std::string phi_filename =
        phi_filenames[left_halfsegment_id][right_halfsegment_id];
      std::string lcp_filename =
        lcp_filenames[left_halfsegment_id][right_halfsegment_id];
      if (utils::file_exists(pos_filename) == false ||
          utils::file_size(pos_filename) == 0) {
        if (utils::file_exists(pos_filename))
          utils::file_delete(pos_filename);
        if (utils::file_exists(phi_filename))
          utils::file_delete(phi_filename);
        continue;
      }

      // Print initial progress message.
      fprintf(stderr, "      Process halfsegments %lu and %lu: ",
          left_halfsegment_id, right_halfsegment_id);

      // Start the timer.
      long double halfsegment_process_start = utils::wclock();

      // Initialize basic stats.
      std::uint64_t local_lcp_sum = 0;
      std::uint64_t extra_io = 0;
      std::uint64_t io_vol = 0;

      // Initialize the readers of irreducible positions.
      typedef async_stream_reader<text_offset_type> irr_pos_reader_type;
      irr_pos_reader_type *pos_reader = new irr_pos_reader_type(pos_filename);
      irr_pos_reader_type *phi_reader = new irr_pos_reader_type(phi_filename);

      // Initialize the writer of lcp values.
      typedef async_stream_writer<std::uint8_t> lcp_writer_type;
      lcp_writer_type *lcp_writer = new lcp_writer_type(lcp_filename);

      // Get the number of pairs to process.
      std::uint64_t n_pairs =
        utils::file_size(pos_filename) / sizeof(text_offset_type);

      // Read left halfsegment from disk (if it wasn't already)
      if (left_halfsegment_loaded == false) {
        std::uint64_t offset = left_halfsegment_beg * sizeof(char_type);
        utils::read_at_offset(left_halfsegment, offset,
            left_halfsegment_ext_size, text_filename);
        left_halfsegment_loaded = true;
        extra_io += left_halfsegment_ext_size * sizeof(char_type);
      }

      // Read right halfsegment from disk.
      char_type *right_halfsegment_ptr = right_halfsegment;
      if (right_halfsegment_id != left_halfsegment_id) {
        std::uint64_t offset = right_halfsegment_beg * sizeof(char_type);
        utils::read_at_offset(right_halfsegment, offset,
            right_halfsegment_ext_size, text_filename);
        extra_io += right_halfsegment_ext_size * sizeof(char_type);
      } else right_halfsegment_ptr = left_halfsegment;

      // Process all pairs.
      std::uint64_t pairs_processed = 0;
      while (pairs_processed < n_pairs) {
        std::uint64_t filled =
          std::min(n_pairs - pairs_processed, local_buf_size);
        pos_reader->read(pos_buf, filled);
        phi_reader->read(phi_buf, filled);

#ifdef _OPENMP
        std::vector<std::uint64_t> long_lcps;
        std::uint64_t max_threads = omp_get_max_threads();
        std::uint64_t max_block_size =
          (filled + max_threads - 1) / max_threads;
        std::uint64_t n_threads =
          (filled + max_block_size - 1) / max_block_size;

        #pragma omp parallel num_threads(n_threads)
        {
          std::uint64_t thread_id = omp_get_thread_num();
          std::uint64_t block_beg = thread_id * max_block_size;
          std::uint64_t block_end =
            std::min(block_beg + max_block_size, filled);

          std::vector<std::uint64_t> local_long_lcps;
          std::uint64_t thread_lcp_sum = 0;

          for (std::uint64_t j = block_beg; j < block_end; ++j) {
            std::uint64_t i = pos_buf[j];
            std::uint64_t phi_i = phi_buf[j];
            std::uint64_t left_idx = i;
            std::uint64_t right_idx = phi_i;

            // Swap positions if necessary.
            if (!(left_halfsegment_beg <= left_idx &&
                  left_idx < left_halfsegment_end &&
                  right_halfsegment_beg <= right_idx &&
                  right_idx < right_halfsegment_end))
              std::swap(left_idx, right_idx);

            // Compute LCP value.
            std::uint64_t lcp = 0;
            while (left_idx + lcp < left_halfsegment_ext_end &&
                right_idx + lcp < right_halfsegment_ext_end &&
                left_halfsegment[left_idx - left_halfsegment_beg + lcp] ==
                right_halfsegment_ptr[right_idx - right_halfsegment_beg + lcp])
              ++lcp;

            // Write LCP to buffer.
            lcp_buf[j] = lcp;

            // If the LCP computation cannot be completed,
            // add it to the list of unfinished LCPs.
            if ((left_idx + lcp == left_halfsegment_ext_end &&
                  left_halfsegment_ext_end < text_length) ||
                (right_idx + lcp == right_halfsegment_ext_end &&
                 right_halfsegment_ext_end < text_length))
              local_long_lcps.push_back(j);
            else thread_lcp_sum += lcp;
          }

          #pragma omp critical
          {

            // Concatenate the list of long LCP processed
            // by a given thread with a global list.
            long_lcps.insert(long_lcps.end(),
                local_long_lcps.begin(), local_long_lcps.end());
            local_lcp_sum += thread_lcp_sum;
          }
        }

        // Finish the computation of long LCPs using naive method.
        for (std::uint64_t j = 0; j < long_lcps.size(); ++j) {
          std::uint64_t which = long_lcps[j];
          std::uint64_t i = pos_buf[which];
          std::uint64_t phi_i = phi_buf[which];
          std::uint64_t left_idx = i;
          std::uint64_t right_idx = phi_i;

          // Swap positions if necessary.
          if (!(left_halfsegment_beg <= left_idx &&
                left_idx < left_halfsegment_end &&
                right_halfsegment_beg <= right_idx &&
                right_idx < right_halfsegment_end))
            std::swap(left_idx, right_idx);

          // Finish computing LCP naively.
          std::uint64_t lcp = lcp_buf[which];
          lcp = naive_lcp<char_type>(left_idx, right_idx,
              lcp, f_text, text_length, io_vol);

          // Write LCP to buffer.
          lcp_buf[which] = lcp;

          // Update stats.
          local_lcp_sum += lcp;
        }

        // Convert LCP values to vbyte encoding (in parallel).
        std::uint64_t vbyte_slab_length =
          convert_to_vbyte_slab(lcp_buf, filled, vbyte_slab);

        // Write LCP values to disk.
        lcp_writer->write(vbyte_slab, vbyte_slab_length);

#else
        for (std::uint64_t j = 0; j < filled; ++j) {
          std::uint64_t i = pos_buf[j];
          std::uint64_t phi_i = phi_buf[j];
          std::uint64_t left_idx = i;
          std::uint64_t right_idx = phi_i;

          // Swap positions if necessary.
          if (!(left_halfsegment_beg <= left_idx &&
                left_idx < left_halfsegment_end &&
                right_halfsegment_beg <= right_idx &&
                right_idx < right_halfsegment_end))
            std::swap(left_idx, right_idx);

          // Compute LCP value.
          std::uint64_t lcp = 0;
          while (left_idx + lcp < left_halfsegment_ext_end &&
              right_idx + lcp < right_halfsegment_ext_end &&
              left_halfsegment[left_idx - left_halfsegment_beg + lcp] ==
              right_halfsegment_ptr[right_idx - right_halfsegment_beg + lcp])
            ++lcp;

          // Finish the computation of long LCP using naive method.
          if ((left_idx + lcp == left_halfsegment_ext_end &&
                left_halfsegment_ext_end < text_length) ||
            (right_idx + lcp == right_halfsegment_ext_end &&
             right_halfsegment_ext_end < text_length))
            lcp = naive_lcp<char_type>(left_idx, right_idx,
                lcp, f_text, text_length, io_vol);

          // Write LCP to file. We convert the value and write to file
          // immediatelly to better overlap of I/O and computation.
          std::uint64_t val = lcp;
          while (val > 127) {
            std::uint64_t x = (val & 0x7f) | 0x80;
            lcp_writer->write((std::uint8_t)x);
            val >>= 7;
          }
          lcp_writer->write((std::uint8_t)val);

          // Update statistics.
          local_lcp_sum += lcp;
        }
#endif

        // Update the number of processed pairs.
        pairs_processed += filled;
      }

      // Stop I/O threads.
      pos_reader->stop_reading();
      phi_reader->stop_reading();

      // Update I/O volume.
      io_vol +=
        pos_reader->bytes_read() +
        phi_reader->bytes_read() +
        lcp_writer->bytes_written() +
        extra_io;
      total_io_volume += io_vol;

      // Clean up.
      delete lcp_writer;
      delete phi_reader;
      delete pos_reader;
      utils::file_delete(phi_filename);

      // Update statistics.
      sum_irreducible_lcps += local_lcp_sum;

      // Print summary.
      long double avg_lcp =
        (long double)local_lcp_sum / (long double)std::max(1UL, n_pairs);
      long double elapsed = utils::wclock() - halfsegment_process_start;
      fprintf(stderr, "time = %.1Lfs, I/O = %.1LfMiB/s, avg. LCP = %.2Lf\n",
          elapsed, (1.L * io_vol / (1L << 20)) / elapsed, avg_lcp);
    }
  }

  // Clean up.
#ifdef _OPENMP
  utils::deallocate(vbyte_slab);
  utils::deallocate(lcp_buf);
#endif

  utils::deallocate(phi_buf);
  utils::deallocate(pos_buf);
  utils::deallocate(right_halfsegment);
  utils::deallocate(left_halfsegment);
  std::fclose(f_text);

  // Print summary.
  long double total_time = utils::wclock() - start;
  fprintf(stderr, "      Total time = %.2Lfs, "
      "total I/O vol = %.2Lfbytes/input symbol\n",
      total_time, (1.L * total_io_volume) / text_length);

  // Return the result.
  return sum_irreducible_lcps;
}

}  // namespace inplace_mode
}  // namespace em_succinct_irreducible_private

#endif  // __SRC_EM_SUCCINCT_IRREDUCIBLE_SRC_PROCESS_HALFSEGMENT_PAIRS_HPP_INCLUDED
