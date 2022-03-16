/**
 * @file    src/em_succinct_irreducible_src/compute_B.hpp
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

#ifndef __SRC_EM_SUCCINCT_IRREDUCIBLE_SRC_COMPUTE_B_HPP_INCLUDED
#define __SRC_EM_SUCCINCT_IRREDUCIBLE_SRC_COMPUTE_B_HPP_INCLUDED

#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>
#include <omp.h>

#include "io/async_stream_reader.hpp"
#include "io/async_stream_vbyte_reader.hpp"

#include "set_bits.hpp"
#include "utils.hpp"


namespace em_succinct_irreducible_private {
namespace normal_mode {

template<typename text_offset_type>
void compute_small_B(
    std::uint64_t text_length,
    std::uint64_t *B,
    std::string low_pos_filename,
    std::string high_pos_filename,
    std::string C_filename,
    std::uint64_t phi_undefined_position,
    std::uint64_t &total_io_volume) {

  // Print initial message, start the
  // timer and initialize I/O volume.
  fprintf(stderr, "  Compute bitvector encoding of PLCP array: ");
  long double start = utils::wclock();
  std::uint64_t io_volume = 0;

  // Fill in the bits in B corresponding to
  // irreducible LCP values. First, handle
  // the low positions (that are < text_length).
  for (std::uint64_t part = 0; part < 2; ++part) {

    // Initialize the reader of positions.
    typedef async_stream_reader<text_offset_type> pos_reader_type;
    std::string pos_filename = (!part) ? low_pos_filename : high_pos_filename;
    pos_reader_type *pos_reader = new pos_reader_type(pos_filename);
    std::uint64_t count =
      utils::file_size(pos_filename) / sizeof(text_offset_type);
    std::uint64_t offset = (!part) ? 0 : text_length;

    // Allocate buffers.
    static const std::uint64_t buffer_size = (1UL << 20);
    text_offset_type *read_buf =
      utils::allocate_array<text_offset_type>(buffer_size);
    std::uint64_t *pos_buf =
      utils::allocate_array<std::uint64_t>(buffer_size);

#ifdef _OPENMP
    std::uint64_t *tempbuf =
      utils::allocate_array<std::uint64_t>(buffer_size);
#endif

    // Stream and set bits inside B.
    {
      std::uint64_t items_processed = 0;
      while (items_processed < count) {
        std::uint64_t filled =
          std::min(count - items_processed, buffer_size);
        pos_reader->read(read_buf, filled);

#ifdef _OPENMP
        #pragma omp parallel for
        for (std::uint64_t j = 0; j < filled; ++j)
          pos_buf[j] = (std::uint64_t)read_buf[j] + offset;
#else
        for (std::uint64_t j = 0; j < filled; ++j)
          pos_buf[j] = (std::uint64_t)read_buf[j] + offset;
#endif

#ifdef _OPENMP
        set_bits(B, 2UL * text_length, pos_buf, filled, tempbuf);
#else
        for (std::uint64_t j = 0; j < filled; ++j) {
          std::uint64_t idx = pos_buf[j];
          B[idx >> 6] |= (1UL << (idx & 63));
        }
#endif

        // Update the counter.
        items_processed += filled;
      }
    }

    // Stop I/O threads.
    pos_reader->stop_reading();

    // Update I/O volume.
    io_volume +=
      pos_reader->bytes_read();

    // Clean up.
#ifdef _OPENMP
    utils::deallocate(tempbuf);
#endif

    // Clean up.
    utils::deallocate(pos_buf);
    utils::deallocate(read_buf);
    delete pos_reader;
    utils::file_delete(pos_filename);
  }

  // Handle special case.
  {
    std::uint64_t idx = 2 * phi_undefined_position;
    B[idx >> 6] |= (1UL << (idx & 63));
  }

  // Fill in reducible LCP values.
  {

    // Initialize reader of C.
    typedef async_stream_reader<std::uint64_t> C_reader_type;
    C_reader_type *C_reader = new C_reader_type(C_filename);

    // Initialize the bit-buffer for reader of C.
    std::uint64_t bitbuf = C_reader->read();
    std::uint64_t bitbuf_pos = 0;
    bool C_bit = (bitbuf & (1UL << (bitbuf_pos++)));

    // Add reducible bits.
    for (std::uint64_t j = 0; j < 2UL * text_length; ++j) {

      // Set the bit in B.
      if (C_bit == 0)
        B[j >> 6] |= (1UL << (j & 63));

      // Read the next bit from C.
      if (B[j >> 6] & (1UL << (j & 63))) {
        if (bitbuf_pos < 64 || C_reader->empty() == false) {
          if (bitbuf_pos == 64) {
            bitbuf = C_reader->read();
            bitbuf_pos = 0;
          }
          C_bit = (bitbuf & (1UL << (bitbuf_pos++)));
        }
      }
    }

    // Stop I/O threads.
    C_reader->stop_reading();

    // Update I/O volume.
    io_volume +=
      C_reader->bytes_read();

    // Clean up.
    delete C_reader;
    utils::file_delete(C_filename);
  }

  // Update I/O volume.
  total_io_volume += io_volume;

  // Print summary.
  long double elapsed = utils::wclock() - start;
  fprintf(stderr, "time = %.2Lfs, I/O = %.2LfMiB/s, "
      "total I/O vol = %.2Lfbytes/input symbol\n",
      elapsed, ((1.L * io_volume) / (1L << 20)) / elapsed,
      (1.L * total_io_volume) / text_length);
}

// Compute the B bitvector. We assume
// that B does not fit in RAM.
template<typename text_offset_type>
void compute_large_B(
    std::uint64_t text_length,
    std::uint64_t max_block_size_B,
    std::uint64_t phi_undefined_position,
    std::string B_filename,
    std::string C_filename,
    std::string *irreducible_bits_filenames,
    std::uint64_t &total_io_volume) {

  // Initialize basic parameters.
  std::uint64_t n_blocks_B =
    (2UL * text_length + max_block_size_B - 1) / max_block_size_B;

  // Print initial message and start the timer.
  fprintf(stderr, "  Compute bitvector encoding of PLCP array: ");
  long double start = utils::wclock();

  // Initialize reader of C.
  typedef async_stream_reader<std::uint64_t> C_reader_type;
  C_reader_type *C_reader = new C_reader_type(C_filename);

  // Initialize the bit-buffer for reader of C.
  std::uint64_t bitbuf = C_reader->read();
  std::uint64_t bitbuf_pos = 0;
  bool C_bit = (bitbuf & (1UL << (bitbuf_pos++)));

  std::uint64_t io_vol = 0;
  std::uint64_t max_block_size_B_in_words = max_block_size_B / 64;
  std::uint64_t *B =
    utils::allocate_array<std::uint64_t>(max_block_size_B_in_words);
  std::FILE *f = utils::file_open(B_filename, "w");

  // Allocate the buffer.
  static const std::uint64_t buffer_size = (1UL << 20);
  text_offset_type *buf =
    utils::allocate_array<text_offset_type>(buffer_size);
#ifdef _OPENMP
  text_offset_type *tempbuf =
    utils::allocate_array<text_offset_type>(buffer_size);
#endif

  for (std::uint64_t block_id = 0; block_id < n_blocks_B; ++block_id) {
    std::uint64_t block_beg = block_id * max_block_size_B;
    std::uint64_t block_end =
      std::min(block_beg + max_block_size_B, 2 * text_length);
    std::uint64_t block_size = block_end - block_beg;
    std::uint64_t block_size_in_words = (block_size + 63) / 64;

    // Zero-initialize the block of B.
    std::fill(B, B + block_size_in_words, 0UL);

    // Initialize the reader of irreducible positions.
    typedef async_stream_reader<text_offset_type>
      irreducible_bits_reader_type;
    irreducible_bits_reader_type *irreducible_bits_reader =
      new irreducible_bits_reader_type(irreducible_bits_filenames[block_id]);

    // Read and set the bits in the block of B.
    std::uint64_t count =
      utils::file_size(irreducible_bits_filenames[block_id]) /
      sizeof(text_offset_type);

    {
      std::uint64_t items_processed = 0;
      while (items_processed < count) {
        std::uint64_t filled =
          std::min(count - items_processed, buffer_size);
        irreducible_bits_reader->read(buf, filled);

#ifdef _OPENMP
        set_bits(B, block_size, buf, filled, tempbuf);
#else
        for (std::uint64_t j = 0; j < filled; ++j) {
          std::uint64_t offset = buf[j];
          B[offset >> 6] |= (1UL << (offset & 63));
        }
#endif

        items_processed += filled;
      }
    }

    // Special case for 1-bit corresponding to PLCP[SA[0]].
    if (block_beg <= 2 * phi_undefined_position &&
        2 * phi_undefined_position < block_end) {
      std::uint64_t offset = 2 * phi_undefined_position - block_beg;
      B[offset >> 6] |= (1UL << (offset & 63));
    }

    // Add reducible bits.
    for (std::uint64_t j = 0; j < block_size; ++j) {

      // Set the bit in B.
      if (C_bit == 0)
        B[j >> 6] |= (1UL << (j & 63));

      // Read the next bit from C.
      if (B[j >> 6] & (1UL << (j & 63))) {
        if (bitbuf_pos < 64 || C_reader->empty() == false) {
          if (bitbuf_pos == 64) {
            bitbuf = C_reader->read();
            bitbuf_pos = 0;
          }
          C_bit = (bitbuf & (1UL << (bitbuf_pos++)));
        }
      }
    }

    // Write current block of B to file.
    utils::write_to_file(B, block_size_in_words, f);

    // Stop I/O threads.
    irreducible_bits_reader->stop_reading();

    // Update I/O volume.
    io_vol +=
      irreducible_bits_reader->bytes_read() +
      block_size_in_words * sizeof(std::uint64_t);

    // Clean up.
    delete irreducible_bits_reader;
    utils::file_delete(irreducible_bits_filenames[block_id]);
  }

  // Stop I/O threads.
  C_reader->stop_reading();

  // Update I/O volume.
  io_vol +=
    C_reader->bytes_read();
  total_io_volume += io_vol;

  // Clean up.
#ifdef _OPENMP
  utils::deallocate(tempbuf);
#endif
  utils::deallocate(buf);
  utils::deallocate(B);
  delete C_reader;
  std::fclose(f);
  utils::file_delete(C_filename);

  // Print summary.
  long double elapsed = utils::wclock() - start;
  fprintf(stderr, "time = %.2Lfs, I/O = %.2LfMiB/s, "
      "total I/O vol = %.2Lfbytes/input symbol\n",
      elapsed, ((1.L * io_vol) / (1L << 20)) / elapsed,
      (1.L * total_io_volume) / text_length);
}

template<typename char_type,
  typename text_offset_type>
std::uint64_t *compute_very_small_B(
    std::uint64_t text_length,
    std::string text_filename,
    std::string sa_filename,
    std::uint64_t &n_irreducible_lcps,
    std::uint64_t &sum_irreducible_lcps,
    std::uint64_t &total_io_volume) {

  // Initialize basic parameters.
  std::uint64_t local_n_irreducible_lcps = 0;
  std::uint64_t local_sum_irreducible_lcps = 0;

  // Allocate bitvectors.
  std::uint64_t B_size_in_words = (2UL * text_length + 63) / 64;
  std::uint64_t C_size_in_words = (text_length + 63) / 64;
  std::uint64_t *B = utils::allocate_array<std::uint64_t>(B_size_in_words);
  std::uint64_t *C = utils::allocate_array<std::uint64_t>(C_size_in_words);
  std::fill(B, B + B_size_in_words, (std::uint64_t)0);
  std::fill(C, C + C_size_in_words, (std::uint64_t)0);

  // Read text.
  char_type *text = utils::allocate_array<char_type>(text_length);
  {

    // Start the timer.
    fprintf(stderr, "  Read text: ");
    long double read_start = utils::wclock();
    std::uint64_t io_volume = 0;

    // Read data.
    utils::read_from_file(text, text_length, text_filename);

    // Update I/O volume.
    io_volume += text_length * sizeof(char_type);
    total_io_volume += io_volume;

    // Print summary.
    long double read_time = utils::wclock() - read_start;
    fprintf(stderr, "time = %.2Lfs, I/O = %.2LfMiB/s, "
        "total I/O vol = %.2Lfbytes/input symbol\n",
        read_time, ((1.L * io_volume) / (1L << 20)) / read_time,
        (1.L * total_io_volume) / text_length);
  }

  // Compute irreducible lcp values.
  {

    // Start the timer.
    fprintf(stderr, "  Compute irreducible LCP values: ");
    long double compute_irr_lcp_start = utils::wclock();

    // Initialize basic statistics.
    std::uint64_t io_volume = 0;

    // Initialize SA reader.
    typedef async_stream_reader<text_offset_type> sa_reader_type;
    sa_reader_type *sa_reader = new sa_reader_type(sa_filename);

    // Allocate buffers.
    static const std::uint64_t buf_size = (1UL << 20);
    text_offset_type *sa_buf =
      utils::allocate_array<text_offset_type>(buf_size);
    char_type *bwt_buf = utils::allocate_array<char_type>(buf_size);

#ifdef _OPENMP
    std::uint64_t *pair_buf =
      utils::allocate_array<std::uint64_t>(buf_size * 2);
    std::uint64_t *ans_buf_B =
      utils::allocate_array<std::uint64_t>(buf_size);
    std::uint64_t *ans_buf_C =
      utils::allocate_array<std::uint64_t>(buf_size);
#endif

    // Processing of SA follows.
    char_type prev_bwt = (char_type)0;
    std::uint64_t sa_items_read = 0;
    std::uint64_t prev_sa = text_length;
    while (sa_items_read < text_length) {
      std::uint64_t buf_filled =
        std::min(buf_size, text_length - sa_items_read);
      sa_reader->read(sa_buf, buf_filled);

      // Compute BWT buffer.
#ifdef _OPENMP
      #pragma omp parallel for
      for (std::uint64_t j = 0; j < buf_filled; ++j) {
        std::uint64_t addr = (std::uint64_t)sa_buf[j];
        if (addr > 0) bwt_buf[j] = text[addr - 1];
      }
#else
      for (std::uint64_t j = 0; j < buf_filled; ++j) {
        std::uint64_t addr = (std::uint64_t)sa_buf[j];
        if (addr > 0) bwt_buf[j] = text[addr - 1];
      }
#endif

      // Process buffer.
#ifdef _OPENMP
      {

        // Bring the irreducible pairs together.
        std::uint64_t buf_irr_filled = 0;
        for (std::uint64_t j = 0; j < buf_filled; ++j) {
          std::uint64_t cur_sa = (std::uint64_t)sa_buf[j];
          char_type cur_bwt = bwt_buf[j];
          if ((sa_items_read == 0 && j == 0) ||
              (cur_sa == 0) ||
              (prev_sa == 0) ||
              (cur_bwt != prev_bwt)) {

            // Current pair is irreducible.
            pair_buf[2 * buf_irr_filled] = cur_sa;
            pair_buf[2 * buf_irr_filled + 1] = prev_sa;
            ++buf_irr_filled;
          }

          prev_sa = cur_sa;
          prev_bwt = cur_bwt;
        }

        // Update statistics.
        local_n_irreducible_lcps += buf_irr_filled;

        if (buf_irr_filled > 0) {

          // Compute lcp values in parallel.
          #pragma omp parallel
          {
            std::uint64_t thread_sum_irreducible_lcps = 0;

            #pragma omp for nowait
            for (std::uint64_t j = 0; j < buf_irr_filled; ++j) {
              std::uint64_t i = pair_buf[2 * j];
              std::uint64_t phi_i = pair_buf[2 * j + 1];
              std::uint64_t lcp = 0;

              while (i + lcp < text_length &&
                  phi_i + lcp < text_length &&
                  text[i + lcp] == text[phi_i + lcp])
                ++lcp;

              thread_sum_irreducible_lcps += lcp;
              ans_buf_C[j] = i;
              ans_buf_B[j] = 2 * i + lcp;
            }

            #pragma omp critical
            {
              local_sum_irreducible_lcps += thread_sum_irreducible_lcps;
            }
          }

          // Set the bits in B and C in parallel.
          set_bits(B, 2UL * text_length,
              ans_buf_B, buf_irr_filled, pair_buf);
          set_bits(C, 1UL * text_length,
              ans_buf_C, buf_irr_filled, pair_buf);
        }
      }
#else
      for (std::uint64_t j = 0; j < buf_filled; ++j) {
        std::uint64_t cur_sa = (std::uint64_t)sa_buf[j];
        char_type cur_bwt = bwt_buf[j];
        if ((sa_items_read == 0 && j == 0) ||
            (cur_sa == 0) ||
            (prev_sa == 0) ||
            (cur_bwt != prev_bwt)) {

          // Compute irreducible lcp(cur_sa, prev_sa) naively.
          std::uint64_t lcp = 0;
          while (cur_sa + lcp < text_length &&
              prev_sa + lcp < text_length &&
              text[cur_sa + lcp] == text[prev_sa + lcp])
            ++lcp;

          // Set the corresponding bits in the B and C.
          std::uint64_t bv_idx = 2UL * cur_sa + lcp;
          B[bv_idx >> 6] |= (1UL << (bv_idx & 63));
          C[cur_sa >> 6] |= (1UL << (cur_sa & 63));

          // Update statistics.
          ++local_n_irreducible_lcps;
          local_sum_irreducible_lcps += lcp;
        }

        prev_sa = cur_sa;
        prev_bwt = cur_bwt;
      }
#endif

      sa_items_read += buf_filled;
    }

    // Stop I/O threads.
    sa_reader->stop_reading();

    // Update I/O volume.
    io_volume +=
      sa_reader->bytes_read();
    total_io_volume += io_volume;

    // Clean up.
#ifdef _OPENMP
    utils::deallocate(ans_buf_C);
    utils::deallocate(ans_buf_B);
    utils::deallocate(pair_buf);
#endif
    utils::deallocate(bwt_buf);
    utils::deallocate(sa_buf);
    delete sa_reader;

    // Print summary.
    long double compute_irr_lcp_time =
      utils::wclock() - compute_irr_lcp_start;
    fprintf(stderr, "time = %.2Lfs, I/O = %.2LfMiB, "
        "total I/O vol = %.2Lfbytes/input symbol\n",
        compute_irr_lcp_time,
        ((1.L * io_volume) / (1L << 20)) / compute_irr_lcp_time,
        (1.L * total_io_volume) / text_length);
  }

  // Clean up.
  utils::deallocate(text);

  // Fill in reducible LCP values.
  {
    fprintf(stderr, "  Fill missing reducible LCP values: ");
    long double fill_in_reduc_start = utils::wclock();

    std::uint64_t B_ptr = 0;
    for (std::uint64_t j = 0; j < text_length; ++j) {
      if ((C[j >> 6] & (1UL << (j & 63))) == 0) {

        // Mark the 1-bit corresponding to reducible LCP value.
        B[B_ptr >> 6] |= (1UL << (B_ptr & 63));
      } else {

        // Find the next 1-bit in B.
        while ((B[B_ptr >> 6] & (1UL << (B_ptr & 63))) == 0)
          ++B_ptr;
      }
      ++B_ptr;
    }

    // Print summary.
    long double fill_in_reduc_time = utils::wclock() - fill_in_reduc_start;
    fprintf(stderr, "time = %.2Lfs\n", fill_in_reduc_time);
  }

  // Clean up.
  utils::deallocate(C);

  // Update reference variables.
  n_irreducible_lcps = local_n_irreducible_lcps;
  sum_irreducible_lcps = local_sum_irreducible_lcps;

  // Return the pointer to B.
  return B;
}

}  // namespace normal_mode

namespace inplace_mode {

template<typename text_offset_type>
void permute_bitvector_positions(
    std::uint64_t text_length,
    std::uint64_t max_block_size_B,
    std::uint64_t max_halfsegment_size,
    std::string **pos_filenames,
    std::string **lcp_filenames,
    std::string *irreducible_bits_filenames,
    std::uint64_t &total_io_volume) {

  // Print initial message and start the timer.
  fprintf(stderr, "    Permute bitvector positions: ");
  long double start = utils::wclock();
  std::uint64_t io_volume = 0;

  // Compute basic parameters.
  std::uint64_t n_blocks_B =
    (2UL * text_length + max_block_size_B - 1) / max_block_size_B;
  std::uint64_t n_halfsegments =
    (text_length + max_halfsegment_size - 1) / max_halfsegment_size;

  // Initialize multiwriter of positions 2i + PLCP[i].
  typedef async_multi_stream_writer<text_offset_type>
    lcp_multiwriter_type;
  lcp_multiwriter_type *lcp_multiwriter = NULL;
  {
    static const std::uint64_t n_free_buffers = 4;
    std::uint64_t buffer_size = (1UL << 20);
    lcp_multiwriter =
      new lcp_multiwriter_type(n_blocks_B, buffer_size, n_free_buffers);
    for (std::uint64_t block_id = 0; block_id < n_blocks_B; ++block_id)
      lcp_multiwriter->add_file(irreducible_bits_filenames[block_id]);
  }

  // Allocate buffers.
  static const std::uint64_t local_buf_size = (1UL << 20);
  text_offset_type *pos_buf =
    utils::allocate_array<text_offset_type>(local_buf_size);
  text_offset_type *lcp_buf =
    utils::allocate_array<text_offset_type>(local_buf_size);

  // Processing of halfsegment pairs follows.
  for (std::uint64_t left_halfsegment_id = 0;
      left_halfsegment_id < n_halfsegments; ++left_halfsegment_id) {

    // Scan all halfsegments to the right of left_halfsegment_id.
    for (std::uint64_t right_halfsegment_id = left_halfsegment_id;
        right_halfsegment_id < n_halfsegments; right_halfsegment_id++) {

      // Check if that pair of halfsegments has any associated pairs.
      std::string pos_filename =
        pos_filenames[left_halfsegment_id][right_halfsegment_id];
      std::string lcp_filename =
        lcp_filenames[left_halfsegment_id][right_halfsegment_id];
      if (utils::file_exists(pos_filename) == false ||
          utils::file_size(pos_filename) == 0) {
        if (utils::file_exists(pos_filename))
          utils::file_delete(pos_filename);
        if (utils::file_exists(lcp_filename))
          utils::file_delete(lcp_filename);
        continue;
      }

      // Get the number of pairs to process.
      std::uint64_t n_pairs =
        utils::file_size(pos_filename) /
        sizeof(text_offset_type);

      // Create the reader of pos values.
      typedef async_stream_reader<text_offset_type> pos_reader_type;
      pos_reader_type *pos_reader =
        new pos_reader_type(pos_filename);

      // Create the reader of lcp values.
      typedef async_stream_vbyte_reader lcp_reader_type;
      lcp_reader_type *lcp_reader =
        new lcp_reader_type(lcp_filename);

      // Process all pairs.
      std::uint64_t pairs_processed = 0;
      while (pairs_processed < n_pairs) {

        // Read next buffer.
        std::uint64_t filled =
          std::min(n_pairs - pairs_processed, local_buf_size);
        pos_reader->read(pos_buf, filled);
        for (std::uint64_t j = 0; j < filled; ++j)
          lcp_buf[j] = lcp_reader->read();

        // Process the buffer.
        for (std::uint64_t i = 0; i < filled; ++i) {
          std::uint64_t pos = pos_buf[i];
          std::uint64_t lcp = lcp_buf[i];

          // Compute answer.
          std::uint64_t pos_B = 2UL * pos + lcp;
          std::uint64_t block_id_B = pos_B / max_block_size_B;
          std::uint64_t block_beg_B = block_id_B * max_block_size_B;
          std::uint64_t offset_B = pos_B - block_beg_B;

          // Write the answer to file.
          lcp_multiwriter->write_to_ith_file(block_id_B, offset_B);
        }

        // Update the number of processed pairs.
        pairs_processed += filled;
      }

      // Stop I/O threads.
      pos_reader->stop_reading();
      lcp_reader->stop_reading();

      // Update I/O volume.
      io_volume +=
        pos_reader->bytes_read() +
        lcp_reader->bytes_read();

      // Clean up.
      delete lcp_reader;
      delete pos_reader;
      utils::file_delete(pos_filename);
      utils::file_delete(lcp_filename);
    }
  }

  // Update I/O volume.
  io_volume += lcp_multiwriter->bytes_written();
  total_io_volume += io_volume;

  // Clean up.
  utils::deallocate(lcp_buf);
  utils::deallocate(pos_buf);
  delete lcp_multiwriter;

  // Print summary.
  long double elapsed = utils::wclock() - start;
  fprintf(stderr, "time = %.2Lfs, I/O = %.2LfMiB/s, "
      "total I/O vol = %.2Lfbytes/input symbol\n",
      elapsed, ((1.L * io_volume) / (1L << 20)) / elapsed,
      (1.L * total_io_volume) / text_length);
}

// Compute the B bitvector. We assume
// that B fits in RAM, but not B and C.
template<typename text_offset_type>
void compute_small_B(
    std::uint64_t text_length,
    std::uint64_t max_halfsegment_size,
    std::uint64_t *B,
    std::string **pos_filenames,
    std::string **lcp_filenames,
    std::string C_filename,
    std::uint64_t phi_undefined_position,
    bool is_last_part,
    std::uint64_t &total_io_volume) {

  // Print initial message, start the
  // timer and initialize I/O volume.
  fprintf(stderr, "    Compute bitvector encoding of PLCP array: ");
  long double start = utils::wclock();
  std::uint64_t io_volume = 0;

  // Compute basic parameters.
  std::uint64_t n_halfsegments =
    (text_length + max_halfsegment_size - 1) / max_halfsegment_size;

  // Allocate buffers.
  static const std::uint64_t local_buf_size = (1UL << 20);
  text_offset_type *pos_buf =
    utils::allocate_array<text_offset_type>(local_buf_size);
  text_offset_type *lcp_buf =
    utils::allocate_array<text_offset_type>(local_buf_size);
  std::uint64_t *buf =
    utils::allocate_array<std::uint64_t>(local_buf_size);

#ifdef _OPENMP
  std::uint64_t *tempbuf =
    utils::allocate_array<std::uint64_t>(local_buf_size);
#endif

  // Fill in the bits in B corresponding
  // to irreducible lcp values. 
  for (std::uint64_t left_halfsegment_id = 0;
      left_halfsegment_id < n_halfsegments; ++left_halfsegment_id) {

    // Scan all halfsegments to the right of left_halfsegment_id.
    for (std::uint64_t right_halfsegment_id = left_halfsegment_id;
        right_halfsegment_id < n_halfsegments; right_halfsegment_id++) {

      // Check if that pair of halfsegments has any associated pairs.
      std::string pos_filename =
        pos_filenames[left_halfsegment_id][right_halfsegment_id];
      std::string lcp_filename =
        lcp_filenames[left_halfsegment_id][right_halfsegment_id];
      if (utils::file_exists(pos_filename) == false ||
          utils::file_size(pos_filename) == 0) {
        if (utils::file_exists(pos_filename))
          utils::file_delete(pos_filename);
        if (utils::file_exists(lcp_filename))
          utils::file_delete(lcp_filename);
        continue;
      }

      // Get the number of pairs to process.
      std::uint64_t n_pairs =
        utils::file_size(pos_filename) /
        sizeof(text_offset_type);

      // Create the reader of pos values.
      typedef async_stream_reader<text_offset_type> pos_reader_type;
      pos_reader_type *pos_reader =
        new pos_reader_type(pos_filename);

      // Create the reader of lcp values.
      typedef async_stream_vbyte_reader lcp_reader_type;
      lcp_reader_type *lcp_reader =
        new lcp_reader_type(lcp_filename);

      // Process all pairs.
      std::uint64_t pairs_processed = 0;
      while (pairs_processed < n_pairs) {

        // Read next buffer.
        std::uint64_t filled =
          std::min(n_pairs - pairs_processed, local_buf_size);
        pos_reader->read(pos_buf, filled);
        for (std::uint64_t j = 0; j < filled; ++j)
          lcp_buf[j] = lcp_reader->read();

        // Process the buffer.
#ifdef _OPENMP
        #pragma omp parallel for
        for (std::uint64_t i = 0; i < filled; ++i)
          buf[i] =
            (std::uint64_t)pos_buf[i] * 2UL+
            (std::uint64_t)lcp_buf[i];
#else
       for (std::uint64_t i = 0; i < filled; ++i)
          buf[i] =
            (std::uint64_t)pos_buf[i] * 2UL +
            (std::uint64_t)lcp_buf[i];
#endif

#ifdef _OPENMP
        set_bits(B, 2UL * text_length, buf, filled, tempbuf);
#else
        for (std::uint64_t j = 0; j < filled; ++j) {
          std::uint64_t idx = buf[j];
          B[idx >> 6] |= (1UL << (idx & 63));
        }
#endif

        // Update the number of processed pairs.
        pairs_processed += filled;
      }

      // Stop I/O threads.
      pos_reader->stop_reading();
      lcp_reader->stop_reading();

      // Update I/O volume.
      io_volume +=
        pos_reader->bytes_read() +
        lcp_reader->bytes_read();

      // Clean up.
      delete lcp_reader;
      delete pos_reader;
      utils::file_delete(pos_filename);
      utils::file_delete(lcp_filename);
    }
  }

  // Handle special case.
  {
    std::uint64_t idx = 2 * phi_undefined_position;
    B[idx >> 6] |= (1UL << (idx & 63));
  }

  // Clean up.
#ifdef _OPENMP
  utils::deallocate(tempbuf);
#endif

  utils::deallocate(buf);
  utils::deallocate(lcp_buf);
  utils::deallocate(pos_buf);

  // Fill in reducible LCP values.
  if (is_last_part) {

    // Initialize reader of C.
    typedef async_stream_reader<std::uint64_t> C_reader_type;
    C_reader_type *C_reader = new C_reader_type(C_filename);

    // Initialize the bit-buffer for reader of C.
    std::uint64_t bitbuf = C_reader->read();
    std::uint64_t bitbuf_pos = 0;
    bool C_bit = (bitbuf & (1UL << (bitbuf_pos++)));

    // Add reducible bits.
    for (std::uint64_t j = 0; j < 2UL * text_length; ++j) {

      // Set the bit in B.
      if (C_bit == 0)
        B[j >> 6] |= (1UL << (j & 63));

      // Read the next bit from C.
      if (B[j >> 6] & (1UL << (j & 63))) {
        if (bitbuf_pos < 64 || C_reader->empty() == false) {
          if (bitbuf_pos == 64) {
            bitbuf = C_reader->read();
            bitbuf_pos = 0;
          }
          C_bit = (bitbuf & (1UL << (bitbuf_pos++)));
        }
      }
    }

    // Stop I/O threads.
    C_reader->stop_reading();

    // Update I/O volume.
    io_volume +=
      C_reader->bytes_read();

    // Clean up.
    delete C_reader;
    utils::file_delete(C_filename);
  }

  // Update I/O volume.
  total_io_volume += io_volume;

  // Print summary.
  long double elapsed = utils::wclock() - start;
  fprintf(stderr, "time = %.2Lfs, I/O = %.2LfMiB/s, "
      "total I/O vol = %.2Lfbytes/input symbol\n",
      elapsed, ((1.L * io_volume) / (1L << 20)) / elapsed,
      (1.L * total_io_volume) / text_length);
}

// Compute the B bitvector. We assume
// that B does not fit in RAM.
template<typename text_offset_type>
void compute_large_B(
    std::uint64_t text_length,
    std::uint64_t max_block_size_B,
    std::uint64_t phi_undefined_position,
    std::string B_filename,
    std::string C_filename,
    std::string *irreducible_bits_filenames,
    bool is_last_part,
    std::uint64_t &total_io_volume) {

  // Print initial message and start the timer.
  fprintf(stderr, "    Compute bitvector encoding of PLCP array: ");
  long double start = utils::wclock();
  std::uint64_t io_volume = 0;

  // Compute basic parameters.
  std::uint64_t B_length = 2 * text_length;
  std::uint64_t n_blocks_B =
    (B_length + max_block_size_B - 1) /
    max_block_size_B;
  std::uint64_t max_block_size_B_in_words =
    max_block_size_B / 64;

  // Create the reader of C.
  typedef async_stream_reader<std::uint64_t>
    C_reader_type;
  C_reader_type *C_reader = NULL;

  std::uint64_t bitbuf = 0;
  std::uint64_t bitbuf_pos = 0;
  bool C_bit = false;

  if (is_last_part) {
    C_reader = new C_reader_type(C_filename);
    bitbuf = C_reader->read();
    bitbuf_pos = 0;
    C_bit = (bitbuf & (1UL << (bitbuf_pos++)));
  }

  // Allocate the block of B.
  std::uint64_t *B =
    utils::allocate_array<std::uint64_t>(max_block_size_B_in_words);

  // Check if file with B already exists. This is
  // to determine if we are processing the first part.
  bool B_exists = utils::file_exists(B_filename);

  // Open the file to write B.
  std::FILE *f = NULL;
  if (B_exists) f = utils::file_open(B_filename, "r+");
  else f = utils::file_open(B_filename, "w");

  // Allocate buffers.
  static const std::uint64_t buffer_size = ((std::uint64_t)1 << 20);
  text_offset_type *buf =
    utils::allocate_array<text_offset_type>(buffer_size);

#ifdef _OPENMP
  text_offset_type *tempbuf =
    utils::allocate_array<text_offset_type>(buffer_size);
#endif

  // Process all block of B.
  for (std::uint64_t block_id = 0; block_id < n_blocks_B; ++block_id) {
    std::uint64_t block_beg = block_id * max_block_size_B;
    std::uint64_t block_end =
      std::min(block_beg + max_block_size_B, 2 * text_length);
    std::uint64_t block_size = block_end - block_beg;
    std::uint64_t block_size_in_words = (block_size + 63) / 64;

    if (B_exists) {

      // Read block of B from disk for update.
      std::uint64_t offset = block_beg / 8;
      utils::read_at_offset(B, offset,
          block_size_in_words, f);
    } else {

      // Zero-initialize the block of B.
      std::fill(B, B + block_size_in_words,
          (std::uint64_t)0);
    }

    // Initialize the reader of irreducible positions.
    typedef async_stream_reader<text_offset_type>
      irreducible_bits_reader_type;
    irreducible_bits_reader_type *irreducible_bits_reader =
      new irreducible_bits_reader_type(
          irreducible_bits_filenames[block_id]);

    // Read and set the bits in the block of B.
    std::uint64_t count =
      utils::file_size(irreducible_bits_filenames[block_id]) /
      sizeof(text_offset_type);

    {
      std::uint64_t items_processed = 0;
      while (items_processed < count) {
        std::uint64_t filled =
          std::min(count - items_processed, buffer_size);
        irreducible_bits_reader->read(buf, filled);

#ifdef _OPENMP
        set_bits(B, block_size, buf, filled, tempbuf);
#else
        for (std::uint64_t j = 0; j < filled; ++j) {
          std::uint64_t offset = buf[j];
          B[offset >> 6] |= ((std::uint64_t)1 << (offset & 63));
        }
#endif

        items_processed += filled;
      }
    }

    // Special case for 1-bit corresponding to PLCP[SA[0]].
    if (block_beg <= 2 * phi_undefined_position &&
        2 * phi_undefined_position < block_end) {
      std::uint64_t offset = 2 * phi_undefined_position - block_beg;
      B[offset >> 6] |= ((std::uint64_t)1 << (offset & 63));
    }

    // Add reducible bits.
    if (is_last_part) {
      for (std::uint64_t j = 0; j < block_size; ++j) {

        // Set the bit in B.
        if (C_bit == 0)
          B[j >> 6] |= ((std::uint64_t)1 << (j & 63));

        // Read the next bit from C.
        if (B[j >> 6] & ((std::uint64_t)1 << (j & 63))) {

          // Refill the buffer if necessary.
          if (bitbuf_pos < 64 || C_reader->empty() == false) {
            if (bitbuf_pos == 64) {
              bitbuf = C_reader->read();
              bitbuf_pos = 0;
            }

            // Set the next bit.
            C_bit = (bitbuf & ((std::uint64_t)1 << (bitbuf_pos++)));
          }
        }
      }
    }

    // Write current block of B to file.
    if (B_exists)
      utils::overwrite_at_offset(B, block_beg >> 6, block_size_in_words, f);
    else utils::write_to_file(B, block_size_in_words, f);

    // Stop I/O threads.
    irreducible_bits_reader->stop_reading();

    // Update I/O volume.
    io_volume +=
      irreducible_bits_reader->bytes_read() +
      block_size_in_words * sizeof(std::uint64_t);

    // Account for the reading of B.
    if (B_exists)
      io_volume +=
        block_size_in_words * sizeof(std::uint64_t);

    // Clean up.
    delete irreducible_bits_reader;
    utils::file_delete(irreducible_bits_filenames[block_id]);
  }

  // Update I/O volume.
  if (is_last_part) {
    C_reader->stop_reading();
    io_volume +=
      C_reader->bytes_read();
  }
  total_io_volume += io_volume;

  // Clean up.
#ifdef _OPENMP
  utils::deallocate(tempbuf);
#endif

  utils::deallocate(buf);
  utils::deallocate(B);
  std::fclose(f);

  if (is_last_part) {
    delete C_reader;
    utils::file_delete(C_filename);
  }

  // Print summary.
  long double elapsed = utils::wclock() - start;
  fprintf(stderr, "time = %.2Lfs, I/O = %.2LfMiB/s, "
      "total I/O vol = %.2Lfbytes/input symbol\n",
      elapsed, ((1.L * io_volume) / (1L << 20)) / elapsed,
      (1.L * total_io_volume) / text_length);
}

template<typename char_type,
  typename text_offset_type>
std::uint64_t *compute_very_small_B(
    std::uint64_t text_length,
    std::string text_filename,
    std::string sa_filename,
    std::uint64_t &n_irreducible_lcps,
    std::uint64_t &sum_irreducible_lcps,
    std::uint64_t &total_io_volume) {

  // Initialize basic parameters.
  std::uint64_t local_n_irreducible_lcps = 0;
  std::uint64_t local_sum_irreducible_lcps = 0;

  // Allocate bitvectors.
  std::uint64_t B_size_in_words = ((std::uint64_t)2 * text_length + 63) / 64;
  std::uint64_t C_size_in_words = (text_length + 63) / 64;
  std::uint64_t *B = utils::allocate_array<std::uint64_t>(B_size_in_words);
  std::uint64_t *C = utils::allocate_array<std::uint64_t>(C_size_in_words);
  std::fill(B, B + B_size_in_words, (std::uint64_t)0);
  std::fill(C, C + C_size_in_words, (std::uint64_t)0);

  // Read text.
  char_type *text = utils::allocate_array<char_type>(text_length);
  {

    // Start the timer.
    fprintf(stderr, "  Read text: ");
    long double read_start = utils::wclock();
    std::uint64_t io_volume = 0;

    // Read data.
    utils::read_from_file(text, text_length, text_filename);

    // Update I/O volume.
    io_volume += text_length * sizeof(char_type);
    total_io_volume += io_volume;

    // Print summary.
    long double read_time = utils::wclock() - read_start;
    fprintf(stderr, "time = %.2Lfs, I/O = %.2LfMiB/s, "
        "total I/O vol = %.2Lfbytes/input symbol\n",
        read_time, ((1.L * io_volume) / (1L << 20)) / read_time,
        (1.L * total_io_volume) / text_length);
  }

  // Compute irreducible lcp values.
  {

    // Start the timer.
    fprintf(stderr, "  Compute irreducible LCP values: ");
    long double compute_irr_lcp_start = utils::wclock();

    // Initialize basic statistics.
    std::uint64_t io_volume = 0;

    // Initialize SA reader.
    typedef async_stream_reader<text_offset_type> sa_reader_type;
    sa_reader_type *sa_reader = new sa_reader_type(sa_filename);

    // Allocate buffers.
    static const std::uint64_t buf_size = ((std::uint64_t)1 << 20);
    text_offset_type *sa_buf =
      utils::allocate_array<text_offset_type>(buf_size);
    char_type *bwt_buf = utils::allocate_array<char_type>(buf_size);

#ifdef _OPENMP
    std::uint64_t *pair_buf =
      utils::allocate_array<std::uint64_t>(buf_size * 2);
    std::uint64_t *ans_buf_B =
      utils::allocate_array<std::uint64_t>(buf_size);
    std::uint64_t *ans_buf_C =
      utils::allocate_array<std::uint64_t>(buf_size);
#endif

    // Processing of SA follows.
    char_type prev_bwt = (char_type)0;
    std::uint64_t sa_items_read = 0;
    std::uint64_t prev_sa = text_length;
    while (sa_items_read < text_length) {
      std::uint64_t buf_filled =
        std::min(buf_size, text_length - sa_items_read);
      sa_reader->read(sa_buf, buf_filled);

      // Compute BWT buffer.
#ifdef _OPENMP
      #pragma omp parallel for
      for (std::uint64_t j = 0; j < buf_filled; ++j) {
        std::uint64_t addr = (std::uint64_t)sa_buf[j];
        if (addr > 0) bwt_buf[j] = text[addr - 1];
      }
#else
      for (std::uint64_t j = 0; j < buf_filled; ++j) {
        std::uint64_t addr = (std::uint64_t)sa_buf[j];
        if (addr > 0) bwt_buf[j] = text[addr - 1];
      }
#endif

      // Process buffer.
#ifdef _OPENMP
      {

        // Bring the irreducible pairs together.
        std::uint64_t buf_irr_filled = 0;
        for (std::uint64_t j = 0; j < buf_filled; ++j) {
          std::uint64_t cur_sa = (std::uint64_t)sa_buf[j];
          char_type cur_bwt = bwt_buf[j];
          if ((sa_items_read == 0 && j == 0) ||
              (cur_sa == 0) ||
              (prev_sa == 0) ||
              (cur_bwt != prev_bwt)) {

            // Current pair is irreducible.
            pair_buf[2 * buf_irr_filled] = cur_sa;
            pair_buf[2 * buf_irr_filled + 1] = prev_sa;
            ++buf_irr_filled;
          }

          prev_sa = cur_sa;
          prev_bwt = cur_bwt;
        }

        // Update statistics.
        local_n_irreducible_lcps += buf_irr_filled;

        if (buf_irr_filled > 0) {

          // Compute lcp values in parallel.
          #pragma omp parallel
          {
            std::uint64_t thread_sum_irreducible_lcps = 0;

            #pragma omp for nowait
            for (std::uint64_t j = 0; j < buf_irr_filled; ++j) {
              std::uint64_t i = pair_buf[2 * j];
              std::uint64_t phi_i = pair_buf[2 * j + 1];
              std::uint64_t lcp = 0;

              while (i + lcp < text_length &&
                  phi_i + lcp < text_length &&
                  text[i + lcp] == text[phi_i + lcp])
                ++lcp;

              thread_sum_irreducible_lcps += lcp;
              ans_buf_C[j] = i;
              ans_buf_B[j] = 2 * i + lcp;
            }

            #pragma omp critical
            {
              local_sum_irreducible_lcps += thread_sum_irreducible_lcps;
            }
          }

          // Set the bits in B and C in parallel.
          set_bits(B, 2UL * text_length,
              ans_buf_B, buf_irr_filled, pair_buf);
          set_bits(C, 1UL * text_length,
              ans_buf_C, buf_irr_filled, pair_buf);
        }
      }

#else
      for (std::uint64_t j = 0; j < buf_filled; ++j) {
        std::uint64_t cur_sa = (std::uint64_t)sa_buf[j];
        char_type cur_bwt = bwt_buf[j];
        if ((sa_items_read == 0 && j == 0) ||
            (cur_sa == 0) ||
            (prev_sa == 0) ||
            (cur_bwt != prev_bwt)) {

          // Compute irreducible lcp(cur_sa, prev_sa) naively.
          std::uint64_t lcp = 0;
          while (cur_sa + lcp < text_length &&
              prev_sa + lcp < text_length &&
              text[cur_sa + lcp] == text[prev_sa + lcp])
            ++lcp;

          // Set the corresponding bits in the B and C.
          std::uint64_t bv_idx = 2UL * cur_sa + lcp;
          B[bv_idx >> 6] |= ((std::uint64_t)1 << (bv_idx & 63));
          C[cur_sa >> 6] |= ((std::uint64_t)1 << (cur_sa & 63));

          // Update statistics.
          ++local_n_irreducible_lcps;
          local_sum_irreducible_lcps += lcp;
        }

        prev_sa = cur_sa;
        prev_bwt = cur_bwt;
      }
#endif

      sa_items_read += buf_filled;
    }

    // Stop I/O threads.
    sa_reader->stop_reading();

    // Update I/O volume.
    io_volume +=
      sa_reader->bytes_read();
    total_io_volume += io_volume;

    // Clean up.
#ifdef _OPENMP
    utils::deallocate(ans_buf_C);
    utils::deallocate(ans_buf_B);
    utils::deallocate(pair_buf);
#endif

    utils::deallocate(bwt_buf);
    utils::deallocate(sa_buf);
    delete sa_reader;

    // Print summary.
    long double compute_irr_lcp_time =
      utils::wclock() - compute_irr_lcp_start;
    fprintf(stderr, "time = %.2Lfs, I/O = %.2LfMiB, "
        "total I/O vol = %.2Lfbytes/input symbol\n",
        compute_irr_lcp_time,
        ((1.L * io_volume) / (1L << 20)) / compute_irr_lcp_time,
        (1.L * total_io_volume) / text_length);
  }

  // Clean up.
  utils::deallocate(text);

  // Fill in reducible LCP values.
  {
    fprintf(stderr, "  Fill missing reducible LCP values: ");
    long double fill_in_reduc_start = utils::wclock();

    std::uint64_t B_ptr = 0;
    for (std::uint64_t j = 0; j < text_length; ++j) {
      if ((C[j >> 6] & ((std::uint64_t)1 << (j & 63))) == 0) {

        // Mark the 1-bit corresponding to reducible LCP value.
        B[B_ptr >> 6] |= ((std::uint64_t)1 << (B_ptr & 63));
      } else {

        // Find the next 1-bit in B.
        while ((B[B_ptr >> 6] & ((std::uint64_t)1 << (B_ptr & 63))) == 0)
          ++B_ptr;
      }
      ++B_ptr;
    }

    // Print summary.
    long double fill_in_reduc_time =
      utils::wclock() - fill_in_reduc_start;
    fprintf(stderr, "time = %.2Lfs\n", fill_in_reduc_time);
  }

  // Clean up.
  utils::deallocate(C);

  // Update reference variables.
  n_irreducible_lcps = local_n_irreducible_lcps;
  sum_irreducible_lcps = local_sum_irreducible_lcps;

  // Return the pointer to B.
  return B;
}

}  // namespace inplace_mode
}  // namespace em_succinct_irreducible_private

#endif  // __SRC_EM_SUCCINCT_IRREDUCIBLE_SRC_COMPUTE_B_HPP_INCLUDED
