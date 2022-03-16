/**
 * @file    src/em_succinct_irreducible_src/distribute_pairs_and_compute_C.hpp
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

#ifndef __SRC_EM_SUCCINCT_IRREDUCIBLE_SRC_DISTRIBUTE_PAIRS_AND_COMPUTE_C_HPP_INCLUDED
#define __SRC_EM_SUCCINCT_IRREDUCIBLE_SRC_DISTRIBUTE_PAIRS_AND_COMPUTE_C_HPP_INCLUDED

#include <cstdio>
#include <cstdint>
#include <string>
#include <algorithm>
#include <omp.h>

#include "io/async_stream_reader.hpp"
#include "io/async_multi_stream_writer.hpp"
#include "set_bits.hpp"
#include "utils.hpp"


namespace em_succinct_irreducible_private {
namespace normal_mode {

template<typename char_type,
  typename text_offset_type>
void distribute_pairs(
    std::uint64_t text_length,
    std::uint64_t max_halfsegment_size,
    std::uint64_t ram_use,
    std::string sa_filename,
    std::string bwt_filename,
    std::string **pairs_filenames,
    std::uint64_t &n_irreducible_lcps,
    std::uint64_t &total_io_volume) {

  // Print initial message.
  fprintf(stderr, "  Distribute (i, Phi[i]) pairs: ");

  // Start the timer.
  long double start = utils::wclock();

  // Compute basic parameters.
  std::uint64_t n_halfsegments =
    (text_length + max_halfsegment_size - 1) /
    max_halfsegment_size;
  std::uint64_t n_irreducible = 0;

  // Create a map from used halfsegment pairs to a contiguous
  // range of integers. This is needed to use multifile writer.
  std::uint64_t **halfseg_ids_to_file_id =
    new std::uint64_t*[n_halfsegments];

  {
    for (std::uint64_t i = 0; i < n_halfsegments; ++i)
      halfseg_ids_to_file_id[i] =
        new std::uint64_t[n_halfsegments];

    std::uint64_t file_counter = 0;
    for (std::uint64_t i = 0; i < n_halfsegments; ++i) {
      for (std::uint64_t j = i; j < n_halfsegments; ++j) {
        halfseg_ids_to_file_id[i][j] = file_counter;
        halfseg_ids_to_file_id[j][i] = file_counter;
        ++file_counter;
      }
    }
  }

  // Initialize multifile writer of (i, Phi[i]) pairs.
  static const std::uint64_t n_free_buffers = 4;
  std::uint64_t halfseg_buffers_ram = ram_use;
  std::uint64_t n_different_halfseg_pairs =
    (n_halfsegments * (n_halfsegments + 1)) / 2;
  std::uint64_t buffer_size = std::max(1UL,
      halfseg_buffers_ram /
      (n_different_halfseg_pairs + n_free_buffers));
  buffer_size = std::min((std::uint64_t)(2 << 20), buffer_size);

  typedef async_multi_stream_writer<text_offset_type>
    irr_pos_multiwriter_type;
  irr_pos_multiwriter_type *irr_pos_multiwriter =
    new irr_pos_multiwriter_type(n_different_halfseg_pairs,
        buffer_size, n_free_buffers);

  for (std::uint64_t i = 0; i < n_halfsegments; ++i)
    for (std::uint64_t j = i; j < n_halfsegments; ++j)
      irr_pos_multiwriter->add_file(pairs_filenames[i][j]);

  // Initialize suffix array reader.
  typedef async_stream_reader<text_offset_type> sa_reader_type;
  sa_reader_type *sa_reader =
    new sa_reader_type(sa_filename);

  // Initialize BWT reader.
  typedef async_stream_reader<char_type> bwt_reader_type;
  bwt_reader_type *bwt_reader =
    new bwt_reader_type(bwt_filename);

  // Distribution follows.
  char_type prev_bwt = 0;
  std::uint64_t prev_sa = 0;
  std::uint64_t prev_halfseg_id = 0;
  for (std::uint64_t i = 0; i < text_length; ++i) {
    std::uint64_t cur_sa = sa_reader->read();
    std::uint64_t cur_halfseg_id = cur_sa / max_halfsegment_size;
    char_type cur_bwt = bwt_reader->read();

    if (i == 0 ||
        cur_sa == 0 ||
        prev_sa == 0 ||
        cur_bwt != prev_bwt) {

      // PLCP[cur_sa] is irreducible.
      // Write (i, Phi[i]) to appropriate file.
      ++n_irreducible;
      if (i > 0) {
        std::uint64_t file_id =
          halfseg_ids_to_file_id[cur_halfseg_id][prev_halfseg_id];
        irr_pos_multiwriter->write_to_ith_file(
            file_id, (text_offset_type)cur_sa);
        irr_pos_multiwriter->write_to_ith_file(
            file_id, (text_offset_type)prev_sa);
      }
    }

    prev_halfseg_id = cur_halfseg_id;
    prev_sa = cur_sa;
    prev_bwt = cur_bwt;
  }

  // Stop I/O threads.
  sa_reader->stop_reading();
  bwt_reader->stop_reading();

  // Update I/O volume.
  std::uint64_t io_volume =
    sa_reader->bytes_read() +
    bwt_reader->bytes_read() +
    irr_pos_multiwriter->bytes_written();
  total_io_volume += io_volume;

  // Print summary.
  long double elapsed = utils::wclock() - start;
  fprintf(stderr, "time = %.2Lfs, I/O = %.2LfMiB/s, "
      "total I/O vol = %.2Lfbytes/input symbol\n",
      elapsed, ((1.L * io_volume) / (1L << 20)) / elapsed,
      (1.L * total_io_volume) / text_length);

  // Clean up.
  delete bwt_reader;
  delete sa_reader;
  delete irr_pos_multiwriter;

  for (std::uint64_t i = n_halfsegments; i > 0; --i)
    delete[] halfseg_ids_to_file_id[i - 1];
  delete[] halfseg_ids_to_file_id;

  // Update reference variables.
  n_irreducible_lcps = n_irreducible;
}

template<typename char_type,
  typename text_offset_type>
void compute_C(
    std::uint64_t text_length,
    std::uint64_t max_halfsegment_size,
    std::uint64_t ram_use,
    std::uint64_t phi_undefined_position,
    std::string **pairs_filenames,
    std::string sa_filename,
    std::string bwt_filename,
    std::string C_filename,
    std::uint64_t &total_io_volume) {

  std::uint64_t n_halfsegments =
    (text_length + max_halfsegment_size - 1) /
    max_halfsegment_size;
  std::uint64_t max_block_size = 8UL * ram_use;
  while (max_block_size & 63UL)
    ++max_block_size;

  std::uint64_t n_blocks =
    (text_length + max_block_size - 1) / max_block_size;
  std::uint64_t io_vol_scan_sa =
    (1 + sizeof(text_offset_type)) * text_length * n_blocks;

  std::uint64_t io_vol_scan_pairs = 0;
  for (std::uint64_t block_id = 0; block_id < n_blocks; ++block_id) {
    std::uint64_t block_beg = block_id * max_block_size;
    std::uint64_t block_end =
      std::min(block_beg + max_block_size, text_length);

    for (std::uint64_t left_halfseg_id = 0;
        left_halfseg_id < n_halfsegments; ++left_halfseg_id) {
      std::uint64_t left_halfseg_beg =
        left_halfseg_id * max_halfsegment_size;
      std::uint64_t left_halfseg_end =
        std::min(left_halfseg_beg + max_halfsegment_size, text_length);

      for (std::uint64_t right_halfseg_id = left_halfseg_id;
          right_halfseg_id < n_halfsegments; ++right_halfseg_id) {
        std::uint64_t right_halfseg_beg =
          right_halfseg_id * max_halfsegment_size;
        std::uint64_t right_halfseg_end =
          std::min(right_halfseg_beg + max_halfsegment_size, text_length);

        if ((left_halfseg_end > block_beg &&
              block_end > left_halfseg_beg) ||
            (right_halfseg_end > block_beg &&
             block_end > right_halfseg_beg)) {
          std::string filename =
            pairs_filenames[left_halfseg_id][right_halfseg_id];
          io_vol_scan_pairs += utils::file_size(filename);
        }
      }
    }
  }

  if (io_vol_scan_sa <= io_vol_scan_pairs) {

    // Print initial message.
    fprintf(stderr, "  Compute bitvector C (method I): ");

    // Initialize timer.
    long double start = utils::wclock();

    // Initialize I/O volume.
    std::uint64_t io_vol = 0;

    // Allocate the array holding the block of C.
    std::uint64_t max_block_size_in_words = max_block_size / 64;
    std::uint64_t *C =
      utils::allocate_array<std::uint64_t>(max_block_size_in_words);
    std::FILE *f = utils::file_open(C_filename, "w");

    // Initialize the buffer.
    static const std::uint64_t buffer_size = (1UL << 20);
    std::uint64_t *buf =
      utils::allocate_array<std::uint64_t>(buffer_size);

#ifdef _OPENMP
    std::uint64_t *tempbuf =
      utils::allocate_array<std::uint64_t>(buffer_size);
#endif

    for (std::uint64_t block_id = 0; block_id < n_blocks; ++block_id) {
      std::uint64_t block_beg = block_id * max_block_size;
      std::uint64_t block_end =
        std::min(block_beg + max_block_size, text_length);
      std::uint64_t block_size = block_end - block_beg;
      std::uint64_t block_size_in_words = (block_size + 63) / 64;

      // Zero-initialize the block of C.
      std::fill(C, C + block_size_in_words, 0UL);

      // Initialize suffix array reader.
      typedef async_stream_reader<text_offset_type> sa_reader_type;
      sa_reader_type *sa_reader = new sa_reader_type(sa_filename);

      // Initialize BWT reader.
      typedef async_stream_reader<char_type> bwt_reader_type;
      bwt_reader_type *bwt_reader = new bwt_reader_type(bwt_filename);

      // Scan SA and BWT left to right.
      std::uint64_t filled = 0;
      char_type prev_bwt = 0;
      std::uint64_t prev_sa = 0;
      for (std::uint64_t i = 0; i < text_length; ++i) {
        std::uint64_t cur_sa = sa_reader->read();
        char_type cur_bwt = bwt_reader->read();

        if (block_beg <= cur_sa &&
            cur_sa < block_end &&
            (i == 0 ||
             cur_sa == 0 ||
             prev_sa == 0 ||
             cur_bwt != prev_bwt)) {

          // PLCP[cur_sa] is irreducible.
          std::uint64_t offset = cur_sa - block_beg;
          buf[filled++] = offset;
          if (filled == buffer_size) {

#ifdef _OPENMP
            set_bits(C, block_size, buf, filled, tempbuf);
#else
            set_bits(C, buf, filled);
#endif

            filled = 0;
          }
        }

        prev_sa = cur_sa;
        prev_bwt = cur_bwt;
      }

      // Flush the remaining items in the buffer.
      if (filled > 0) {

#ifdef _OPENMP
        set_bits(C, block_size, buf, filled, tempbuf);
#else
        set_bits(C, buf, filled);
#endif

        filled = 0;
      }

      // Write current block of C to file.
      utils::write_to_file(C, block_size_in_words, f);

      // Stop I/O threads.
      sa_reader->stop_reading();
      bwt_reader->stop_reading();

      // Update I/O volume.
      io_vol +=
        sa_reader->bytes_read() +
        bwt_reader->bytes_read() +
        block_size_in_words * sizeof(std::uint64_t);

      // Clean up.
      delete bwt_reader;
      delete sa_reader;
    }

    // Clean up.
#ifdef _OPENMP
    utils::deallocate(tempbuf);
#endif

    utils::deallocate(buf);
    utils::deallocate(C);
    std::fclose(f);

    // Update I/O volume.
    total_io_volume += io_vol;

    // Print summary.
    long double elapsed = utils::wclock() - start;
    fprintf(stderr, "time = %.2Lfs, I/O = %.2LfMiB/s, "
        "total I/O vol = %.2Lfbytes/input symbol\n", elapsed,
        ((1.L * io_vol) / (1L << 20)) / elapsed,
        (1.L * total_io_volume) / text_length);

  } else {

    // Print initial message, start the timer
    // and initialize I/O volume.
    fprintf(stderr, "  Compute bitvector C (method II): ");
    long double start = utils::wclock();
    std::uint64_t io_vol = 0;

    // Allocate the array holding the block of C.
    std::uint64_t max_block_size_in_words = max_block_size / 64;
    std::uint64_t *C =
      utils::allocate_array<std::uint64_t>(max_block_size_in_words);
    std::FILE *f = utils::file_open(C_filename, "w");

    // Initialize the buffer.
    static const std::uint64_t buffer_size = (1UL << 20);
    std::uint64_t *buf =
      utils::allocate_array<std::uint64_t>(buffer_size);

#ifdef _OPENMP
    std::uint64_t *tempbuf =
      utils::allocate_array<std::uint64_t>(buffer_size);
#endif

    // Process blocks of C left to right.
    for (std::uint64_t block_id = 0; block_id < n_blocks; ++block_id) {
      std::uint64_t block_beg = block_id * max_block_size;
      std::uint64_t block_end =
        std::min(block_beg + max_block_size, text_length);
      std::uint64_t block_size = block_end - block_beg;
      std::uint64_t block_size_in_words = (block_size + 63) / 64;

      // Zero-initialize the block of C.
      std::fill(C, C + block_size_in_words, 0UL);

      // Iterate through all pairs of halfsegments.
      std::uint64_t filled = 0;
      for (std::uint64_t left_halfseg_id = 0;
          left_halfseg_id < n_halfsegments; ++left_halfseg_id) {
        std::uint64_t left_halfseg_beg =
          left_halfseg_id * max_halfsegment_size;
        std::uint64_t left_halfseg_end =
          std::min(left_halfseg_beg + max_halfsegment_size, text_length);

        for (std::uint64_t right_halfseg_id = left_halfseg_id;
            right_halfseg_id < n_halfsegments; ++right_halfseg_id) {
          std::uint64_t right_halfseg_beg =
            right_halfseg_id * max_halfsegment_size;
          std::uint64_t right_halfseg_end =
            std::min(right_halfseg_beg + max_halfsegment_size, text_length);

          if ((left_halfseg_end > block_beg &&
                block_end > left_halfseg_beg) ||
              (right_halfseg_end > block_beg &&
               block_end > right_halfseg_beg)) {

            // Initialize reading of pairs.
            typedef async_stream_reader<text_offset_type> pair_reader_type;
            pair_reader_type *pair_reader =
              new pair_reader_type(
                  pairs_filenames[left_halfseg_id][right_halfseg_id]);

            while (pair_reader->empty() == false) {
              std::uint64_t i = pair_reader->read();
              pair_reader->read();  // Skip Phi[i].

              if (block_beg <= i && i < block_end) {
                std::uint64_t offset = i - block_beg;
                buf[filled++] = offset;
                if (filled == buffer_size) {

#ifdef _OPENMP
                  set_bits(C, block_size, buf, filled, tempbuf);
#else
                  set_bits(C, buf, filled);
#endif

                  filled = 0;
                }
              }
            }

            // Stop I/O threads.
            pair_reader->stop_reading();

            // Update I/O volume.
            io_vol +=
              pair_reader->bytes_read();

            // Clean up.
            delete pair_reader;
          }
        }
      }

      // Flush the remaining items in the buffer.
      if (filled > 0) {

#ifdef _OPENMP
        set_bits(C, block_size, buf, filled, tempbuf);
#else
        set_bits(C, buf, filled);
#endif

        filled = 0;
      }

      // Special case.
      if (block_beg <= phi_undefined_position &&
          phi_undefined_position < block_end) {
        std::uint64_t offset = phi_undefined_position - block_beg;
        C[offset >> 6] |= (1UL << (offset & 63));
      }

      // Write current block of C to file.
      utils::write_to_file(C, block_size_in_words, f);

      // Update I/O volume.
      io_vol += block_size_in_words * sizeof(std::uint64_t);
    }

    // Clean up.
#ifdef _OPENMP
    utils::deallocate(tempbuf);
#endif

    utils::deallocate(buf);
    utils::deallocate(C);
    std::fclose(f);

    // Update I/O volume.
    total_io_volume += io_vol;

    // Print summary.
    long double elapsed = utils::wclock() - start;
    fprintf(stderr, "time = %.2Lfs, I/O = %.2LfMiB/s, "
        "total I/O vol = %.2Lfbytes/input symbol\n",
        elapsed, ((1.L * io_vol) / (1L << 20)) / elapsed,
        (1.L * total_io_volume) / text_length);
  }
}

template<typename char_type,
  typename text_offset_type>
void distribute_pairs_and_compute_C(
    std::uint64_t text_length,
    std::uint64_t max_halfsegment_size,
    std::uint64_t ram_use,
    std::string sa_filename,
    std::string bwt_filename,
    std::string C_filename,
    std::string **pairs_filenames,
    std::uint64_t &n_irreducible_lcps,
    std::uint64_t &total_io_volume) {

  // Print initial message.
  fprintf(stderr, "  Distribute (i, Phi[i]) pairs and compute C: ");

  // Initialize the timer.
  long double start = utils::wclock();

  // Initialize basic parameters.
  std::uint64_t n_halfsegments =
    (text_length + max_halfsegment_size - 1) /
    max_halfsegment_size;
  std::uint64_t n_different_halfseg_pairs =
    (n_halfsegments * (n_halfsegments + 1)) / 2;
  std::uint64_t io_volume = 0;
  std::uint64_t n_irreducible = 0;

  // Allocate bitvector C.
  std::uint64_t C_size_in_words = (text_length + 63) / 64;
  std::uint64_t C_size_in_bytes = (text_length + 7) / 8;
  std::uint64_t *C = utils::allocate_array<std::uint64_t>(C_size_in_words);
  std::fill(C, C + C_size_in_words, (std::uint64_t)0);

  // Create a map from used halfsegment pairs to a contiguous
  // range of integers. This is needed to use multifile writer.
  std::uint64_t **halfseg_ids_to_file_id =
    new std::uint64_t*[n_halfsegments];

  {
    for (std::uint64_t i = 0; i < n_halfsegments; ++i)
      halfseg_ids_to_file_id[i] = new std::uint64_t[n_halfsegments];

    std::uint64_t file_counter = 0;
    for (std::uint64_t i = 0; i < n_halfsegments; ++i) {
      for (std::uint64_t j = i; j < n_halfsegments; ++j) {
        halfseg_ids_to_file_id[i][j] = file_counter;
        halfseg_ids_to_file_id[j][i] = file_counter;
        ++file_counter;
      }
    }
  }

  // Initialize multifile writer of (i, Phi[i]) pairs.
  // XXX: for large alphabet 1 << 20 is an issue.
  static const std::uint64_t n_free_buffers = 4;
  std::uint64_t halfseg_buffers_ram = ram_use - C_size_in_bytes;
#if 0
  std::uint64_t buffer_size = std::max((1UL << 20),
      halfseg_buffers_ram /
      (n_different_halfseg_pairs + n_free_buffers));
#else
  std::uint64_t buffer_size = std::max((64UL << 10),
      halfseg_buffers_ram /
      (n_different_halfseg_pairs + n_free_buffers));
#endif

  typedef async_multi_stream_writer<text_offset_type>
    irr_pos_multiwriter_type;
  irr_pos_multiwriter_type *irr_pos_multiwriter =
    new irr_pos_multiwriter_type(n_different_halfseg_pairs,
        buffer_size, n_free_buffers);

  for (std::uint64_t i = 0; i < n_halfsegments; ++i)
    for (std::uint64_t j = i; j < n_halfsegments; ++j)
      irr_pos_multiwriter->add_file(pairs_filenames[i][j]);

  // Initialize suffix array reader.
  typedef async_stream_reader<text_offset_type> sa_reader_type;
  sa_reader_type *sa_reader = new sa_reader_type(sa_filename);

  // Initialize BWT reader.
  typedef async_stream_reader<char_type> bwt_reader_type;
  bwt_reader_type *bwt_reader = new bwt_reader_type(bwt_filename);

  // Initialize the buffer.
  static const std::uint64_t local_buffer_size = (1UL << 20);
  std::uint64_t *buf =
    utils::allocate_array<std::uint64_t>(local_buffer_size);
#ifdef _OPENMP
  std::uint64_t *tempbuf =
    utils::allocate_array<std::uint64_t>(local_buffer_size);
#endif

  // Distribution follows.
  char_type prev_bwt = (char_type)0;
  std::uint64_t filled = 0;
  std::uint64_t prev_sa = 0;
  std::uint64_t prev_halfseg_id = 0;
  for (std::uint64_t i = 0; i < text_length; ++i) {
    std::uint64_t cur_sa = sa_reader->read();
    std::uint64_t cur_halfseg_id = cur_sa / max_halfsegment_size;
    char_type cur_bwt = bwt_reader->read();

    if (i == 0 ||
        cur_sa == 0 ||
        prev_sa == 0 ||
        cur_bwt != prev_bwt) {

      // PLCP[cur_sa] is irreducible.
      // Write (i, Phi[i]) to appropriate file.
      ++n_irreducible;
      buf[filled++] = cur_sa;
      if (filled == local_buffer_size) {

#ifdef _OPENMP
        set_bits(C, text_length, buf, filled, tempbuf);
#else
        set_bits(C, buf, filled);
#endif

        filled = 0;
      }

      if (i > 0) {
        std::uint64_t file_id =
          halfseg_ids_to_file_id[cur_halfseg_id][prev_halfseg_id];
        irr_pos_multiwriter->write_to_ith_file(
            file_id, (text_offset_type)cur_sa);
        irr_pos_multiwriter->write_to_ith_file(
            file_id, (text_offset_type)prev_sa);
      }
    }

    prev_halfseg_id = cur_halfseg_id;
    prev_sa = cur_sa;
    prev_bwt = cur_bwt;
  }

  // Flush the remaining items in the buffer.
  if (filled > 0) {

#ifdef _OPENMP
    set_bits(C, text_length, buf, filled, tempbuf);
#else
    set_bits(C, buf, filled);
#endif

    filled = 0;
  }

  // Write C to disk.
  utils::write_to_file(C,
      C_size_in_words, C_filename);

  // Stop I/O threads.
  sa_reader->stop_reading();
  bwt_reader->stop_reading();

  // Update I/O volume.
  io_volume +=
    sa_reader->bytes_read() +
    bwt_reader->bytes_read() +
    irr_pos_multiwriter->bytes_written() +
    C_size_in_words * sizeof(std::uint64_t);
  total_io_volume += io_volume;

  // Clean up.
#ifdef _OPENMP
  utils::deallocate(tempbuf);
#endif

  utils::deallocate(buf);
  delete bwt_reader;
  delete sa_reader;
  delete irr_pos_multiwriter;
  utils::deallocate(C);

  for (std::uint64_t i = n_halfsegments; i > 0; --i)
    delete[] halfseg_ids_to_file_id[i - 1];
  delete[] halfseg_ids_to_file_id;

  // Print summary.
  long double elapsed = utils::wclock() - start;
  fprintf(stderr, "time = %.2Lfs, I/O = %.2LfMiB/s, "
      "total I/O vol = %.2Lfbytes/input symbol\n",
      elapsed, ((1.L * io_volume) / (1L << 20)) / elapsed,
      (1.L * total_io_volume) / text_length);

  // Update reference variables.
  n_irreducible_lcps = n_irreducible;
}

}  // namespace normal_mode

namespace inplace_mode {

template<typename char_type,
  typename text_offset_type>
void distribute_pairs_text_partitioning(
    std::uint64_t text_length,
    std::uint64_t max_halfsegment_size,
    std::uint64_t ram_use,
    std::uint64_t part_id,
    std::uint64_t ***items_per_halfseg_pair,
    std::string sa_filename,
    std::string bwt_filename,
    std::string **pos_filenames,
    std::string **phi_filenames,
    std::uint64_t &total_io_volume) {

  // Initial setup.
  std::uint64_t n_halfsegments =
    (text_length + max_halfsegment_size - 1) /
    max_halfsegment_size;

  // Print initial message.
  fprintf(stderr, "    Distribute (i, Phi[i]) pairs:\n");

  // Initialize I/O volume.
  std::uint64_t io_volume = 0;

  // Initialize the timer.
  long double start = utils::wclock();

  // Compute how many items belonging for each pairs of halfsegments
  // were processed. Negative values allows skipping items.
  std::int64_t **pairs_processed = new std::int64_t*[n_halfsegments];
  for (std::uint64_t i = 0; i < n_halfsegments; ++i) {
    pairs_processed[i] = new std::int64_t[n_halfsegments];
    std::fill(pairs_processed[i], pairs_processed[i] + n_halfsegments,
        (std::uint64_t)0);

    for (std::uint64_t j = i; j < n_halfsegments; ++j) {
      for (std::uint64_t prev_id = 0; prev_id < part_id; ++prev_id) {
        std::uint64_t count = items_per_halfseg_pair[prev_id][i][j];
        pairs_processed[i][j] -= (std::int64_t)count;
      }
    }
  }

  // Compute the number of used
  // different halfsegment pairs.
  std::uint64_t different_used_pairs_of_halfsegments = 0;
  for (std::uint64_t i = 0; i < n_halfsegments; ++i)
    for (std::uint64_t j = i; j < n_halfsegments; ++j)
      if (items_per_halfseg_pair[part_id][i][j] > 0)
        ++different_used_pairs_of_halfsegments;

  // Compute buffer size.
  static const std::uint64_t n_free_buffers = 4;
  std::uint64_t halfseg_buffers_ram = ram_use;
  std::uint64_t buffer_size = std::max(1UL, halfseg_buffers_ram /
    (2UL * different_used_pairs_of_halfsegments + n_free_buffers));
  buffer_size = std::min((std::uint64_t)(2 << 20), buffer_size);

  fprintf(stderr, "      Different halfsegment pairs = %lu\n",
      different_used_pairs_of_halfsegments);
  fprintf(stderr, "      Buffer size = %lu\n", buffer_size);

  // Create a multifile writer for files
  // storing irreducible positions.
  typedef async_multi_stream_writer<text_offset_type>
    irr_pos_multiwriter_type;
  irr_pos_multiwriter_type *irr_pos_multiwriter =
    new irr_pos_multiwriter_type(
        2UL * different_used_pairs_of_halfsegments,
        buffer_size, n_free_buffers);

  // Create a map from used halfsegment pairs
  // to a contiguous range of integers. This
  // is needed to use multifile writer.
  std::uint64_t **halfseg_ids_to_pos_file_id =
    new std::uint64_t*[n_halfsegments];
  std::uint64_t **halfseg_ids_to_phi_file_id =
    new std::uint64_t*[n_halfsegments];

  {
    for (std::uint64_t i = 0; i < n_halfsegments; ++i) {
      halfseg_ids_to_pos_file_id[i] = new std::uint64_t[n_halfsegments];
      halfseg_ids_to_phi_file_id[i] = new std::uint64_t[n_halfsegments];
    }

    std::uint64_t file_counter = 0;
    for (std::uint64_t i = 0; i < n_halfsegments; ++i) {
      for (std::uint64_t j = i; j < n_halfsegments; ++j) {
        if (items_per_halfseg_pair[part_id][i][j] > 0) {
          halfseg_ids_to_pos_file_id[i][j] = file_counter;
          halfseg_ids_to_pos_file_id[j][i] = file_counter;
          irr_pos_multiwriter->add_file(pos_filenames[i][j]);
          ++file_counter;
          halfseg_ids_to_phi_file_id[i][j] = file_counter;
          halfseg_ids_to_phi_file_id[j][i] = file_counter;
          irr_pos_multiwriter->add_file(phi_filenames[i][j]);
          ++file_counter;
        }
      }
    }
  }

  // Create suffix array reader.
  typedef async_stream_reader<text_offset_type> sa_reader_type;
  sa_reader_type *sa_reader = new sa_reader_type(sa_filename);

  // Creat BWT reader.
  typedef async_stream_reader<char_type> bwt_reader_type;
  bwt_reader_type *bwt_reader = new bwt_reader_type(bwt_filename);

  // Distribution follows.
  char_type prev_bwt = 0;
  std::uint64_t prev_sa = 0;
  std::uint64_t prev_halfseg_id = 0;
  std::uint64_t **toprocess = items_per_halfseg_pair[part_id];

  for (std::uint64_t i = 0; i < text_length; ++i) {
    std::uint64_t cur_sa = sa_reader->read();
    std::uint64_t cur_halfseg_id = cur_sa / max_halfsegment_size;
    char_type cur_bwt = bwt_reader->read();

    if ((i > 0) &&
        (cur_sa == 0 ||
         prev_sa == 0 ||
         cur_bwt != prev_bwt)) {

      std::uint64_t h1 = cur_halfseg_id;
      std::uint64_t h2 = prev_halfseg_id;

      if (h1 > h2)
        std::swap(h1, h2);

      // Check if current item should be
      // processed in this text part.
      ++pairs_processed[h1][h2];
      if (pairs_processed[h1][h2] > (std::int64_t)0 &&
          pairs_processed[h1][h2] <= (std::int64_t)toprocess[h1][h2]) {

        // PLCP[cur_sa] is irreducible. Write
        // i and Phi[i] to appropriate files.
        std::uint64_t pos_file_id =
          halfseg_ids_to_pos_file_id[cur_halfseg_id][prev_halfseg_id];
        std::uint64_t phi_file_id =
          halfseg_ids_to_phi_file_id[cur_halfseg_id][prev_halfseg_id];
        irr_pos_multiwriter->write_to_ith_file(
            pos_file_id, (text_offset_type)cur_sa);
        irr_pos_multiwriter->write_to_ith_file(
            phi_file_id, (text_offset_type)prev_sa);
      }
    }

    // Update prev values.
    prev_halfseg_id = cur_halfseg_id;
    prev_sa = cur_sa;
    prev_bwt = cur_bwt;
  }

  // Stop I/O threads.
  sa_reader->stop_reading();
  bwt_reader->stop_reading();

  // Update I/O volume.
  io_volume +=
    sa_reader->bytes_read() +
    bwt_reader->bytes_read() +
    irr_pos_multiwriter->bytes_written();
  total_io_volume += io_volume;

  // Clean up.
  delete bwt_reader;
  delete sa_reader;

  for (std::uint64_t i = n_halfsegments; i > 0; --i) {
    delete[] halfseg_ids_to_phi_file_id[i - 1];
    delete[] halfseg_ids_to_pos_file_id[i - 1];
  }

  delete[] halfseg_ids_to_phi_file_id;
  delete[] halfseg_ids_to_pos_file_id;

  delete irr_pos_multiwriter;

  for (std::uint64_t i = n_halfsegments; i > 0; --i)
    delete[] pairs_processed[i - 1];
  delete[] pairs_processed;

  // Print summary.
  long double elapsed = utils::wclock() - start;
  fprintf(stderr, "      Summary: time = %.2Lfs, I/O = %.2LfMiB/s, "
      "total I/O vol = %.2Lfbytes/input symbol\n",
      elapsed, ((1.L * io_volume) / (1L << 20)) / elapsed,
      (1.L * total_io_volume) / text_length);
}

template<typename char_type,
  typename text_offset_type>
std::uint64_t distribute_pairs_lex_partitioning(
    std::uint64_t text_length,
    std::uint64_t max_halfsegment_size,
    std::uint64_t ram_use,
    std::uint64_t cur_sa_range_beg,
    std::uint64_t max_pairs_written,
    std::string sa_filename,
    std::string bwt_filename,
    std::string **pos_filenames,
    std::string **phi_filenames,
    std::uint64_t &n_irreducible_lcps,
    std::uint64_t &total_io_volume) {

  // Initial setup.
  std::uint64_t n_halfsegments =
    (text_length + max_halfsegment_size - 1) /
    max_halfsegment_size;
  std::uint64_t n_irreducible = 0;

  // Print initial message.
  fprintf(stderr, "    Distribute (i, Phi[i]) pairs: ");

  // Initialize I/O volume.
  std::uint64_t io_volume = 0;

  // Initialize the timer.
  long double start = utils::wclock();

  // Create a map from used halfsegment pairs to a contiguous
  // range of integers. This is needed to use multifile writer.
  std::uint64_t **halfseg_ids_to_pos_file_id =
    new std::uint64_t*[n_halfsegments];
  std::uint64_t **halfseg_ids_to_phi_file_id =
    new std::uint64_t*[n_halfsegments];

  {
    for (std::uint64_t i = 0; i < n_halfsegments; ++i) {
      halfseg_ids_to_pos_file_id[i] = new std::uint64_t[n_halfsegments];
      halfseg_ids_to_phi_file_id[i] = new std::uint64_t[n_halfsegments];
    }

    std::uint64_t file_counter = 0;
    for (std::uint64_t i = 0; i < n_halfsegments; ++i) {
      for (std::uint64_t j = i; j < n_halfsegments; ++j) {
        halfseg_ids_to_pos_file_id[i][j] = file_counter;
        halfseg_ids_to_pos_file_id[j][i] = file_counter;
        ++file_counter;
        halfseg_ids_to_phi_file_id[i][j] = file_counter;
        halfseg_ids_to_phi_file_id[j][i] = file_counter;
        ++file_counter;
      }
    }
  }

  // Initialize multifile writer for files
  // storing irreducible positions.
  static const std::uint64_t n_free_buffers = 4;
  std::uint64_t halfseg_buffers_ram = ram_use;
  std::uint64_t n_different_halfseg_pairs =
    (n_halfsegments * (n_halfsegments + 1)) / 2;
  std::uint64_t buffer_size = std::max(1UL,
      halfseg_buffers_ram /
      (2UL * n_different_halfseg_pairs + n_free_buffers));
  buffer_size = std::min((std::uint64_t)(2 << 20), buffer_size);

  typedef async_multi_stream_writer<text_offset_type>
    irr_pos_multiwriter_type;
  irr_pos_multiwriter_type *irr_pos_multiwriter =
    new irr_pos_multiwriter_type(2UL * n_different_halfseg_pairs,
        buffer_size, n_free_buffers);

  for (std::uint64_t i = 0; i < n_halfsegments; ++i) {
    for (std::uint64_t j = i; j < n_halfsegments; ++j) {
      irr_pos_multiwriter->add_file(pos_filenames[i][j]);
      irr_pos_multiwriter->add_file(phi_filenames[i][j]);
    }
  }

  // Initialize suffix array reader.
  typedef async_stream_reader<text_offset_type> sa_reader_type;
  sa_reader_type *sa_reader =
    new sa_reader_type(sa_filename, cur_sa_range_beg);

  // Initialize BWT reader.
  typedef async_stream_reader<char_type> bwt_reader_type;
  bwt_reader_type *bwt_reader =
    new bwt_reader_type(bwt_filename, cur_sa_range_beg);

  // Distribution follows.
  char_type prev_bwt = 0;
  std::uint64_t prev_sa = 0;
  std::uint64_t prev_halfseg_id = 0;

  if (cur_sa_range_beg > 0) {
    text_offset_type val = 0;
    std::uint64_t bwt_file_offset =
      (cur_sa_range_beg - 1) * sizeof(char_type);
    std::uint64_t sa_file_offset =
      (cur_sa_range_beg - 1) * sizeof(text_offset_type);
    utils::read_at_offset(&prev_bwt, bwt_file_offset, 1, bwt_filename);
    utils::read_at_offset(&val, sa_file_offset, 1, sa_filename);
    io_volume += sizeof(text_offset_type) + sizeof(char_type);

    prev_sa = val;
    prev_halfseg_id = prev_sa / max_halfsegment_size;
  }

  std::uint64_t new_sa_range_beg = cur_sa_range_beg;
  std::uint64_t pairs_written = 0;
  while (pairs_written < max_pairs_written &&
      new_sa_range_beg < text_length) {
    std::uint64_t cur_sa = sa_reader->read();
    std::uint64_t cur_halfseg_id = cur_sa / max_halfsegment_size;
    char_type cur_bwt = bwt_reader->read();

    if (new_sa_range_beg == 0 ||
        cur_sa == 0 ||
        prev_sa == 0 ||
        cur_bwt != prev_bwt) {

      // PLCP[cur_sa] is irreducible. Write
      // i and Phi[i] to appropriate files.
      ++n_irreducible;
      if (new_sa_range_beg > 0) {
        ++pairs_written;
        std::uint64_t pos_file_id =
          halfseg_ids_to_pos_file_id[cur_halfseg_id][prev_halfseg_id];
        std::uint64_t phi_file_id =
          halfseg_ids_to_phi_file_id[cur_halfseg_id][prev_halfseg_id];
        irr_pos_multiwriter->write_to_ith_file(
            pos_file_id, (text_offset_type)cur_sa);
        irr_pos_multiwriter->write_to_ith_file(
            phi_file_id, (text_offset_type)prev_sa);
      }
    }

    prev_halfseg_id = cur_halfseg_id;
    prev_sa = cur_sa;
    prev_bwt = cur_bwt;
    ++new_sa_range_beg;
  }

  // Stop I/O threads.
  sa_reader->stop_reading();
  bwt_reader->stop_reading();

  // Update I/O volume.
  io_volume +=
    sa_reader->bytes_read() +
    bwt_reader->bytes_read() +
    irr_pos_multiwriter->bytes_written();
  total_io_volume += io_volume;

  // Clean up.
  delete bwt_reader;
  delete sa_reader;
  delete irr_pos_multiwriter;

  for (std::uint64_t i = n_halfsegments; i > 0; --i) {
    delete[] halfseg_ids_to_phi_file_id[i - 1];
    delete[] halfseg_ids_to_pos_file_id[i - 1];
  }

  delete[] halfseg_ids_to_phi_file_id;
  delete[] halfseg_ids_to_pos_file_id;

  // Print summary.
  long double elapsed = utils::wclock() - start;
  fprintf(stderr, "time = %.2Lfs, I/O = %.2LfMiB/s, "
      "total I/O vol = %.2Lfbytes/input symbol\n",
      elapsed, ((1.L * io_volume) / (1L << 20)) / elapsed,
      (1.L * total_io_volume) / text_length);

  // Update reference variables.
  n_irreducible_lcps = n_irreducible;

  // Return the result.
  return new_sa_range_beg;
}

template<typename char_type,
  typename text_offset_type>
void compute_C(
    std::uint64_t text_length,
    std::uint64_t max_halfsegment_size,
    std::uint64_t ram_use,
    std::uint64_t phi_undefined_position,
    std::string **pos_filenames,
    std::string C_filename,
    std::uint64_t &total_io_volume) {

  // Initialize basic parameters.
  std::uint64_t n_halfsegments =
    (text_length + max_halfsegment_size - 1) / max_halfsegment_size;
  std::uint64_t max_block_size = 8UL * ram_use;
  while (max_block_size & 63UL)
    ++max_block_size;
  std::uint64_t n_blocks =
    (text_length + max_block_size - 1) / max_block_size;

  // Print initial message, start the
  // timer and initialize I/O volume.
  fprintf(stderr, "    Compute bitvector C: ");
  long double start = utils::wclock();
  std::uint64_t io_vol = 0;

  // Allocate the array holding the block of C.
  std::uint64_t max_block_size_in_words = max_block_size / 64;
  std::uint64_t *C =
    utils::allocate_array<std::uint64_t>(max_block_size_in_words);

  // Check if file with B already exists. This is
  // to determine if we are processing the first part.
  bool C_exists = utils::file_exists(C_filename);

  // Open the file to write C.
  std::FILE *f = NULL;
  if (C_exists) f = utils::file_open(C_filename, "r+");
  else f = utils::file_open(C_filename, "w");

  // Allocate buffers.
  static const std::uint64_t buffer_size = (1UL << 20);
  std::uint64_t *buf =
    utils::allocate_array<std::uint64_t>(buffer_size);

#ifdef _OPENMP
  std::uint64_t *tempbuf =
    utils::allocate_array<std::uint64_t>(buffer_size);
#endif

  // Process blocks of C left to right.
  for (std::uint64_t block_id = 0; block_id < n_blocks; ++block_id) {
    std::uint64_t block_beg = block_id * max_block_size;
    std::uint64_t block_end =
      std::min(block_beg + max_block_size, text_length);
    std::uint64_t block_size = block_end - block_beg;
    std::uint64_t block_size_in_words = (block_size + 63) / 64;

    if (C_exists) {

      // Read block of C from disk for update.
      std::uint64_t offset = block_beg / 8;
      utils::read_at_offset(C, offset,
          block_size_in_words, f);
    } else {

      // Zero-initialize the block of C.
      std::fill(C, C + block_size_in_words,
          (std::uint64_t)0);
    }

    // Iterate through left halfsegmenets.
    std::uint64_t filled = 0;
    for (std::uint64_t left_halfseg_id = 0;
        left_halfseg_id < n_halfsegments; ++left_halfseg_id) {
      std::uint64_t left_halfseg_beg =
        left_halfseg_id * max_halfsegment_size;
      std::uint64_t left_halfseg_end =
        std::min(left_halfseg_beg + max_halfsegment_size, text_length);

      // Iterate through right halfsegments.
      for (std::uint64_t right_halfseg_id = left_halfseg_id;
          right_halfseg_id < n_halfsegments; ++right_halfseg_id) {
        std::uint64_t right_halfseg_beg =
          right_halfseg_id * max_halfsegment_size;
        std::uint64_t right_halfseg_end =
          std::min(right_halfseg_beg + max_halfsegment_size, text_length);
        std::string pos_filename =
          pos_filenames[left_halfseg_id][right_halfseg_id];

        // Continue only if any of the current halfsegments
        // can contain a position inside the current block of C.
        if (((left_halfseg_end > block_beg &&
                block_end > left_halfseg_beg) ||
            (right_halfseg_end > block_beg &&
             block_end > right_halfseg_beg)) &&
            utils::file_exists(pos_filename)) {

          // Initialize the reader of positions.
          typedef async_stream_reader<text_offset_type> pos_reader_type;
          pos_reader_type *pos_reader = new pos_reader_type(pos_filename);

          // Process all positions.
          while (!pos_reader->empty()) {
            std::uint64_t i = pos_reader->read();
            if (block_beg <= i && i < block_end) {
              std::uint64_t offset = i - block_beg;
              buf[filled++] = offset;
              if (filled == buffer_size) {

#ifdef _OPENMP
                set_bits(C, block_size, buf, filled, tempbuf);
#else
                set_bits(C, buf, filled);
#endif

                filled = 0;
              }
            }
          }

          // Stop I/O threads.
          pos_reader->stop_reading();

          // Update I/O volume.
          io_vol +=
            pos_reader->bytes_read();

          // Clean up.
          delete pos_reader;
        }
      }
    }

    // Flush the remaining items in the buffer.
    if (filled > 0) {

#ifdef _OPENMP
      set_bits(C, block_size, buf, filled, tempbuf);
#else
      set_bits(C, buf, filled);
#endif

      filled = 0;
    }

    // Special case.
    if (block_beg <= phi_undefined_position &&
        phi_undefined_position < block_end) {
      std::uint64_t offset = phi_undefined_position - block_beg;
      C[offset >> 6] |= (1UL << (offset & 63));
    }

    // Write current block of C to file.
    if (C_exists)
      utils::overwrite_at_offset(C, block_beg >> 6, block_size_in_words, f);
    else utils::write_to_file(C, block_size_in_words, f);

    // Update I/O volume.
    io_vol +=
      block_size_in_words * sizeof(std::uint64_t);

    // Account for the reading of C.
    if (C_exists)
      io_vol += block_size_in_words * sizeof(std::uint64_t);
  }

    // Clean up.
#ifdef _OPENMP
  utils::deallocate(tempbuf);
#endif

  utils::deallocate(buf);
  utils::deallocate(C);
  std::fclose(f);

  // Update I/O volume.
  total_io_volume += io_vol;

  // Print summary.
  long double elapsed = utils::wclock() - start;
  fprintf(stderr, "time = %.2Lfs, I/O = %.2LfMiB/s, "
      "total I/O vol = %.2Lfbytes/input symbol\n", elapsed,
      ((1.L * io_vol) / (1L << 20)) / elapsed,
      (1.L * total_io_volume) / text_length);
}

template<typename char_type,
  typename text_offset_type>
void distribute_pairs_and_compute_C_text_partitioning(
    std::uint64_t text_length,
    std::uint64_t max_halfsegment_size,
    std::uint64_t ram_use,
    std::uint64_t part_id,
    std::uint64_t ***items_per_halfseg_pair,
    std::uint64_t phi_undefined_position,
    std::string sa_filename,
    std::string bwt_filename,
    std::string C_filename,
    std::string **pos_filenames,
    std::string **phi_filenames,
    std::uint64_t &total_io_volume) {

  // Print initial message.
  fprintf(stderr, "    Distribute (i, Phi[i]) pairs and compute C:\n");

  // Initialize the timer.
  long double start = utils::wclock();

  // Initialize I/O volume.
  std::uint64_t io_volume = 0;

  // Compute basic parameters.
  std::uint64_t n_halfsegments =
    (text_length + max_halfsegment_size - 1) / max_halfsegment_size;

  // Allocate bitvector C.
  std::uint64_t C_size_in_words = (text_length + 63) / 64;
  std::uint64_t C_size_in_bytes = (text_length + 7) / 8;
  std::uint64_t *C = NULL;
  if (part_id == 0) {
    C = utils::allocate_array<std::uint64_t>(C_size_in_words);
    std::fill(C, C + C_size_in_words, (std::uint64_t)0);
  }

  // Compute how many items belonging for each pairs of halfsegments
  // were processed. Negative values allows skipping items.
  std::int64_t **pairs_processed = new std::int64_t*[n_halfsegments];
  for (std::uint64_t i = 0; i < n_halfsegments; ++i) {
    pairs_processed[i] = new std::int64_t[n_halfsegments];
    std::fill(pairs_processed[i], pairs_processed[i] + n_halfsegments,
        (std::uint64_t)0);

    for (std::uint64_t j = i; j < n_halfsegments; ++j) {
      for (std::uint64_t prev_id = 0; prev_id < part_id; ++prev_id) {
        std::uint64_t count = items_per_halfseg_pair[prev_id][i][j];
        pairs_processed[i][j] -= (std::int64_t)count;
      }
    }
  }

  // Compute the number of used
  // different halfsegment pairs.
  std::uint64_t different_used_pairs_of_halfsegments = 0;
  for (std::uint64_t i = 0; i < n_halfsegments; ++i)
    for (std::uint64_t j = i; j < n_halfsegments; ++j)
      if (items_per_halfseg_pair[part_id][i][j] > 0)
        ++different_used_pairs_of_halfsegments;

  // Compute buffer size.
  static const std::uint64_t n_free_buffers = 4;
  std::uint64_t halfseg_buffers_ram = ram_use;
  if (part_id == 0) {

    // XXX for safety, but in general this should be
    // replaced with a check that causes error and a
    // more subtle condition for entering this function.
    if (C_size_in_bytes > halfseg_buffers_ram)
      halfseg_buffers_ram = 0;
    else halfseg_buffers_ram -= C_size_in_bytes;
  }

  // XXX 64 << 10
  std::uint64_t buffer_size = std::max((64UL << 10), halfseg_buffers_ram /
    (2UL * different_used_pairs_of_halfsegments + n_free_buffers));

  fprintf(stderr, "      Different halfsegment pairs = %lu\n",
      different_used_pairs_of_halfsegments);
  fprintf(stderr, "      Buffer size = %lu\n", buffer_size);

  // Create a multifile writer for files
  // storing irreducible positions.
  typedef async_multi_stream_writer<text_offset_type>
    irr_pos_multiwriter_type;
  irr_pos_multiwriter_type *irr_pos_multiwriter =
    new irr_pos_multiwriter_type(
        2UL * different_used_pairs_of_halfsegments,
        buffer_size, n_free_buffers);

  // Create a map from used halfsegment pairs
  // to a contiguous range of integers. This
  // is needed to use multifile writer.
  std::uint64_t **halfseg_ids_to_pos_file_id =
    new std::uint64_t*[n_halfsegments];
  std::uint64_t **halfseg_ids_to_phi_file_id =
    new std::uint64_t*[n_halfsegments];

  {
    for (std::uint64_t i = 0; i < n_halfsegments; ++i) {
      halfseg_ids_to_pos_file_id[i] = new std::uint64_t[n_halfsegments];
      halfseg_ids_to_phi_file_id[i] = new std::uint64_t[n_halfsegments];
    }

    std::uint64_t file_counter = 0;
    for (std::uint64_t i = 0; i < n_halfsegments; ++i) {
      for (std::uint64_t j = i; j < n_halfsegments; ++j) {
        if (items_per_halfseg_pair[part_id][i][j] > 0) {
          halfseg_ids_to_pos_file_id[i][j] = file_counter;
          halfseg_ids_to_pos_file_id[j][i] = file_counter;
          irr_pos_multiwriter->add_file(pos_filenames[i][j]);
          ++file_counter;
          halfseg_ids_to_phi_file_id[i][j] = file_counter;
          halfseg_ids_to_phi_file_id[j][i] = file_counter;
          irr_pos_multiwriter->add_file(phi_filenames[i][j]);
          ++file_counter;
        }
      }
    }
  }

  // Initialize suffix array reader.
  typedef async_stream_reader<text_offset_type> sa_reader_type;
  sa_reader_type *sa_reader = new sa_reader_type(sa_filename);

  // Initialize BWT reader.
  typedef async_stream_reader<char_type> bwt_reader_type;
  bwt_reader_type *bwt_reader = new bwt_reader_type(bwt_filename);

  // Allocate buffers.
  static const std::uint64_t local_buffer_size = (1UL << 20);
  std::uint64_t *buf =
    utils::allocate_array<std::uint64_t>(local_buffer_size);

#ifdef _OPENMP
  std::uint64_t *tempbuf =
    utils::allocate_array<std::uint64_t>(local_buffer_size);
#endif

  // Distribution follows.
  char_type prev_bwt = 0;
  std::uint64_t filled = 0;
  std::uint64_t prev_sa = 0;
  std::uint64_t prev_halfseg_id = 0;
  std::uint64_t **toprocess =
    items_per_halfseg_pair[part_id];

  for (std::uint64_t i = 0; i < text_length; ++i) {
    std::uint64_t cur_sa = sa_reader->read();
    std::uint64_t cur_halfseg_id = cur_sa / max_halfsegment_size;
    char_type cur_bwt = bwt_reader->read();

    if ((i > 0) &&
        (cur_sa == 0 ||
         prev_sa == 0 ||
         cur_bwt != prev_bwt)) {

      if (part_id == 0) {
        buf[filled++] = cur_sa;
        if (filled == local_buffer_size) {

#ifdef _OPENMP
          set_bits(C, text_length, buf, filled, tempbuf);
#else
          set_bits(C, buf, filled);
#endif

          filled = 0;
        }
      }

      std::uint64_t h1 = cur_halfseg_id;
      std::uint64_t h2 = prev_halfseg_id;

      if (h1 > h2)
        std::swap(h1, h2);

      // Check if current item should be
      // processed in this text part.
      ++pairs_processed[h1][h2];
      if (pairs_processed[h1][h2] > (std::int64_t)0 &&
          pairs_processed[h1][h2] <= (std::int64_t)toprocess[h1][h2]) {

        // PLCP[cur_sa] is irreducible. Write
        // i and Phi[i] to appropriate files.
        std::uint64_t pos_file_id =
          halfseg_ids_to_pos_file_id[cur_halfseg_id][prev_halfseg_id];
        std::uint64_t phi_file_id =
          halfseg_ids_to_phi_file_id[cur_halfseg_id][prev_halfseg_id];
        irr_pos_multiwriter->write_to_ith_file(
            pos_file_id, (text_offset_type)cur_sa);
        irr_pos_multiwriter->write_to_ith_file(
            phi_file_id, (text_offset_type)prev_sa);
      }
    }

    // Update prev values.
    prev_halfseg_id = cur_halfseg_id;
    prev_sa = cur_sa;
    prev_bwt = cur_bwt;
  }

  // Flush the remaining items in the buffer.
  if (part_id == 0 && filled > 0) {

#ifdef _OPENMP
    set_bits(C, text_length, buf, filled, tempbuf);
#else
    set_bits(C, buf, filled);
#endif

    filled = 0;
  }

  // Add undefined bit and write C to disk.
  if (part_id == 0) {
    std::uint64_t offset = phi_undefined_position;
    C[offset >> 6] |= (1UL << (offset & 63));
    utils::write_to_file(C, C_size_in_words, C_filename);
  }

  // Stop I/O threads.
  sa_reader->stop_reading();
  bwt_reader->stop_reading();

  // Update I/O volume.
  io_volume +=
    sa_reader->bytes_read() +
    bwt_reader->bytes_read() +
    irr_pos_multiwriter->bytes_written();

  if (part_id == 0)
    io_volume +=
      C_size_in_words * sizeof(std::uint64_t);

  total_io_volume += io_volume;

  // Clean up.
#ifdef _OPENMP
  utils::deallocate(tempbuf);
#endif

  utils::deallocate(buf);
  delete bwt_reader;
  delete sa_reader;

  for (std::uint64_t i = n_halfsegments; i > 0; --i) {
    delete[] halfseg_ids_to_phi_file_id[i - 1];
    delete[] halfseg_ids_to_pos_file_id[i - 1];
  }

  delete[] halfseg_ids_to_phi_file_id;
  delete[] halfseg_ids_to_pos_file_id;

  delete irr_pos_multiwriter;

  for (std::uint64_t i = n_halfsegments; i > 0; --i)
    delete[] pairs_processed[i - 1];
  delete[] pairs_processed;

  if (part_id == 0)
    utils::deallocate(C);

  // Print summary.
  long double elapsed = utils::wclock() - start;
  fprintf(stderr, "      Summary: time = %.2Lfs, I/O = %.2LfMiB/s, "
      "total I/O vol = %.2Lfbytes/input symbol\n",
      elapsed, ((1.L * io_volume) / (1L << 20)) / elapsed,
      (1.L * total_io_volume) / text_length);
}

template<typename char_type,
  typename text_offset_type>
std::uint64_t distribute_pairs_and_compute_C_lex_partitioning(
    std::uint64_t text_length,
    std::uint64_t max_halfsegment_size,
    std::uint64_t ram_use,
    std::uint64_t cur_sa_range_beg,
    std::uint64_t max_pairs_written,
    std::uint64_t phi_undefined_position,
    std::string sa_filename,
    std::string bwt_filename,
    std::string C_filename,
    std::string **pos_filenames,
    std::string **phi_filenames,
    std::uint64_t &n_irreducible_lcps,
    std::uint64_t &total_io_volume) {

  // Print initial message.
  fprintf(stderr, "    Distribute (i, Phi[i]) pairs and compute C: ");

  // Initialize the timer.
  long double start = utils::wclock();

  // Initialize I/O volume.
  std::uint64_t io_volume = 0;

  // Compute basic parameters.
  std::uint64_t n_halfsegments =
    (text_length + max_halfsegment_size - 1) / max_halfsegment_size;
  std::uint64_t n_different_halfseg_pairs =
    (n_halfsegments * (n_halfsegments + 1)) / 2;
  std::uint64_t n_irreducible = 0;

  // Allocate bitvector C.
  std::uint64_t C_size_in_words = (text_length + 63) / 64;
  std::uint64_t C_size_in_bytes = (text_length + 7) / 8;
  std::uint64_t *C = utils::allocate_array<std::uint64_t>(C_size_in_words);

  bool C_exists = utils::file_exists(C_filename);
  if (C_exists) utils::read_from_file(C, C_size_in_words, C_filename);
  else std::fill(C, C + C_size_in_words, (std::uint64_t)0);

  // Create a map from used halfsegment pairs to a contiguous
  // range of integers. This is needed to use multifile writer.
  std::uint64_t **halfseg_ids_to_pos_file_id =
    new std::uint64_t*[n_halfsegments];
  std::uint64_t **halfseg_ids_to_phi_file_id =
    new std::uint64_t*[n_halfsegments];

  {
    for (std::uint64_t i = 0; i < n_halfsegments; ++i) {
      halfseg_ids_to_pos_file_id[i] = new std::uint64_t[n_halfsegments];
      halfseg_ids_to_phi_file_id[i] = new std::uint64_t[n_halfsegments];
    }

    std::uint64_t file_counter = 0;
    for (std::uint64_t i = 0; i < n_halfsegments; ++i) {
      for (std::uint64_t j = i; j < n_halfsegments; ++j) {
        halfseg_ids_to_pos_file_id[i][j] = file_counter;
        halfseg_ids_to_pos_file_id[j][i] = file_counter;
        ++file_counter;
        halfseg_ids_to_phi_file_id[i][j] = file_counter;
        halfseg_ids_to_phi_file_id[j][i] = file_counter;
        ++file_counter;
      }
    }
  }

  // Initialize multifile writer for files
  // storing irreducible positions.
  static const std::uint64_t n_free_buffers = 4;
  std::uint64_t halfseg_buffers_ram = ram_use - C_size_in_bytes;
  std::uint64_t buffer_size = std::max((1UL << 20),
      halfseg_buffers_ram /
      (2UL * n_different_halfseg_pairs + n_free_buffers));

  typedef async_multi_stream_writer<text_offset_type>
    irr_pos_multiwriter_type;
  irr_pos_multiwriter_type *irr_pos_multiwriter =
    new irr_pos_multiwriter_type(2UL * n_different_halfseg_pairs,
        buffer_size, n_free_buffers);

  for (std::uint64_t i = 0; i < n_halfsegments; ++i)
    for (std::uint64_t j = i; j < n_halfsegments; ++j) {
      irr_pos_multiwriter->add_file(pos_filenames[i][j]);
      irr_pos_multiwriter->add_file(phi_filenames[i][j]);
    }

  // Initialize suffix array reader.
  typedef async_stream_reader<text_offset_type> sa_reader_type;
  sa_reader_type *sa_reader =
    new sa_reader_type(sa_filename, cur_sa_range_beg);

  // Initialize BWT reader.
  typedef async_stream_reader<char_type> bwt_reader_type;
  bwt_reader_type *bwt_reader =
    new bwt_reader_type(bwt_filename, cur_sa_range_beg);

  // Allocate buffers.
  static const std::uint64_t local_buffer_size = (1UL << 20);
  std::uint64_t *buf =
    utils::allocate_array<std::uint64_t>(local_buffer_size);

#ifdef _OPENMP
  std::uint64_t *tempbuf =
    utils::allocate_array<std::uint64_t>(local_buffer_size);
#endif

  // Distribution follows.
  char_type prev_bwt = 0;
  std::uint64_t filled = 0;
  std::uint64_t prev_sa = 0;
  std::uint64_t prev_halfseg_id = 0;

  if (cur_sa_range_beg > 0) {
    text_offset_type val = 0;
    std::uint64_t bwt_file_offset =
      (cur_sa_range_beg - 1) * sizeof(char_type);
    std::uint64_t sa_file_offset =
      (cur_sa_range_beg - 1) * sizeof(text_offset_type);
    utils::read_at_offset(&prev_bwt, bwt_file_offset, 1, bwt_filename);
    utils::read_at_offset(&val, sa_file_offset, 1, sa_filename);
    io_volume += sizeof(text_offset_type) + sizeof(char_type);

    prev_sa = val;
    prev_halfseg_id = prev_sa / max_halfsegment_size;
  }

  std::uint64_t new_sa_range_beg = cur_sa_range_beg;
  std::uint64_t pairs_written = 0;
  while (pairs_written < max_pairs_written &&
      new_sa_range_beg < text_length) {
    std::uint64_t cur_sa = sa_reader->read();
    std::uint64_t cur_halfseg_id = cur_sa / max_halfsegment_size;
    char_type cur_bwt = bwt_reader->read();

    if ((new_sa_range_beg > 0) &&
        (cur_sa == 0 ||
         prev_sa == 0 ||
         cur_bwt != prev_bwt)) {

      // PLCP[cur_sa] is irreducible. Write
      // i and Phi[i] to appropriate files.
      ++n_irreducible;
      buf[filled++] = cur_sa;
      if (filled == local_buffer_size) {

#ifdef _OPENMP
        set_bits(C, text_length, buf, filled, tempbuf);
#else
        set_bits(C, buf, filled);
#endif

        filled = 0;
      }

      ++pairs_written;
      std::uint64_t pos_file_id =
        halfseg_ids_to_pos_file_id[cur_halfseg_id][prev_halfseg_id];
      std::uint64_t phi_file_id =
        halfseg_ids_to_phi_file_id[cur_halfseg_id][prev_halfseg_id];
      irr_pos_multiwriter->write_to_ith_file(
          pos_file_id, (text_offset_type)cur_sa);
      irr_pos_multiwriter->write_to_ith_file(
          phi_file_id, (text_offset_type)prev_sa);
    }

    prev_halfseg_id = cur_halfseg_id;
    prev_sa = cur_sa;
    prev_bwt = cur_bwt;
    ++new_sa_range_beg;
  }

  // Flush the remaining items in the buffer.
  if (filled > 0) {

#ifdef _OPENMP
    set_bits(C, text_length, buf, filled, tempbuf);
#else
    set_bits(C, buf, filled);
#endif

    filled = 0;
  }

  // Handle special case.
  {
    std::uint64_t offset = phi_undefined_position;
    C[offset >> 6] |= (1UL << (offset & 63));
  }

  // Account for phi_undefined_position.
  ++n_irreducible;

  // Write C to disk.
  utils::write_to_file(C,
      C_size_in_words, C_filename);

  // Stop I/O threads.
  sa_reader->stop_reading();
  bwt_reader->stop_reading();

  // Update I/O volume.
  io_volume +=
    sa_reader->bytes_read() +
    bwt_reader->bytes_read() +
    irr_pos_multiwriter->bytes_written() +
    C_size_in_words * sizeof(std::uint64_t);
  if (C_exists)
    io_volume += C_size_in_words * sizeof(std::uint64_t);
  total_io_volume += io_volume;

  // Clean up.
#ifdef _OPENMP
  utils::deallocate(tempbuf);
#endif

  utils::deallocate(buf);

  delete bwt_reader;
  delete sa_reader;
  delete irr_pos_multiwriter;

  for (std::uint64_t i = n_halfsegments; i > 0; --i) {
    delete[] halfseg_ids_to_phi_file_id[i - 1];
    delete[] halfseg_ids_to_pos_file_id[i - 1];
  }

  delete[] halfseg_ids_to_phi_file_id;
  delete[] halfseg_ids_to_pos_file_id;

  utils::deallocate(C);

  // Print summary.
  long double elapsed = utils::wclock() - start;
  fprintf(stderr, "time = %.2Lfs, I/O = %.2LfMiB/s, "
      "total I/O vol = %.2Lfbytes/input symbol\n",
      elapsed, ((1.L * io_volume) / (1L << 20)) / elapsed,
      (1.L * total_io_volume) / text_length);

  // Update reference variables.
  n_irreducible_lcps = n_irreducible;

  // Return result.
  return new_sa_range_beg;
}

}  // namespace inplace_mode
}  // namespace em_succinct_irreducible_private

#endif  // __SRC_EM_SUCCINCT_IRREDUCIBLE_SRC_DISTRIBUTE_PAIRS_AND_COMPUTE_C_HPP_INCLUDED
