/**
 * @file    src/em_succinct_irreducible_src/compute_plcp_bitvector.hpp
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

#ifndef __SRC_EM_SUCCINCT_IRREDUCIBLE_SRC_COMPUTE_PLCP_BITVECTOR_HPP_INCLUDED
#define __SRC_EM_SUCCINCT_IRREDUCIBLE_SRC_COMPUTE_PLCP_BITVECTOR_HPP_INCLUDED

#include <cstdio>
#include <cstdint>
#include <ctime>
#include <string>
#include <algorithm>
#include <omp.h>
#include <unistd.h>

#include "utils.hpp"
#include "distribute_pairs_and_compute_C.hpp"
#include "process_halfsegment_pairs.hpp"
#include "compute_B.hpp"


namespace em_succinct_irreducible_private {
namespace normal_mode {

// A version that returns the B
// bitvector as a file on disk.
template<typename char_type,
  typename text_offset_type>
void compute_plcp_bitvector_large_B(
    std::uint64_t text_length,
    std::uint64_t ram_use,
    std::string text_filename,
    std::string sa_filename,
    std::string bwt_filename,
    std::string B_filename,
    std::uint64_t &n_irreducible_lcps,
    std::uint64_t &sum_irreducible_lcps,
    std::uint64_t &total_io_volume) {

  // Print initial message, start timer
  // and initializer the I/O volume.
  fprintf(stderr, "Compute PLCP bitvector (dest = EM):\n");
  long double compute_plcp_bitvector_start = utils::wclock();
  std::uint64_t io_volume = 0;

  // Compute basic parameters.
  static const std::uint64_t max_overflow_size = (1UL << 20);
  std::uint64_t max_halfsegment_size =
    std::max(1UL, ram_use / (2UL * sizeof(char_type)));
  std::uint64_t n_halfsegments =
    (text_length + max_halfsegment_size - 1) / max_halfsegment_size;
  std::uint64_t n_different_halfsegment_pairs =
    (n_halfsegments * (n_halfsegments + 1)) / 2;

  // Instruction below allow text_offset_type
  // to encode offsets inside blocks of B.
  std::uint64_t max_block_size_B = std::min(text_length, ram_use << 3);
  if (max_block_size_B < 64) max_block_size_B = 64;
  else { while (max_block_size_B & 63) --max_block_size_B; }
  std::uint64_t n_blocks_B =
    (2UL * text_length + max_block_size_B - 1) / max_block_size_B;
  long double text_to_ram_ratio =
    (long double)text_length / (long double)ram_use;

  // Print info about halfsegments.
  fprintf(stderr, "  Halfsegment size = %lu (%.2LfMiB)\n",
      max_halfsegment_size,
      (1.L * max_halfsegment_size * sizeof(char_type)) / (1UL << 20));
  fprintf(stderr, "  Number of halfsegments = %lu\n", n_halfsegments);
  fprintf(stderr, "  Number of halfsegment pairs = %lu\n",
      n_different_halfsegment_pairs);

  // Initialize file names with halfsegment pairs.
  std::string **pairs_filenames = new std::string*[n_halfsegments];
  for (std::uint64_t i = 0; i < n_halfsegments; ++i) {
    pairs_filenames[i] = new std::string[n_halfsegments];
    for (std::uint64_t j = i; j < n_halfsegments; ++j) {
      std::string filename = B_filename +
        ".pairs." + utils::intToStr(i) + "_" + utils::intToStr(j);
      pairs_filenames[i][j] = filename;
    }
  }

  // Compute undefined Phi position.
  std::uint64_t phi_undefined_position = 0;
  {
    text_offset_type pos;
    utils::read_at_offset(&pos, 0, 1, sa_filename);
    io_volume += sizeof(text_offset_type);
    phi_undefined_position = pos;
  }

  // Distribute pairs (i, Phi[i]) such that PLCP[i] is irreducible
  // into files corresponding to different halfsegment pairs and
  // compute the C bitvector.
  std::string C_filename = B_filename + ".irreducible_positions_bv";
  if (text_to_ram_ratio > 8.0L) {

    // Distribute pairs.
    distribute_pairs<char_type, text_offset_type>(
        text_length, max_halfsegment_size, ram_use, sa_filename,
        bwt_filename, pairs_filenames, n_irreducible_lcps, io_volume);

    // Compute C.
    compute_C<char_type, text_offset_type>(text_length, max_halfsegment_size,
        ram_use, phi_undefined_position, pairs_filenames, sa_filename,
        bwt_filename, C_filename, io_volume);

  } else {

    // Distribute pairs and compute C.
    distribute_pairs_and_compute_C<char_type, text_offset_type>(
          text_length, max_halfsegment_size, ram_use, sa_filename,
          bwt_filename, C_filename, pairs_filenames,
          n_irreducible_lcps, io_volume);
  }

  // Initialize files with bitvector positions.
  std::string *irreducible_bits_filenames = new std::string[n_blocks_B];
  for (std::uint64_t block_id = 0; block_id < n_blocks_B; ++block_id) {
    std::string filename = B_filename +
      ".irreducible_bits_bv." + utils::intToStr(block_id);
    irreducible_bits_filenames[block_id] = filename;
  }

  // Process all pairs of halfsegments.
  sum_irreducible_lcps =
    process_halfsegment_pairs_large_B<char_type, text_offset_type>(
        text_filename, text_length, max_block_size_B,
        max_halfsegment_size, max_overflow_size, pairs_filenames,
        irreducible_bits_filenames, io_volume);

  // Clean up.
  for (std::uint64_t i = n_halfsegments; i > 0; --i)
    delete[] pairs_filenames[i - 1];
  delete[] pairs_filenames;

  // Compute B.
  compute_large_B<text_offset_type>(text_length,
      max_block_size_B, phi_undefined_position,
      B_filename, C_filename, irreducible_bits_filenames,
      io_volume);

  // Update I/O volume.
  total_io_volume += io_volume;

  // Clean up.
  delete[] irreducible_bits_filenames;

  // Print summary.
  long double compute_plcp_bitvector_time =
    utils::wclock() - compute_plcp_bitvector_start;
  fprintf(stderr, "  Summary: time = %.2Lfs, "
      "total I/O vol = %.2Lfbytes/input symbol\n",
      compute_plcp_bitvector_time,
      (1.L * io_volume) / text_length);
}

// A version, that returns a pointer to B
// bitvector. Requires at least 2n bits of RAM.
template<typename char_type,
  typename text_offset_type>
std::uint64_t* compute_plcp_bitvector_small_B(
    std::uint64_t text_length,
    std::uint64_t ram_use,
    std::string text_filename,
    std::string sa_filename,
    std::string bwt_filename,
    std::string output_filename,
    std::uint64_t &n_irreducible_lcps,
    std::uint64_t &sum_irreducible_lcps,
    std::uint64_t &total_io_volume) {

  // Print initial message and start the timer.
  fprintf(stderr, "Compute PLCP bitvector (dest = RAM):\n");
  long double compute_plcp_bitvector_start = utils::wclock();
  std::uint64_t io_volume = 0;

  // Compute basic parameters.
  long double ram_to_text_ratio =
    (long double)ram_use / (long double)text_length;
  std::uint64_t *B = NULL;

  if (ram_to_text_ratio < sizeof(char_type) + 0.375L) {

    // Compute undefined Phi position.
    std::uint64_t phi_undefined_position = 0;
    {
      text_offset_type pos;
      utils::read_at_offset(&pos, 0, 1, sa_filename);
      io_volume += sizeof(text_offset_type);
      phi_undefined_position = pos;
    }

    // Initialize basic parameters.
    static const std::uint64_t max_overflow_size = (1UL << 20);
    std::uint64_t max_halfsegment_size =
      std::max(1UL, ram_use / (2UL * sizeof(char_type)));
    std::uint64_t n_halfsegments =
      (text_length + max_halfsegment_size - 1) / max_halfsegment_size;
    std::uint64_t n_different_halfsegment_pairs =
      (n_halfsegments * (n_halfsegments + 1)) / 2;

    // Print info about halfsegments.
    fprintf(stderr, "  Halfsegment size = %lu (%.2LfMiB)\n",
        max_halfsegment_size,
        (1.L * max_halfsegment_size * sizeof(char_type)) / (1UL << 20));
    fprintf(stderr, "  Number of halfsegments = %lu\n", n_halfsegments);
    fprintf(stderr, "  Number of halfsegment pairs = %lu\n",
        n_different_halfsegment_pairs);

    // Initialize file names with halfsegment pairs.
    std::string **pairs_filenames = new std::string*[n_halfsegments];
    for (std::uint64_t i = 0; i < n_halfsegments; ++i) {
      pairs_filenames[i] = new std::string[n_halfsegments];
      for (std::uint64_t j = i; j < n_halfsegments; ++j) {
        std::string filename = output_filename + ".pairs." +
          utils::intToStr(i) + "_" + utils::intToStr(j);
        pairs_filenames[i][j] = filename;
      }
    }

    // Distribute pairs (i, Phi[i]) such that PLCP[i] is irreducible
    // into files corresponding to different halfsegment pairs and
    // compute the C bitvector.
    std::string C_filename = output_filename + ".irreducible_positions_bv";
    distribute_pairs_and_compute_C<char_type, text_offset_type>(text_length,
        max_halfsegment_size, ram_use, sa_filename, bwt_filename, C_filename,
        pairs_filenames, n_irreducible_lcps, io_volume);

    // Process all pairs of halfsegments.
    // XXX use base filename?
    std::string low_pos_filename = output_filename + ".low_pos";
    std::string high_pos_filename = output_filename + ".high_pos";
    sum_irreducible_lcps =
      process_halfsegment_pairs_small_B<char_type, text_offset_type>(
          text_filename, text_length, max_halfsegment_size,
          max_overflow_size, pairs_filenames, low_pos_filename,
          high_pos_filename, io_volume);

    // Clean up.
    for (std::uint64_t i = n_halfsegments; i > 0; --i)
      delete[] pairs_filenames[i - 1];
    delete[] pairs_filenames;

    // Allocate B.
    std::uint64_t B_size_in_words = (2UL * text_length + 63) / 64;
    B = utils::allocate_array<std::uint64_t>(B_size_in_words);
    std::fill(B, B + B_size_in_words, (std::uint64_t)0);

    // Compute B.
    compute_small_B<text_offset_type>(text_length, B,
        low_pos_filename, high_pos_filename, C_filename,
        phi_undefined_position, io_volume);

  } else {

    // Compute B.
    B = compute_very_small_B<char_type, text_offset_type>(
        text_length, text_filename, sa_filename, n_irreducible_lcps,
        sum_irreducible_lcps, io_volume);
  }

  // Update I/O volume.
  total_io_volume += io_volume;

  // Print summary.
  long double compute_plcp_bitvector_time =
    utils::wclock() - compute_plcp_bitvector_start;
  fprintf(stderr, "  Summary: time = %.2Lfs, "
      "total I/O vol = %.2Lfbytes/input symbol\n",
      compute_plcp_bitvector_time,
      (1.L * io_volume) / text_length);

  // Return pointer to B.
  return B;
}

template<typename char_type,
  typename text_offset_type>
void compute_plcp_bitvector(
    std::uint64_t text_length,
    std::uint64_t ram_use,
    std::string text_filename,
    std::string sa_filename,
    std::string bwt_filename,
    std::string output_filename,
    std::uint64_t &n_irreducible_lcps,
    std::uint64_t &sum_irreducible_lcps,
    std::uint64_t &total_io_volume) {

  long double text_to_ram_ratio =
    (long double)text_length / (long double)ram_use;

  if (text_to_ram_ratio > 4.0L) {

    // Not enough RAM to hold B in RAM.
    compute_plcp_bitvector_large_B<char_type, text_offset_type>(text_length,
        ram_use, text_filename, sa_filename, bwt_filename, output_filename,
        n_irreducible_lcps, sum_irreducible_lcps, total_io_volume);

  } else {

    // Enough RAM to hold B in RAM.
    std::uint64_t *B = compute_plcp_bitvector_small_B<char_type,
      text_offset_type>(text_length, ram_use, text_filename, sa_filename,
          bwt_filename, output_filename, n_irreducible_lcps,
          sum_irreducible_lcps, total_io_volume);

    // Write B to disk.
    {

      // Start the timer.
      fprintf(stderr, "Write PLCP bitvector to disk: ");
      long double write_plcp_start = utils::wclock();
      std::uint64_t io_volume = 0;

      // Write the data.
      std::uint64_t length_of_B_in_words = (2UL * text_length + 63) / 64;
      utils::write_to_file(B, length_of_B_in_words, output_filename);

      // Update I/O volume.
      io_volume += length_of_B_in_words * sizeof(std::uint64_t);
      total_io_volume += io_volume;

      // Print summary.
      long double write_plcp_time = utils::wclock() - write_plcp_start;
      fprintf(stderr, "time = %.2Lfs, I/O = %.2LfMiB/s, "
          "I/O vol = %.2Lfbytes/input symbol\n", write_plcp_time,
          ((1.L * io_volume) / (1L << 20)) / write_plcp_time,
          (1.L * io_volume) / text_length);
    }

    // Clean up.
    utils::deallocate(B);
  }
}

}  // namespace normal_mode

namespace inplace_mode {

// A version that returns the B
// bitvector as a file on disk.
template<typename char_type,
  typename text_offset_type>
void compute_plcp_bitvector_large_B(
    std::uint64_t text_length,
    std::uint64_t ram_use,
    std::string text_filename,
    std::string sa_filename,
    std::string bwt_filename,
    std::string B_filename,
    std::uint64_t &n_irreducible_lcps,
    std::uint64_t &sum_irreducible_lcps,
    std::uint64_t &total_io_volume) {

  // XXX pass to this function, should also use random_hash?
  std::string base_filename = B_filename;

  // Print initial message, start timer
  // and initialize I/O volume.
  fprintf(stderr, "Compute PLCP bitvector (dest = EM):\n");
  long double compute_plcp_bitvector_start = utils::wclock();
  std::uint64_t io_volume = 0;

  // Compute the size of block of B. The size
  // is compute so that text_offset_type
  // can encode offsets inside blocks of B.
  std::uint64_t max_block_size_B =
    std::min(text_length, ram_use << 3);
  if (max_block_size_B < 64) max_block_size_B = 64;
  else {
    while (max_block_size_B & 63)
      --max_block_size_B;
  }

  // Compute basic parameters.
  // XXX scale max overflow size, max halfsegment size, etc.
  static const std::uint64_t max_overflow_size = (1UL << 20);
  std::uint64_t max_halfsegment_size =
    std::max(1UL, ram_use / (2UL * sizeof(char_type)));
  std::uint64_t n_halfsegments =
    (text_length + max_halfsegment_size - 1) / max_halfsegment_size;
  std::uint64_t n_different_halfsegment_pairs =
    (n_halfsegments * (n_halfsegments + 1)) / 2;
  std::uint64_t n_blocks_B =
    (2UL * text_length + max_block_size_B - 1) / max_block_size_B;

  long double text_to_ram_ratio =
    (long double)text_length / (long double)ram_use;

  // Print info about halfsegments.
  fprintf(stderr, "  Halfsegment size = %lu (%.2LfMiB)\n",
      max_halfsegment_size,
      (1.L * max_halfsegment_size * sizeof(char_type)) / (1UL << 20));
  fprintf(stderr, "  Number of halfsegments = %lu\n", n_halfsegments);
  fprintf(stderr, "  Number of halfsegment pairs = %lu\n",
      n_different_halfsegment_pairs);

  // Initialize file names with halfsegment pairs.
  // There is a separate set of files containing values i
  // and a separate set of files containing values Phi[i].
  std::string **pos_filenames = new std::string*[n_halfsegments];
  std::string **phi_filenames = new std::string*[n_halfsegments];
  std::string **lcp_filenames = new std::string*[n_halfsegments];
  for (std::uint64_t i = 0; i < n_halfsegments; ++i) {
    pos_filenames[i] = new std::string[n_halfsegments];
    phi_filenames[i] = new std::string[n_halfsegments];
    lcp_filenames[i] = new std::string[n_halfsegments];
    for (std::uint64_t j = i; j < n_halfsegments; ++j) {
      std::string suffix = utils::intToStr(i) + "_"+ utils::intToStr(j);
      pos_filenames[i][j] = base_filename + ".pos." + suffix;
      phi_filenames[i][j] = base_filename + ".phi." + suffix;
      lcp_filenames[i][j] = base_filename + ".lcp." + suffix;
    }
  }

  // Initialize files with bitvector positions.
  std::string *irreducible_bits_filenames = new std::string[n_blocks_B];
  for (std::uint64_t block_id = 0; block_id < n_blocks_B; ++block_id) {
    std::string filename = B_filename +
      ".B." + utils::intToStr(block_id);
    irreducible_bits_filenames[block_id] = filename;
  }

  // Compute undefined Phi position.
  std::uint64_t phi_undefined_position = 0;
  {
    text_offset_type pos;
    utils::read_at_offset(&pos, 0, 1, sa_filename);
    io_volume += sizeof(text_offset_type);;
    phi_undefined_position = pos;
  }

  // Distribute pairs (i, Phi[i]) such that PLCP[i] is irreducible
  // into files corresponding to different halfsegment pairs and
  // compute the C bitvector.
  std::string C_filename = base_filename + ".C";

  if (ram_use * sizeof(text_offset_type) <
      text_length * sizeof(char_type)) {

    // Use text-partitioning.
    fprintf(stderr, "  Partitioning type = text-order\n");

    // Allocate the pairs counts.
    std::uint64_t **pair_count = new std::uint64_t*[n_halfsegments];
    for (std::uint64_t i = 0; i < n_halfsegments; ++i) {
      pair_count[i] = new std::uint64_t[n_halfsegments];
      std::fill(pair_count[i], pair_count[i] + n_halfsegments,
          (std::uint64_t)0);
    }

    // Compute the number of pairs for each pair of halfsegments.
    {

      // Print initial message.
      fprintf(stderr, "  Compute pair counts: ");

      // Start the timer.
      long double pair_count_start = utils::wclock();

      // Initialize I/O volume.
      std::uint64_t io_vol = 0;

      // Create BWT reader.
      typedef async_stream_reader<char_type> bwt_reader_type;
      bwt_reader_type *bwt_reader = new bwt_reader_type(bwt_filename);

      // Create SA reader.
      typedef async_stream_reader<text_offset_type> sa_reader_type;
      sa_reader_type *sa_reader = new sa_reader_type(sa_filename);

      // Compute counts.
      std::uint64_t prev_sa = 0;
      std::uint64_t prev_halfseg_id = 0;
      char_type prev_bwt = 0;
      for (std::uint64_t i = 0; i < text_length; ++i) {
        std::uint64_t cur_sa = sa_reader->read();
        std::uint64_t cur_halfseg_id = cur_sa / max_halfsegment_size;
        char_type cur_bwt = bwt_reader->read();

        // Update pair count.
        if (i > 0 &&
            (cur_sa == 0 ||
             prev_sa == 0 ||
             cur_bwt != prev_bwt))
          ++pair_count[prev_halfseg_id][cur_halfseg_id];

        // Update prev values.
        prev_sa = cur_sa;
        prev_bwt = cur_bwt;
        prev_halfseg_id = cur_halfseg_id;
      }

      // Accumulate the counts for
      // symmetric halfsegment pairs.
      for (std::uint64_t i = 0; i < n_halfsegments; ++i)
        for (std::uint64_t j = i + 1; j < n_halfsegments; ++j)
          pair_count[i][j] += pair_count[j][i];

      // Stop I/O threads.
      sa_reader->stop_reading();
      bwt_reader->stop_reading();

      // Update I/O volume.
      io_vol +=
        sa_reader->bytes_read() +
        bwt_reader->bytes_read();
      io_volume += io_vol;

      // Clean up.
      delete sa_reader;
      delete bwt_reader;

      // Print summary.
      long double pair_count_time = utils::wclock() - pair_count_start;
      fprintf(stderr, "time = %.2Lfs, I/O = %.2LfMiB/s, "
          "total I/O vol = %.2Lfbytes/input symbol\n", pair_count_time,
          ((1.L * io_vol) / (1L << 20)) / pair_count_time,
          (1.L * io_volume) / text_length);
    }

    // Compute the total number of pairs to process.
    std::uint64_t total_pair_count = 0;
    for (std::uint64_t i = 0; i < n_halfsegments; ++i)
      for (std::uint64_t j = i; j < n_halfsegments; ++j)
        total_pair_count += pair_count[i][j];

    // Compute the number of irreducible LCP values.
    // +1 accounts for LCP[0] which is not processed.
    n_irreducible_lcps = total_pair_count + 1;

    // Print the total number of pairs to process.
    fprintf(stderr, "  Number of pairs to process = %lu "
        "(%.2Lf%% of all text positions)\n",
        total_pair_count, (100.L * total_pair_count) / text_length);

    // Compute the number of text parts.
    std::uint64_t n_parts = 0;
    {
      std::uint64_t items_left = total_pair_count;
      while (items_left > 0) {

        // Compute the number of items we can process in this part.
        // If n' is the part size then the peak disk usage for 1st part
        // is at most 0.625n + (1 + 2 * sizeof(text_offset_type))n'. For
        // other parts it's the same, except we replace 0.625n with 0.875n
        // because we need to store the B bitvector (0.25n bytes). We want
        // this to be less than sizeof(text_offset_type) * text_length.
        long double c1 = sizeof(text_offset_type) -
          ((n_parts == 0) ? 0.625L : 0.875L);
        long double c2 = (1 + 2 * sizeof(text_offset_type));
        std::uint64_t cur_part_max_items = (c1 / c2) * text_length;
        std::uint64_t cur_part_items = std::min(cur_part_max_items,
            items_left);

        ++n_parts;
        items_left -= cur_part_items;
      }
    }

    // Compute the number of items to process in each part.
    std::uint64_t *items_per_part = new std::uint64_t[n_parts];
    {
      std::uint64_t items_left = total_pair_count;
      for (std::uint64_t part_id = 0; part_id < n_parts; ++part_id) {
        long double c1 = sizeof(text_offset_type) -
          ((part_id == 0) ? 0.625L : 0.875L);
        long double c2 = (1 + 2 * sizeof(text_offset_type));
        std::uint64_t cur_part_max_items = (c1 / c2) * text_length;
        std::uint64_t cur_part_items = std::min(cur_part_max_items,
          items_left);

        items_per_part[part_id] = cur_part_items;
        items_left -= cur_part_items;
      }
    }

    // Compute the number of items inside each
    // segment to process in each part.
    std::uint64_t ***items_per_halfseg_pair = new std::uint64_t**[n_parts];
    for (std::uint64_t part_id = 0; part_id < n_parts; ++part_id) {
      items_per_halfseg_pair[part_id] = new std::uint64_t*[n_halfsegments];
      for (std::uint64_t i = 0; i < n_halfsegments; ++i) {
        items_per_halfseg_pair[part_id][i] =
          new std::uint64_t[n_halfsegments];
        std::fill(items_per_halfseg_pair[part_id][i],
            items_per_halfseg_pair[part_id][i] + n_halfsegments,
            (std::uint64_t)0);
      }

      std::uint64_t items_left = items_per_part[part_id];
      for (std::uint64_t diff = 0; diff < n_halfsegments; ++diff) {
        for (std::uint64_t j = n_halfsegments; j > diff; --j) {
          std::uint64_t i = (j - 1) - diff;
          std::uint64_t count = std::min(items_left, pair_count[i][j - 1]);
          items_per_halfseg_pair[part_id][i][j - 1] = count;
          pair_count[i][j - 1] -= count;
          items_left -= count;

          if (items_left == 0)
            break;
        }

        if (items_left == 0)
          break;
      }
    }

    // Process all text parts.
    for (std::uint64_t part_id = 0; part_id < n_parts; ++part_id) {

      // Print initial message.
      fprintf(stderr, "  Process part %lu/%lu:\n", part_id + 1, n_parts);

      // Print info on the number of processed pairs.
      fprintf(stderr, "    Number of pairs to process = %lu\n",
          items_per_part[part_id]);

      // XXX This should be a more subtle condition.
      if (text_to_ram_ratio > 8.0L) {

        // Distribute pairs.
        distribute_pairs_text_partitioning<char_type, text_offset_type>(
            text_length, max_halfsegment_size, ram_use, part_id,
            items_per_halfseg_pair, sa_filename, bwt_filename, pos_filenames,
            phi_filenames, io_volume);

        // Compute C.
        compute_C<char_type, text_offset_type>(text_length,
            max_halfsegment_size, ram_use, phi_undefined_position,
            pos_filenames, C_filename, io_volume);
      } else {

        // Distribute pairs and compute C.
        distribute_pairs_and_compute_C_text_partitioning<char_type,
          text_offset_type>(text_length, max_halfsegment_size, ram_use,
              part_id, items_per_halfseg_pair, phi_undefined_position,
              sa_filename, bwt_filename, C_filename, pos_filenames,
              phi_filenames, io_volume);
      }

      // Process all pairs of halfsegments.
      sum_irreducible_lcps +=
        process_halfsegment_pairs_large_B<char_type, text_offset_type>(
            text_filename, text_length, max_halfsegment_size,
            max_overflow_size, pos_filenames, phi_filenames,
            lcp_filenames, io_volume);

      // Permute bitvector positions into blocks of B.
      permute_bitvector_positions<text_offset_type>(
          text_length, max_block_size_B, max_halfsegment_size,
          pos_filenames, lcp_filenames, irreducible_bits_filenames,
          io_volume);

      // Compute B.
      bool is_last_part = (part_id + 1 == n_parts);
      compute_large_B<text_offset_type>(
          text_length, max_block_size_B, phi_undefined_position,
          B_filename, C_filename, irreducible_bits_filenames,
          is_last_part, io_volume);
    }

    // Clean up.
    for (std::uint64_t part_id = n_parts; part_id > 0; --part_id) {
      for (std::uint64_t i = n_halfsegments; i > 0; --i)
        delete[] items_per_halfseg_pair[part_id - 1][i - 1];
      delete[] items_per_halfseg_pair[part_id - 1];
    }

    delete[] items_per_halfseg_pair;
    delete[] items_per_part;
    for (std::uint64_t i = n_halfsegments; i > 0; --i)
      delete[] pair_count[i - 1];

    delete[] pair_count;
  } else {

    // Use lex-partitioning.
    fprintf(stderr, "  Partitioning type = lex-order\n");

    // Process all lex parts.
    std::uint64_t cur_sa_range_beg = 0;
    std::uint64_t part_id = 0;
    while (cur_sa_range_beg < text_length) {

      // Print initial message.
      fprintf(stderr, "  Process part %lu (current progress = %.2Lf%%):\n",
          part_id + 1, (100.L * cur_sa_range_beg) / text_length);

      // Compute the number of items we can process in this part.
      // If n' is the part size then the peak disk usage for 1st part
      // is at most 0.625n + (1 + 2 * sizeof(text_offset_type))n'. For
      // other parts it's the same, except we replace 0.625n with 0.875n
      // because we need to store the B bitvector (0.25n bytes). We want
      // this to be less than sizeof(text_offset_type) * text_length.
      long double c1 = sizeof(text_offset_type) -
        ((cur_sa_range_beg == 0) ? 0.625L : 0.875L);
      long double c2 = (1 + 2 * sizeof(text_offset_type));
      std::uint64_t cur_part_max_items = (c1 / c2) * text_length;

      // Print info on the number of allowed items.
      fprintf(stderr, "    Max allowed items = %lu (%.2Lf%%)\n",
          cur_part_max_items, (100.L * cur_part_max_items) / text_length);

      // XXX This should be a more subtle condition.
      std::uint64_t local_n_irreducible_lcps = 0;
      std::uint64_t new_sa_range_beg = 0;
      if (text_to_ram_ratio > 8.0L) {

        // Distribute pairs.
        new_sa_range_beg =
          distribute_pairs_lex_partitioning<char_type, text_offset_type>(
              text_length, max_halfsegment_size, ram_use, cur_sa_range_beg,
              cur_part_max_items, sa_filename, bwt_filename, pos_filenames,
              phi_filenames, local_n_irreducible_lcps, io_volume);

        // Compute C.
        compute_C<char_type, text_offset_type>(text_length,
            max_halfsegment_size, ram_use, phi_undefined_position,
            pos_filenames, C_filename, io_volume);
      } else {

        // Distribute pairs and compute C.
        new_sa_range_beg =
          distribute_pairs_and_compute_C_lex_partitioning<char_type,
          text_offset_type>(text_length, max_halfsegment_size, ram_use,
              cur_sa_range_beg, cur_part_max_items, phi_undefined_position,
              sa_filename, bwt_filename, C_filename, pos_filenames,
              phi_filenames, local_n_irreducible_lcps, io_volume);
      }

      // Process all pairs of halfsegments.
      sum_irreducible_lcps +=
        process_halfsegment_pairs_large_B<char_type, text_offset_type>(
            text_filename, text_length, max_halfsegment_size,
            max_overflow_size, pos_filenames, phi_filenames,
            lcp_filenames, io_volume);

      // Permute bitvector positions into blocks of B.
      permute_bitvector_positions<text_offset_type>(
          text_length, max_block_size_B, max_halfsegment_size,
          pos_filenames, lcp_filenames, irreducible_bits_filenames,
          io_volume);

      // Compute B.
      bool is_last_part = (new_sa_range_beg == text_length);
      compute_large_B<text_offset_type>(
          text_length, max_block_size_B, phi_undefined_position,
          B_filename, C_filename, irreducible_bits_filenames,
          is_last_part, io_volume);

      // Update counters.
      n_irreducible_lcps += local_n_irreducible_lcps;
      ++part_id;
      cur_sa_range_beg = new_sa_range_beg;
    }
  }

  // Clean up.
  for (std::uint64_t i = n_halfsegments; i > 0; --i) {
    delete[] lcp_filenames[i - 1];
    delete[] phi_filenames[i - 1];
    delete[] pos_filenames[i - 1];
  }

  delete[] lcp_filenames;
  delete[] phi_filenames;
  delete[] pos_filenames;
  delete[] irreducible_bits_filenames;

  // Update I/O volume.
  total_io_volume += io_volume;

  // Print summary.
  long double compute_plcp_bitvector_time =
    utils::wclock() - compute_plcp_bitvector_start;
  fprintf(stderr, "  Summary: time = %.2Lfs, "
      "total I/O vol = %.2Lfbytes/input symbol\n",
      compute_plcp_bitvector_time,
      (1.L * io_volume) / text_length);
}

// A version, that returns a pointer to B
// bitvector. Requires at least 2n bits of RAM.
template<typename char_type,
  typename text_offset_type>
std::uint64_t* compute_plcp_bitvector_small_B(
    std::uint64_t text_length,
    std::uint64_t ram_use,
    std::string text_filename,
    std::string sa_filename,
    std::string bwt_filename,
    std::string output_filename,
    std::uint64_t &n_irreducible_lcps,
    std::uint64_t &sum_irreducible_lcps,
    std::uint64_t &total_io_volume) {

  std::string base_filename = output_filename;  // XXX random_hash?

  // Print initial message and start the timer.
  fprintf(stderr, "Compute PLCP bitvector (dest = RAM):\n");
  long double compute_plcp_bitvector_start = utils::wclock();
  std::uint64_t io_volume = 0;

  // Initialize basic parameters.
  long double ram_to_text_ratio =
    (long double)ram_use / (long double)text_length;
  std::uint64_t *B = NULL;

  if (ram_to_text_ratio < sizeof(char_type) + 0.375L) {

    // Compute undefined Phi position.
    std::uint64_t phi_undefined_position = 0;
    {
      text_offset_type pos;
      utils::read_at_offset(&pos, 0, 1, sa_filename);
      io_volume += sizeof(text_offset_type);
      phi_undefined_position = pos;
    }

    // Compute basic parameters.
    static const std::uint64_t max_overflow_size = (1UL << 20);
    std::uint64_t max_halfsegment_size =
      std::max(1UL, ram_use / (2UL * sizeof(char_type)));
    std::uint64_t n_halfsegments =
      (text_length + max_halfsegment_size - 1) / max_halfsegment_size;
    std::uint64_t n_different_halfsegment_pairs =
      (n_halfsegments * (n_halfsegments + 1)) / 2;

    // Print info about halfsegments.
    fprintf(stderr, "  Halfsegment size = %lu (%.2LfMiB)\n",
        max_halfsegment_size,
        (1.L * max_halfsegment_size * sizeof(char_type)) / (1UL << 20));
    fprintf(stderr, "  Number of halfsegments = %lu\n", n_halfsegments);
    fprintf(stderr, "  Number of halfsegment pairs = %lu\n",
        n_different_halfsegment_pairs);

    // Initialize file names with halfsegment pairs.
    // There is a separate set of files containing values i
    // and a separate set of files containing values Phi[i].
    std::string **pos_filenames = new std::string*[n_halfsegments];
    std::string **phi_filenames = new std::string*[n_halfsegments];
    std::string **lcp_filenames = new std::string*[n_halfsegments];
    for (std::uint64_t i = 0; i < n_halfsegments; ++i) {
      pos_filenames[i] = new std::string[n_halfsegments];
      phi_filenames[i] = new std::string[n_halfsegments];
      lcp_filenames[i] = new std::string[n_halfsegments];
      for (std::uint64_t j = i; j < n_halfsegments; ++j) {
        std::string suffix = utils::intToStr(i) + "_"+ utils::intToStr(j);
        pos_filenames[i][j] = base_filename + ".pos." + suffix;
        phi_filenames[i][j] = base_filename + ".phi." + suffix;
        lcp_filenames[i][j] = base_filename + ".lcp." + suffix;
      }
    }

    std::string C_filename =
      base_filename + ".irreducible_positions_bv";
    std::string B_filename =
      base_filename + ".temp_B";

    // Process all lex parts.
    std::uint64_t cur_sa_range_beg = 0;
    std::uint64_t part_id = 0;
    while (cur_sa_range_beg < text_length) {

      // Print initial message.
      fprintf(stderr, "  Process part %lu (current progress = %.2Lf%%):\n",
          part_id + 1, (100.L * cur_sa_range_beg) / text_length);

      // Compute the number of items we can process in this part.
      // If n' is the part size then the peak disk usage for 1st part
      // is at most 0.625n + (1 + 2 * sizeof(text_offset_type))n'. For
      // other parts it's the same, except we replace 0.625n with 0.875n
      // because we need to store the B bitvector (0.25n bytes). We want
      // this to be less than sizeof(text_offset_type) * text_length.
      long double c1 = sizeof(text_offset_type) -
        ((cur_sa_range_beg == 0) ? 0.625L : 0.875L);
      long double c2 = (1 + 2 * sizeof(text_offset_type));
      std::uint64_t cur_part_max_items = (c1 / c2) * text_length;

      // Print info on the number of allowed items.
      fprintf(stderr, "    Max allowed items = %lu (%.2Lf%%)\n",
          cur_part_max_items, (100.L * cur_part_max_items) / text_length);

      // Distribute pairs (i, Phi[i]) such that PLCP[i] is irreducible
      // into files corresponding to different halfsegment pairs and
      // compute the C bitvector.
      std::uint64_t local_n_irreducible_lcps = 0;
      std::uint64_t new_sa_range_beg =
        distribute_pairs_and_compute_C_lex_partitioning<char_type,
        text_offset_type>(text_length, max_halfsegment_size, ram_use,
            cur_sa_range_beg, cur_part_max_items, phi_undefined_position,
            sa_filename, bwt_filename, C_filename, pos_filenames,
            phi_filenames, local_n_irreducible_lcps, io_volume);

      // Process all pairs of halfsegments.
      sum_irreducible_lcps +=
        process_halfsegment_pairs_small_B<char_type, text_offset_type>(
            text_filename, text_length, max_halfsegment_size,
            max_overflow_size, pos_filenames, phi_filenames,
            lcp_filenames, io_volume);

      // Allocate B.
      std::uint64_t B_size_in_words = (2UL * text_length + 63) / 64;
      B = utils::allocate_array<std::uint64_t>(B_size_in_words);
      bool B_exists = utils::file_exists(B_filename);
      if (B_exists) {

        // Read B from disk.
        // XXX this updates I/O volume
        // should there be message?
        utils::read_from_file(B, B_size_in_words, B_filename);
        io_volume += B_size_in_words * sizeof(std::uint64_t);
      } else {

        // Zero-initialize B.
        std::fill(B, B + B_size_in_words, (std::uint64_t)0);
      }

      // Compute B.
      bool is_last_part = (new_sa_range_beg == text_length);
      compute_small_B<text_offset_type>(
          text_length, max_halfsegment_size, B, pos_filenames,
          lcp_filenames, C_filename, phi_undefined_position,
          is_last_part, io_volume);

      if (!is_last_part) {

        // Write B to disk.
        // XXX this updates I/O volume
        // should there be message?
        utils::write_to_file(B, B_size_in_words, B_filename);
        io_volume += B_size_in_words * sizeof(std::uint64_t);
        utils::deallocate(B);
      } else {

        // Delete the file with B from disk, if exists.
        if (utils::file_exists(B_filename))
          utils::file_delete(B_filename);
      }

      // Update counters.
      n_irreducible_lcps += local_n_irreducible_lcps;
      ++part_id;
      cur_sa_range_beg = new_sa_range_beg;
    }

    // Clean up.
    for (std::uint64_t i = n_halfsegments; i > 0; --i) {
      delete[] lcp_filenames[i - 1];
      delete[] phi_filenames[i - 1];
      delete[] pos_filenames[i - 1];
    }
    delete[] lcp_filenames;
    delete[] phi_filenames;
    delete[] pos_filenames;

  } else {

    // Compute B.
    B = compute_very_small_B<char_type, text_offset_type>(
        text_length, text_filename, sa_filename, n_irreducible_lcps,
        sum_irreducible_lcps, io_volume);
  }

  // Update I/O volume.
  total_io_volume += io_volume;

  // Print summary.
  long double compute_plcp_bitvector_time =
    utils::wclock() - compute_plcp_bitvector_start;
  fprintf(stderr, "  Summary: time = %.2Lfs, "
      "total I/O vol = %.2Lfbytes/input symbol\n",
      compute_plcp_bitvector_time,
      (1.L * io_volume) / text_length);

  // Return pointer to B.
  return B;
}

template<typename char_type,
  typename text_offset_type>
void compute_plcp_bitvector(
    std::uint64_t text_length,
    std::uint64_t ram_use,
    std::string text_filename,
    std::string sa_filename,
    std::string bwt_filename,
    std::string output_filename,
    std::uint64_t &n_irreducible_lcps,
    std::uint64_t &sum_irreducible_lcps,
    std::uint64_t &total_io_volume) {

  long double text_to_ram_ratio =
    (long double)text_length / (long double)ram_use;

  if (text_to_ram_ratio > 4.0L) {

    // Not enough RAM to hold B in RAM.
    compute_plcp_bitvector_large_B<char_type, text_offset_type>(
        text_length, ram_use, text_filename, sa_filename, bwt_filename,
        output_filename, n_irreducible_lcps, sum_irreducible_lcps,
        total_io_volume);

  } else {

    // Enough RAM to hold B in RAM.
    std::uint64_t *B = compute_plcp_bitvector_small_B<char_type,
      text_offset_type>(text_length, ram_use,
          text_filename, sa_filename, bwt_filename, output_filename,
          n_irreducible_lcps, sum_irreducible_lcps, total_io_volume);

    // Write B to disk.
    {

      // Start the timer.
      fprintf(stderr, "Write PLCP bitvector to disk: ");
      long double write_plcp_start = utils::wclock();
      std::uint64_t io_volume = 0;

      // Write the data.
      std::uint64_t length_of_B_in_words = (2UL * text_length + 63) / 64;
      utils::write_to_file(B, length_of_B_in_words, output_filename);

      // Update I/O volume.
      io_volume += length_of_B_in_words * sizeof(std::uint64_t);
      total_io_volume += io_volume;

      // Print summary.
      long double write_plcp_time = utils::wclock() - write_plcp_start;
      fprintf(stderr, "time = %.2Lfs, I/O = %.2LfMiB/s, "
          "I/O vol = %.2Lfbytes/input symbol\n", write_plcp_time,
          ((1.L * io_volume) / (1L << 20)) / write_plcp_time,
          (1.L * io_volume) / text_length);
    }

    // Clean up.
    utils::deallocate(B);
  }
}

}  // namespace inplace_mode

template<typename char_type,
  typename text_offset_type>
void compute_plcp_bitvector(
    std::string text_filename,
    std::string sa_filename,
    std::string bwt_filename,
    std::string output_filename,
    std::uint64_t ram_use,
    bool inplace_mode) {

  // Empty page cache and initialize
  // (pseudo)random number generator.
  srand(time(0) + getpid());
  utils::initialize_stats();
  utils::empty_page_cache(text_filename);
  utils::empty_page_cache(sa_filename);
  utils::empty_page_cache(bwt_filename);

  // Start the timer and initialize the I/O volume.
  long double global_start = utils::wclock();
  std::uint64_t total_io_volume = 0;

  // Compute basic parameters.
  std::uint64_t text_file_size = utils::file_size(text_filename);
  std::uint64_t text_length = text_file_size / sizeof(char_type);
  std::uint64_t n_irreducible_lcps = 0;
  std::uint64_t sum_irreducible_lcps = 0;

  // Check if the text is non-empty.
  if (text_length == 0) {
    fprintf(stderr, "Error: the input file is empty!\n");
    std::exit(EXIT_FAILURE);
  }

  // Sanity check.
  if (text_file_size % sizeof(char_type)) {
    fprintf(stderr, "\nError: size of text file is not a "
        "multiple of sizeof(char_type)!\n");
    std::exit(EXIT_FAILURE);
  }

  // Check if all types are sufficiently large.
  // XXX are these assumption correct?
  {

    // text_offset_type must be able to
    // hold values in range [0..text_length).
    std::uint64_t text_offset_type_max_value =
      std::numeric_limits<text_offset_type>::max();
    if (text_offset_type_max_value < text_length - 1) {
      fprintf(stderr, "\nError: text_offset_type is too small:\n"
          "\tnumeric_limits<text_offset_type>::max() = %lu\n"
          "\ttext length = %lu\n",
          (std::uint64_t)std::numeric_limits<text_offset_type>::max(),
          text_length);

      std::exit(EXIT_FAILURE);
    }
  }

  // Turn paths absolute.
  text_filename = utils::absolute_path(text_filename);
  sa_filename = utils::absolute_path(sa_filename);
  bwt_filename = utils::absolute_path(bwt_filename);
  output_filename = utils::absolute_path(output_filename);

  // Print summary of basic parameters.
  fprintf(stderr, "Running EM-SuccinctIrreducible v0.2.0\n");
  fprintf(stderr, "Mode = construct PLCP bitvector\n");
  fprintf(stderr, "Timestamp = %s", utils::get_timestamp().c_str());
  fprintf(stderr, "Text filename = %s\n", text_filename.c_str());
  fprintf(stderr, "SA filename = %s\n", sa_filename.c_str());
  fprintf(stderr, "BWT filename = %s\n", bwt_filename.c_str());
  fprintf(stderr, "Output (PLCP) filename = %s\n",
      output_filename.c_str());
  fprintf(stderr, "Text length = %lu (%.2LfMiB)\n", text_length,
      (1.L * text_length * sizeof(char_type)) / (1 << 20));
  fprintf(stderr, "RAM use = %lu bytes (%.2LfMiB)\n",
      ram_use, ram_use / (1024.L * 1024));
  fprintf(stderr, "sizeof(char_type) = %lu\n", sizeof(char_type));
  fprintf(stderr, "sizeof(text_offset_type) = %lu\n",
      sizeof(text_offset_type));
  fprintf(stderr, "Inplace mode = %s\n", inplace_mode ? "ON" : "OFF");
  fprintf(stderr, "Parallel mode = ");

#ifdef _OPENMP
  fprintf(stderr, "ON\n");
  fprintf(stderr, "Number of threads = %d\n", omp_get_max_threads());
#else
  fprintf(stderr, "OFF\n");
#endif

  fprintf(stderr, "\n\n");

  if (inplace_mode == true)
    inplace_mode::compute_plcp_bitvector<char_type, text_offset_type>(
        text_length, ram_use, text_filename, sa_filename, bwt_filename,
        output_filename, n_irreducible_lcps, sum_irreducible_lcps,
        total_io_volume);
  else
    normal_mode::compute_plcp_bitvector<char_type, text_offset_type>(
        text_length, ram_use, text_filename, sa_filename, bwt_filename,
        output_filename, n_irreducible_lcps, sum_irreducible_lcps,
        total_io_volume);

  // Print summary.
  long double total_time = utils::wclock() - global_start;
  fprintf(stderr, "\n\nComputation finished. Summary:\n");
  fprintf(stderr, "  Total time = %.2Lfs\n", total_time);
  fprintf(stderr, "  Relative time = %.2Lfus/input symbol\n",
      (1000000.0 * total_time) / text_length);
  fprintf(stderr, "  I/O volume = %lu bytes (%.2Lfbytes/input symbol)\n",
      total_io_volume, (1.L * total_io_volume) / text_length);

#ifdef MONITOR_DISK_USAGE
  fprintf(stderr, "  Internal I/O volume counter = %lu\n",
      utils::get_current_io_volume());
#endif

  fprintf(stderr, "  RAM allocation: cur = %lu bytes, peak = %.2LfMiB\n",
      utils::get_current_ram_allocation(),
      (1.L * utils::get_peak_ram_allocation()) / (1UL << 20));

#ifdef MONITOR_DISK_USAGE
  fprintf(stderr, "  Disk allocation: cur = %.2LfGiB, peak = %.2LfGiB\n",
      (1.L * utils::get_current_disk_allocation()) / (1UL << 30),
      (1.L * utils::get_peak_disk_allocation()) / (1UL << 30));
#endif

  fprintf(stderr, "  Number of irreducible LCPs = %lu\n",
      n_irreducible_lcps);
  fprintf(stderr, "  Sum of irreducible LCPs = %lu\n",
      sum_irreducible_lcps);
}

}  // namespace em_succinct_irreducible_private

#endif  // __SRC_EM_SUCCINCT_IRREDUCIBLE_SRC_COMPUTE_PLCP_BITVECTOR_HPP_INCLUDED
