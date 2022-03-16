/**
 * @file    src/em_succinct_irreducible_src/compute_lcp_array.hpp
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

#ifndef __SRC_EM_SUCCINCT_IRREDUCIBLE_SRC_COMPUTE_LCP_ARRAY_HPP_INCLUDED
#define __SRC_EM_SUCCINCT_IRREDUCIBLE_SRC_COMPUTE_LCP_ARRAY_HPP_INCLUDED

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <ctime>
#include <string>
#include <limits>
#include <algorithm>
#include <unistd.h>

#include "compute_plcp_bitvector.hpp"
#include "compute_lcp_from_plcp.hpp"
#include "utils.hpp"


namespace em_succinct_irreducible_private {
namespace normal_mode {

template<typename char_type,
  typename text_offset_type>
void compute_lcp_array(
    std::uint64_t text_length,
    std::uint64_t ram_use,
    std::string text_filename,
    std::string sa_filename,
    std::string bwt_filename,
    std::string output_filename,
    std::uint64_t &max_lcp,
    std::uint64_t &lcp_sum,
    std::uint64_t &n_irreducible_lcps,
    std::uint64_t &sum_irreducible_lcps,
    std::uint64_t &total_io_volume) {

  long double text_to_ram_ratio =
    (long double)text_length / (long double)ram_use;

  if (text_to_ram_ratio > 4.0L) {

    // Not enough RAM to hold B in RAM.
    std::string B_filename = output_filename +
      ".plcp." + utils::random_string_hash();
    compute_plcp_bitvector_large_B<char_type, text_offset_type>(text_length,
        ram_use, text_filename, sa_filename, bwt_filename, B_filename,
        n_irreducible_lcps, sum_irreducible_lcps, total_io_volume);
    fprintf(stderr, "\n");

    compute_lcp_from_plcp<text_offset_type>(text_length, ram_use,
        sa_filename, output_filename, B_filename, total_io_volume,
        max_lcp, lcp_sum);

  } else {

    // Enough RAM to hold B in RAM.
    std::uint64_t *B = compute_plcp_bitvector_small_B<char_type,
      text_offset_type>(text_length, ram_use, text_filename, sa_filename,
          bwt_filename, output_filename, n_irreducible_lcps,
          sum_irreducible_lcps, total_io_volume);
    fprintf(stderr, "\n");

    compute_lcp_from_plcp<text_offset_type>(text_length, ram_use,
        B, sa_filename, output_filename, total_io_volume, max_lcp,
        lcp_sum);
  }
}

}  // namespace normal_mode

namespace inplace_mode {

template<typename char_type,
  typename text_offset_type>
void compute_lcp_array(
    std::uint64_t text_length,
    std::uint64_t ram_use,
    std::string text_filename,
    std::string sa_filename,
    std::string bwt_filename,
    std::string output_filename,
    std::uint64_t &max_lcp,
    std::uint64_t &lcp_sum,
    std::uint64_t &n_irreducible_lcps,
    std::uint64_t &sum_irreducible_lcps,
    std::uint64_t &total_io_volume) {

  long double text_to_ram_ratio =
    (long double)text_length / (long double)ram_use;

  if (text_to_ram_ratio > 4.0L) {

    // Not enough RAM to hold B in RAM.
    std::string B_filename = output_filename +
      ".plcp." + utils::random_string_hash();
    compute_plcp_bitvector_large_B<char_type, text_offset_type>(text_length,
        ram_use, text_filename, sa_filename, bwt_filename, B_filename,
        n_irreducible_lcps, sum_irreducible_lcps, total_io_volume);
    fprintf(stderr, "\n");

    compute_lcp_from_plcp<text_offset_type>(text_length, ram_use,
        sa_filename, output_filename, B_filename, total_io_volume,
        max_lcp, lcp_sum);

  } else {

    // Enough RAM to hold B in RAM.
    std::uint64_t *B =
      compute_plcp_bitvector_small_B<char_type, text_offset_type>(
          text_length, ram_use, text_filename, sa_filename,
          bwt_filename, output_filename, n_irreducible_lcps,
          sum_irreducible_lcps, total_io_volume);
    fprintf(stderr, "\n");

    compute_lcp_from_plcp<text_offset_type>(text_length, ram_use,
        B, sa_filename, output_filename, total_io_volume, max_lcp,
        lcp_sum);
  }
}

}  // namespace inplace_mode

template<typename char_type,
  typename text_offset_type>
void compute_lcp_array(
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

  // Start the timer.
  long double global_start = utils::wclock();

  // Initialize basic parameters.
  std::uint64_t text_file_size = utils::file_size(text_filename);
  std::uint64_t text_length = text_file_size / sizeof(char_type);
  std::uint64_t lcp_sum = 0;
  std::uint64_t max_lcp = 0;
  std::uint64_t n_irreducible_lcps = 0;
  std::uint64_t sum_irreducible_lcps = 0;
  std::uint64_t total_io_volume = 0;

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

  // XXX Check if all types are sufficiently large.
  {

    // text_offset_type must be able to
    // hold values in range [0..text_length).
    std::uint64_t max_text_offset_type =
      std::numeric_limits<text_offset_type>::max();
    if (max_text_offset_type < text_length - 1) {
      fprintf(stderr, "\nError: text_offset_type is too small:\n"
          "\tnumeric_limits<text_offset_type>::max() = %lu\n"
          "\ttext length = %lu\n", max_text_offset_type, text_length);
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
  fprintf(stderr, "Mode = construct LCP array\n");
  fprintf(stderr, "Timestamp = %s", utils::get_timestamp().c_str());
  fprintf(stderr, "Text filename = %s\n", text_filename.c_str());
  fprintf(stderr, "SA filename = %s\n", sa_filename.c_str());
  fprintf(stderr, "BWT filename = %s\n", bwt_filename.c_str());
  fprintf(stderr, "Output (LCP) filename = %s\n", output_filename.c_str());
  fprintf(stderr, "Text length = %lu (%.2LfMiB)\n",
      text_length, (1.L * text_length * sizeof(char_type)) / (1L << 20));
  fprintf(stderr, "RAM use = %lu bytes (%.2LfMiB)\n",
      ram_use, (1.L * ram_use) / (1L << 20));
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

  // Compute LCP array.
  if (inplace_mode == true)
    inplace_mode::compute_lcp_array<char_type, text_offset_type>(text_length,
        ram_use, text_filename, sa_filename, bwt_filename, output_filename,
        max_lcp, lcp_sum, n_irreducible_lcps, sum_irreducible_lcps,
        total_io_volume);
  else
    normal_mode::compute_lcp_array<char_type, text_offset_type>(text_length,
        ram_use, text_filename, sa_filename, bwt_filename, output_filename,
        max_lcp, lcp_sum, n_irreducible_lcps, sum_irreducible_lcps,
        total_io_volume);

  // Print summary.
  long double total_time = utils::wclock() - global_start;
  long double avg_lcp = (long double)lcp_sum / text_length;
  fprintf(stderr, "\n\nComputation finished. Summary:\n");
  fprintf(stderr, "  Total time = %.2Lfs\n", total_time);
  fprintf(stderr, "  Relative time = %.2Lfus/input symbol\n",
      (1000000.L * total_time) / text_length);
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

  fprintf(stderr, "  Number of irreducible LCPs = %lu\n", n_irreducible_lcps);
  fprintf(stderr, "  Sum of irreducible LCPs = %lu\n", sum_irreducible_lcps);
  fprintf(stderr, "  Sum of all LCPs = %lu\n", lcp_sum);
  fprintf(stderr, "  Average LCP = %.2Lf\n", avg_lcp);
  fprintf(stderr, "  Maximal LCP = %lu\n", max_lcp);
}

}  // namespace em_succinct_irreducible_private

#endif  // __SRC_EM_SUCCINCT_IRREDUCIBLE_SRC_COMPUTE_LCP_ARRAY_HPP_INCLUDED
