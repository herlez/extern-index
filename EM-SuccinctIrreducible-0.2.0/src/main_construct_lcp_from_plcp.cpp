/**
 * @file    src/main_construct_lcp_from_plcp.cpp
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

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <sstream>
#include <string>
#include <getopt.h>
#include <unistd.h>

#include "em_succinct_irreducible_src/compute_lcp_from_plcp.hpp"
#include "uint24.hpp"
#include "uint40.hpp"
#include "uint48.hpp"
#include "uint56.hpp"


char *program_name;

void usage(int status) {
  printf(

"Usage: %s [OPTION]... PLCPFILE SAFILE\n"
"Compute the LCP array, given the PLCP array (bitvector representation) and the\n"
"suffix array stored in, respectively, PLCPFILE and SAFILE.\n"
"\n"
"Mandatory arguments to long options are mandatory for short options too.\n"
"  -f, --fast              enable faster computation at the price of higher disk\n"
"                          use (the algorithm in this mode is no longer inplace)\n"
"  -h, --help              display this help and exit\n"
"  -i, --intsize=SIZE      use integers of SIZE bytes. Int type needs to be wide\n"
"                          enough to encode any position in FILE. Default: 5.\n"
"                          Currently supported values are: 4-8\n"
"  -m, --mem=MEM           use MEM bytes of RAM for computation. Metric and IEC\n"
"                          suffixes are recognized, e.g., -l 10k, -l 1Mi, -l 3G\n"
"                          gives MEM = 10^4, 2^20, 3*10^6. Default: 3584Mi\n"
"  -o, --output=OUTFILE    specify output filename. Default: PLCPFILE.lcpX,\n"
"                          where X = integer size, see the -i flag\n",

    program_name);

  std::exit(status);
}

bool file_exists(std::string filename) {
  std::FILE *f = std::fopen(filename.c_str(), "r");
  bool ret = (f != NULL);
  if (f != NULL) std::fclose(f);

  return ret;
}

template<typename int_type>
std::string intToStr(int_type x) {
  std::stringstream ss;
  ss << x;
  return ss.str();
}

template<typename int_type>
bool parse_number(char *str, int_type *ret) {
  *ret = 0;
  std::uint64_t n_digits = 0;
  std::uint64_t str_len = std::strlen(str);
  while (n_digits < str_len && std::isdigit(str[n_digits])) {
    std::uint64_t digit = str[n_digits] - '0';
    *ret = (*ret) * 10 + digit;
    ++n_digits;
  }

  if (n_digits == 0)
    return false;

  std::uint64_t suffix_length = str_len - n_digits;
  if (suffix_length > 0) {
    if (suffix_length > 2)
      return false;

    for (std::uint64_t j = 0; j < suffix_length; ++j)
      str[n_digits + j] = std::tolower(str[n_digits + j]);
    if (suffix_length == 2 && str[n_digits + 1] != 'i')
      return false;

    switch(str[n_digits]) {
      case 'k':
        if (suffix_length == 1)
          *ret *= 1000;
        else
          *ret <<= 10;
        break;
      case 'm':
        if (suffix_length == 1)
          *ret *= 1000000;
        else
          *ret <<= 20;
        break;
      case 'g':
        if (suffix_length == 1)
          *ret *= 1000000000;
        else
          *ret <<= 30;
        break;
      case 't':
        if (suffix_length == 1)
          *ret *= 1000000000000;
        else
          *ret <<= 40;
        break;
      default:
        return false;
    }
  }

  return true;
}

int main(int argc, char **argv) {
  srand(time(0) + getpid());
  program_name = argv[0];

  static struct option long_options[] = {
    {"fast",    no_argument,       NULL, 'f'},
    {"help",    no_argument,       NULL, 'h'},
    {"intsize", required_argument, NULL, 'i'},
    {"mem",     required_argument, NULL, 'm'},
    {"output",  required_argument, NULL, 'o'},
    {NULL, 0, NULL, 0}
  };

  std::uint64_t int_size = 5;
  std::uint64_t ram_use = 3584UL << 20;
  std::string output_filename("");
  bool inplace_mode = true;

  // Parse command-line options.
  int c;
  while ((c = getopt_long(argc, argv,
          "fhi:m:o:", long_options, NULL)) != -1) {
    switch(c) {
      case 'f':
        inplace_mode = false;
        break;
      case 'h':
        usage(EXIT_FAILURE);
      case 'i':
        int_size = std::atol(optarg);
        if (int_size < 4 || int_size > 8) {
          fprintf(stderr, "Error: invalid int size (%lu)\n\n", int_size);
          usage(EXIT_FAILURE);
        }
        break;
      case 'm':
        {
          bool ok = parse_number(optarg, &ram_use);
          if (!ok) {
            fprintf(stderr, "Error: parsing phrase length "
                "limit (%s) failed\n\n", optarg);
            usage(EXIT_FAILURE);
          }
          if (ram_use == 0) {
            fprintf(stderr, "Error: invalid RAM limit (%lu)\n\n", ram_use);
            usage(EXIT_FAILURE);
          }
          break;
        }
      case 'o':
        output_filename = std::string(optarg);
        break;
      default:
        usage(EXIT_FAILURE);
    }
  }

  if (optind >= argc) {
    fprintf(stderr, "Error: PLCPFILE and SAFILE not provided\n\n");
    usage(EXIT_FAILURE);
  }

  // Parse the PLCP filename.
  std::string plcp_filename = std::string(argv[optind++]);

  if (optind >= argc) {
    fprintf(stderr, "Error: SAFILE not provided\n\n");
    usage(EXIT_FAILURE);
  }

  // Parse suffix array filename.
  std::string sa_filename = std::string(argv[optind++]);

  // Set default filenames (if not provided).
  if (output_filename.empty())
    output_filename = plcp_filename + ".lcp" + intToStr(int_size);

  // Check if PLCP and suffix array exist.
  if (!file_exists(plcp_filename)) {
    fprintf(stderr, "Error: PLCP file (%s) does not exist\n\n",
        plcp_filename.c_str());
    usage(EXIT_FAILURE);
  }
  if (!file_exists(sa_filename)) {
    fprintf(stderr, "Error: suffix array (%s) does not exist\n\n",
        sa_filename.c_str());
    usage(EXIT_FAILURE);
  }

  if (file_exists(output_filename)) {

    // Output file exists, should we proceed?
    char *line = NULL;
    std::uint64_t buflen = 0;
    std::int64_t len = 0L;

    do {
      printf("Output file (%s) exists. Overwrite? [y/n]: ",
          output_filename.c_str());
      if ((len = getline(&line, &buflen, stdin)) == -1) {
        printf("\nError: failed to read answer\n\n");
        std::fflush(stdout);
        usage(EXIT_FAILURE);
      }
    } while (len != 2 || (line[0] != 'y' && line[0] != 'n'));

    if (line[0] == 'n') {
      free(line);
      std::exit(EXIT_FAILURE);
    }
    free(line);
  }

  // Run the algorithm.
  if (int_size == 4)
    em_succinct_irreducible_private::compute_lcp_from_plcp<std::uint32_t>(
        plcp_filename, sa_filename, output_filename, ram_use, inplace_mode);
  else if (int_size == 5)
    em_succinct_irreducible_private::compute_lcp_from_plcp<uint40>(
        plcp_filename, sa_filename, output_filename, ram_use, inplace_mode);
  else if (int_size == 6)
    em_succinct_irreducible_private::compute_lcp_from_plcp<uint48>(
        plcp_filename, sa_filename, output_filename, ram_use, inplace_mode);
  else if (int_size == 7)
    em_succinct_irreducible_private::compute_lcp_from_plcp<uint56>(
        plcp_filename, sa_filename, output_filename, ram_use, inplace_mode);
  else
    em_succinct_irreducible_private::compute_lcp_from_plcp<std::uint64_t>(
        plcp_filename, sa_filename, output_filename, ram_use, inplace_mode);
}
