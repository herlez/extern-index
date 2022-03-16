/**
 * @file    src/em_succinct_irreducible_src/compute_lcp_from_plcp.hpp
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

#ifndef __SRC_EM_SUCCINCT_IRREDUCIBLE_SRC_COMPUTE_LCP_FROM_PLCP_HPP_INCLUDED
#define __SRC_EM_SUCCINCT_IRREDUCIBLE_SRC_COMPUTE_LCP_FROM_PLCP_HPP_INCLUDED

#include <cstdio>
#include <cstdint>
#include <string>
#include <algorithm>
#include <omp.h>

#include "convert_to_vbyte_slab.hpp"
#include "io/async_stream_reader.hpp"
#include "io/async_stream_vbyte_reader_multipart.hpp"
#include "io/async_stream_writer.hpp"
#include "io/async_stream_writer_multipart.hpp"
#include "io/async_stream_writer_multipart_2.hpp"
#include "io/async_multi_stream_vbyte_reader.hpp"
#include "io/async_multi_stream_writer.hpp"
#include "io/async_multi_stream_reader_multipart.hpp"
#include "utils.hpp"


namespace em_succinct_irreducible_private {
namespace normal_mode {

template<typename text_offset_type>
void compute_lcp_from_plcp(
    std::uint64_t text_length,
    std::uint64_t ram_use,
    std::string sa_filename,
    std::string output_filename,
    std::string B_filename,
    std::uint64_t &global_io_volume,
    std::uint64_t &max_lcp,
    std::uint64_t &lcp_sum,
    bool keep_plcp = false) {

  // Print initial message and start the timer.
  fprintf(stderr, "Convert PLCP to LCP:\n");
  long double convert_plcp_to_lcp_start = utils::wclock();

  // Compute basic parameters.
  std::uint64_t max_block_size = ram_use / sizeof(text_offset_type);
  std::uint64_t n_blocks =
    (text_length + max_block_size - 1) / max_block_size;
  std::uint64_t local_lcp_sum = 0;
  std::uint64_t local_max_lcp = 0;
  std::uint64_t total_io_volume = 0;

  // Print info about blocks.
  fprintf(stderr, "  Block size = %lu\n", max_block_size);
  fprintf(stderr, "  Number of blocks = %lu\n", n_blocks);

  // Set the filenames of files storing SA and LCP subsequences.
  std::string *sa_subsequences_filenames = new std::string[n_blocks];
  std::string *lcp_subsequences_filenames = new std::string[n_blocks];
  for (std::uint64_t block_id = 0; block_id < n_blocks; ++block_id) {
    sa_subsequences_filenames[block_id] = output_filename + ".sa_subseq." +
      utils::intToStr(block_id) + "." + utils::random_string_hash();
    lcp_subsequences_filenames[block_id] = output_filename + ".lcp_sebseq." +
      utils::intToStr(block_id) + "." + utils::random_string_hash();
  }

  // Compute SA subsequences.
  {

    // Print initial message, start the
    // timer and initialize I/O volume.
    fprintf(stderr, "  Compute SA subsequences: ");
    long double compute_sa_subseq_start = utils::wclock();
    std::uint64_t io_volume = 0;

    // Initialize streaming of suffix array.
    typedef async_stream_reader<text_offset_type> sa_reader_type;
    sa_reader_type *sa_reader = new sa_reader_type(sa_filename);

    // Initialize multifile writer of SA subsequences.
    static const std::uint64_t n_free_buffers = 4;
    std::uint64_t total_buffers_ram = ram_use;
    std::uint64_t buffer_size = std::min((16UL << 20),
        total_buffers_ram / (n_blocks + n_free_buffers));
    typedef async_multi_stream_writer<text_offset_type>
      sa_multiwriter_type;
    sa_multiwriter_type *sa_multiwriter =
      new sa_multiwriter_type(n_blocks, buffer_size, n_free_buffers);
    for (std::uint64_t block_id = 0; block_id < n_blocks; ++block_id)
      sa_multiwriter->add_file(sa_subsequences_filenames[block_id]);

    // Read SA / write SA subsequences.
    for (std::uint64_t j = 0; j < text_length; ++j) {
      std::uint64_t sa_j = sa_reader->read();
      std::uint64_t block_id = sa_j / max_block_size;
      sa_multiwriter->write_to_ith_file(block_id, sa_j);
    }

    // Stop I/O threads.
    sa_reader->stop_reading();

    // Update I/O volume.
    io_volume +=
      sa_reader->bytes_read() +
      sa_multiwriter->bytes_written();
    total_io_volume += io_volume;

    // Clean up.
    delete sa_multiwriter;
    delete sa_reader;

    // Print summary.
    long double compute_sa_subseq_time =
      utils::wclock() - compute_sa_subseq_start;
    fprintf(stderr, "time = %.2Lfs, I/O = %.2LfMiB/s, "
        "total I/O vol = %.2Lfbytes/input symbol\n",
        compute_sa_subseq_time,
        ((1.L * io_volume) / (1L << 20)) / compute_sa_subseq_time,
        (1.L * total_io_volume) / text_length);
  }

  // Compute LCP subsequences.
  {

    // Print initial message, start the
    // timer, and initialize I/O volume.
    fprintf(stderr, "  Compute LCP subsequences: ");
    long double compute_lcp_subseq_start = utils::wclock();
    std::uint64_t io_volume = 0;

    // Create the reader of PLCP bitvector.
    typedef async_stream_reader<std::uint64_t>
      plcp_bitvector_reader_type;
    plcp_bitvector_reader_type *plcp_bitvector_reader =
      new plcp_bitvector_reader_type(B_filename);

    // Initialize the buffer.
    std::uint64_t bitbuf = plcp_bitvector_reader->read();
    std::uint64_t bitpos = 0;
    std::uint64_t cur_plcp = 1;

    // Allocate the array holding the block of PLCP.
    text_offset_type *plcp_block =
      utils::allocate_array<text_offset_type>(max_block_size);

    // Allocate buffer.
    static const std::uint64_t buffer_size = (1UL << 20);
    text_offset_type *buf =
      utils::allocate_array<text_offset_type>(buffer_size);
    text_offset_type *outbuf =
      utils::allocate_array<text_offset_type>(buffer_size);

    // Process blocks left to right.
    for (std::uint64_t block_id = 0; block_id < n_blocks; ++block_id) {
      std::uint64_t block_beg = block_id * max_block_size;
      std::uint64_t block_end =
        std::min(block_beg + max_block_size, text_length);
      std::uint64_t block_size = block_end - block_beg;

      // Read a block of PLCP into RAM.
      for (std::uint64_t j = 0; j < block_size; ++j) {

        // Increment cur_plcp for every 0 in the bitvector.
        while ((bitbuf & (1UL << bitpos)) == 0) {
          ++cur_plcp;
          ++bitpos;
          if (bitpos == 64) {
            bitbuf = plcp_bitvector_reader->read();
            bitpos = 0;
          }
        }

        // We decrement last because cur_plcp is unsigned.
        --cur_plcp;
        plcp_block[j] = cur_plcp;

        // Skip the 1-bit in the bitvector.
        ++bitpos;
        if (bitpos == 64) {
          if (plcp_bitvector_reader->empty() == false)
            bitbuf = plcp_bitvector_reader->read();
          bitpos = 0;
        }
      }

      // Compute LCP subsequence and write to file.
      {

        // Initialize SA subsequence reader.
        typedef async_stream_reader<text_offset_type> sa_subseq_reader_type;
        sa_subseq_reader_type *sa_subseq_reader =
          new sa_subseq_reader_type(sa_subsequences_filenames[block_id]);

        // Initialize LCP subsequence writer.
        std::uint64_t single_file_max_bytes = text_length / (n_blocks * 2UL);
        typedef async_stream_writer_multipart<text_offset_type>
          lcp_subseq_writer_type;
        lcp_subseq_writer_type *lcp_subseq_writer =
          new lcp_subseq_writer_type(
              lcp_subsequences_filenames[block_id],
              single_file_max_bytes);

        // Compute LCP subsequence.
        std::uint64_t subseq_size =
          utils::file_size(sa_subsequences_filenames[block_id]) /
          sizeof(text_offset_type);
        std::uint64_t items_processed = 0;

        while (items_processed < subseq_size) {
          std::uint64_t filled =
            std::min(buffer_size, subseq_size - items_processed);
          sa_subseq_reader->read(buf, filled);

#ifdef _OPENMP
          #pragma omp parallel for
          for (std::uint64_t j = 0; j < filled; ++j) {
            std::uint64_t sa_val = buf[j];
            std::uint64_t lcp_val = plcp_block[sa_val - block_beg];
            outbuf[j] = lcp_val;
          }
#else
          for (std::uint64_t j = 0; j < filled; ++j) {
            std::uint64_t sa_val = buf[j];
            std::uint64_t lcp_val = plcp_block[sa_val - block_beg];
            outbuf[j] = lcp_val;
          }
#endif

          lcp_subseq_writer->write(outbuf, filled);
          items_processed += filled;
        }

        // Stop I/O threads.
        sa_subseq_reader->stop_reading();

        // Update I/O volume.
        io_volume +=
          sa_subseq_reader->bytes_read() +
          lcp_subseq_writer->bytes_written();

        // Clean up.
        delete lcp_subseq_writer;
        delete sa_subseq_reader;
      }

      utils::file_delete(sa_subsequences_filenames[block_id]);
    }

    // Stop I/O threads.
    plcp_bitvector_reader->stop_reading();

    // Update I/O volume.
    io_volume +=
      plcp_bitvector_reader->bytes_read();
    total_io_volume += io_volume;

    // Clean up.
    utils::deallocate(outbuf);
    utils::deallocate(buf);
    utils::deallocate(plcp_block);
    delete plcp_bitvector_reader;
    if (keep_plcp == false)
      utils::file_delete(B_filename);

    // Print summary.
    long double compute_lcp_subseq_time =
      utils::wclock() - compute_lcp_subseq_start;
    fprintf(stderr, "time = %.2Lfs, I/O = %.2LfMiB/s, "
        "total I/O vol = %.2Lfbytes/input symbol\n",
        compute_lcp_subseq_time,
        ((1.L * io_volume) / (1L << 20)) / compute_lcp_subseq_time,
        (1.L * total_io_volume) / text_length);
  }

  // Merge LCP subsequences.
  {
    fprintf(stderr, "  Merge LCP subsequences: ");
    long double merge_lcp_subseq_start = utils::wclock();
    std::uint64_t io_volume = 0;

    // Initialize the reader of LCP subsequences.
    std::uint64_t total_buffers_ram = ram_use;
    std::uint64_t buffer_size = total_buffers_ram / n_blocks;
    typedef async_multi_stream_reader_multipart<text_offset_type>
      lcp_subseq_multireader_type;
    lcp_subseq_multireader_type *lcp_subseq_multireader =
      new lcp_subseq_multireader_type(n_blocks, buffer_size);

    for (std::uint64_t block_id = 0; block_id < n_blocks; ++block_id)
      lcp_subseq_multireader->add_file(lcp_subsequences_filenames[block_id]);

    // Initialize the writer of the final LCP array.
    typedef async_stream_writer<text_offset_type> lcp_writer_type;
    lcp_writer_type *lcp_writer = new lcp_writer_type(output_filename);

    // Initialize the reader of SA.
    typedef async_stream_reader<text_offset_type> sa_reader_type;
    sa_reader_type *sa_reader = new sa_reader_type(sa_filename);

    // Compute final LCP.
    for (std::uint64_t j = 0; j < text_length; ++j) {
      std::uint64_t sa_j = sa_reader->read();
      std::uint64_t block_id = sa_j / max_block_size;
      std::uint64_t lcp_j =
        lcp_subseq_multireader->read_from_ith_file(block_id);

      local_max_lcp = std::max(local_max_lcp, lcp_j);
      local_lcp_sum += lcp_j;
      lcp_writer->write(lcp_j);
    }

    // Stop I/O threads.
    sa_reader->stop_reading();
    lcp_subseq_multireader->stop_reading();

    // Update I/O volume.
    io_volume +=
      sa_reader->bytes_read() +
      lcp_subseq_multireader->bytes_read() +
      lcp_writer->bytes_written();
    total_io_volume += io_volume;

    // Clean up.
    delete sa_reader;
    delete lcp_writer;
    delete lcp_subseq_multireader;

    // Print summary.
    long double merge_lcp_subseq_time =
      utils::wclock() - merge_lcp_subseq_start;
    fprintf(stderr, "time = %.2Lfs, I/O = %.2LfMiB/s, "
        "total I/O vol = %.2Lfbytes/input symbol\n",
        merge_lcp_subseq_time,
        ((1.L * io_volume) / (1L << 20)) / merge_lcp_subseq_time,
        (1.L * total_io_volume) / text_length);
  }

  // Clean up.
  delete[] lcp_subsequences_filenames;
  delete[] sa_subsequences_filenames;

  // Print summary.
  long double convert_plcp_to_lcp_time =
    utils::wclock() - convert_plcp_to_lcp_start;
  fprintf(stderr, "  Summary: time = %.2Lfs, "
      "total I/O vol = %.2Lfbytes/input symbol\n",
      convert_plcp_to_lcp_time,
      (1.L * total_io_volume) / text_length);

  // Update reference variables.
  global_io_volume += total_io_volume;
  max_lcp = local_max_lcp;
  lcp_sum = local_lcp_sum;
}

template<typename text_offset_type>
void compute_lcp_from_plcp(
    std::uint64_t text_length,
    std::uint64_t ram_use,
    std::uint64_t *B,
    std::string sa_filename,
    std::string output_filename,
    std::uint64_t &total_io_volume,
    std::uint64_t &max_lcp,
    std::uint64_t &lcp_sum) {

  // Write B to disk.
  std::string B_filename = output_filename +
    ".plcp." + utils::random_string_hash();
  {

    // Start the timer.
    fprintf(stderr, "Write PLCP bitvector to disk: ");
    long double write_plcp_start = utils::wclock();
    std::uint64_t io_volume = 0;

    // Write the data.
    std::uint64_t length_of_B_in_words = (2UL * text_length + 63) / 64;
    utils::write_to_file(B, length_of_B_in_words, B_filename);

    // Update I/O volume.
    io_volume += length_of_B_in_words * sizeof(std::uint64_t);
    total_io_volume += io_volume;
    long double write_plcp_time = utils::wclock() - write_plcp_start;
    fprintf(stderr, "time = %.2Lfs, I/O = %.2LfMiB/s, "
        "I/O vol = %.2Lfbytes/input symbol\n\n",
        write_plcp_time,
        ((1.L * io_volume) / (1L << 20)) / write_plcp_time,
        (1.L * io_volume) / text_length);
  }
  utils::deallocate(B);

  // Convert PLCP to LCP using EM method.
  compute_lcp_from_plcp<text_offset_type>(text_length,
      ram_use, sa_filename, output_filename, B_filename,
      total_io_volume, max_lcp, lcp_sum);
}

}  // namespace normal_mode

namespace inplace_mode {

template<typename text_offset_type>
void compute_lcp_from_plcp(
    std::uint64_t text_length,
    std::uint64_t ram_use,
    std::string sa_filename,
    std::string output_filename,
    std::string B_filename,
    std::uint64_t &global_io_volume,
    std::uint64_t &max_lcp,
    std::uint64_t &lcp_sum,
    bool keep_plcp = false) {

  // Print initial message, start the timer,
  // and initialize the I/O volume.
  fprintf(stderr, "Convert PLCP to LCP:\n");
  long convert_plcp_to_lcp_start = utils::wclock();
  std::uint64_t io_volume = 0;

  // Initialize local statistic.
  std::uint64_t local_lcp_sum = 0;
  std::uint64_t local_max_lcp = 0;

  // Compute basic parameters.
  std::uint64_t sparse_plcp_ram = ram_use / 2;
  std::uint64_t sparse_plcp_size =
    sparse_plcp_ram / sizeof(text_offset_type);
  sparse_plcp_size = std::max(sparse_plcp_size, (std::uint64_t)1);
  sparse_plcp_size = std::min(sparse_plcp_size, text_length);
  std::uint64_t plcp_sampling_rate =
    (text_length + sparse_plcp_size - 1) / sparse_plcp_size;
  sparse_plcp_size =
    (text_length + plcp_sampling_rate - 1) / plcp_sampling_rate;
  sparse_plcp_ram = sparse_plcp_size * sizeof(text_offset_type);
  std::uint64_t ram_left = 0;
  if (sparse_plcp_ram > ram_use)
    ram_left = 0;
  else ram_left = ram_use - sparse_plcp_ram;

  // Print PLCP sampling rate.
  fprintf(stderr, "  Sparse PLCP sampling rate = %lu\n",
      plcp_sampling_rate);

  // Allocate sparse PLCP array.
  text_offset_type *sparse_plcp =
    utils::allocate_array<text_offset_type>(sparse_plcp_size);

  // Compute sparse PLCP array.
  {

    // Print initial message, start the
    // timer, and initialize I/O volume.
    fprintf(stderr, "  Compute sparse PLCP: ");
    long double compute_sparse_plcp_start = utils::wclock();
    std::uint64_t io_vol = 0;

    // Create the reader of PLCP bitvector.
    typedef async_stream_reader<std::uint64_t> plcp_reader_type;
    plcp_reader_type *plcp_reader = new plcp_reader_type(B_filename);

    // Stream PLCP bitvector.
    std::uint64_t i_mod = 0;
    std::uint64_t i_div = 0;
    std::uint64_t i = 0;
    std::uint64_t bits_processed = 0;
    std::uint64_t plcp_bitvector_length = 2 * text_length;
    while (bits_processed < plcp_bitvector_length) {
      std::uint64_t bitbuf = plcp_reader->read();
      std::uint64_t filled = std::min((std::uint64_t)64,
          plcp_bitvector_length - bits_processed);

      // Process buffer.
      for (std::uint64_t j = 0; j < filled; ++j) {
        if (bitbuf & (((std::uint64_t)1) << j)) {
          if (i_mod == 0) {
            std::uint64_t lcp = (bits_processed + j) - (2 * i);
            sparse_plcp[i_div] = lcp;
          }

          ++i;
          ++i_mod;
          if (i_mod == plcp_sampling_rate) {
            i_mod = 0;
            ++i_div;
          }
        }
      }

      // Update the number of processed bits.
      bits_processed += filled;
    }

    // Stop I/O threads.
    plcp_reader->stop_reading();

    // Update I/O volume.
    io_vol +=
      plcp_reader->bytes_read();
    io_volume += io_vol;

    // Clean up.
    delete plcp_reader;

    // Print summary.
    long double compute_sparse_plcp_time =
      utils::wclock() - compute_sparse_plcp_start;
    fprintf(stderr, "time = %.2Lfs, I/O = %.2LfMiB/s, "
        "total I/O vol = %.2Lfbytes/input symbol\n",
        compute_sparse_plcp_time,
        ((1.L * io_vol) / (1L << 20)) / compute_sparse_plcp_time,
        (1.L * io_volume) / text_length);
  }

  // Initialize part sizes for lex-partitioning.
  static const std::uint64_t n_parts = 2;
  std::uint64_t *part_sizes =
    utils::allocate_array<std::uint64_t>(n_parts);
  part_sizes[0] = text_length * 0.66L;
  part_sizes[1] = text_length - part_sizes[0];

  // Compute the PLCP block size.
  std::uint64_t plcp_block_size =
    std::max((std::uint64_t)1, ram_left / sizeof(text_offset_type));
  std::uint64_t n_plcp_blocks =
    (text_length + plcp_block_size - 1) / plcp_block_size;

  fprintf(stderr, "  PLCP block size = %lu\n", plcp_block_size);
  fprintf(stderr, "  Number of blocks = %lu\n", n_plcp_blocks);

  // Set the filenames of files storing SA and LCP subsequences.
  std::string *lcp_delta_subsequences_filenames =
    new std::string[n_plcp_blocks];
  std::string *sa_subsequences_filenames =
    new std::string[n_plcp_blocks];
  for (std::uint64_t block_id = 0; block_id < n_plcp_blocks; ++block_id) {
    sa_subsequences_filenames[block_id] =
      output_filename + ".sa_subseq." +
      utils::intToStr(block_id) + "." + utils::random_string_hash();
    lcp_delta_subsequences_filenames[block_id] =
      output_filename + ".lcp_delta_sebseq." +
      utils::intToStr(block_id) + "." + utils::random_string_hash();
  }

  // Lex-partitioning follows.
  for (std::uint64_t part_id = 0; part_id < n_parts; ++part_id) {
    std::uint64_t part_beg = 0;
    for (std::uint64_t i = 0; i < part_id; ++i)
      part_beg += part_sizes[i];
    std::uint64_t part_size = part_sizes[part_id];

    // Print info.
    fprintf(stderr, "  Process part %lu/%lu:\n", part_id + 1, n_parts);
    fprintf(stderr, "    Part size = %lu (%.2Lf%% of all positions)\n",
        part_size, (100.L * part_size) / text_length);

    // Compute SA subsequences.
    {

      // Print initial message, start the
      // timer and initialize I/O volume.
      fprintf(stderr, "    Compute SA subsequences: ");
      long double compute_sa_subseq_start = utils::wclock();
      std::uint64_t io_vol = 0;

      // Create the suffix array reader.
      typedef async_stream_reader<text_offset_type> sa_reader_type;
      sa_reader_type *sa_reader =
        new sa_reader_type(sa_filename, part_beg);

      // Initialize multifile writer of SA subsequences.
      static const std::uint64_t n_free_buffers = 4;
      std::uint64_t total_buffers_ram = ram_left;
      std::uint64_t buffer_size = std::min(((std::uint64_t)16) << 20,
          total_buffers_ram / (n_plcp_blocks + n_free_buffers));
      typedef async_multi_stream_writer<text_offset_type>
        sa_multiwriter_type;
      sa_multiwriter_type *sa_multiwriter = new sa_multiwriter_type(
          n_plcp_blocks, buffer_size, n_free_buffers);

      // Add files to multiwriter.
      for (std::uint64_t i = 0; i < n_plcp_blocks; ++i)
        sa_multiwriter->add_file(sa_subsequences_filenames[i]);

      // Read SA / write SA subsequences.
      for (std::uint64_t i = 0; i < part_size; ++i) {
        std::uint64_t sa_val = sa_reader->read();
        std::uint64_t block_id = sa_val / plcp_block_size;
        sa_multiwriter->write_to_ith_file(block_id, sa_val);
      }

      // Stop I/O threads.
      sa_reader->stop_reading();

      // Update I/O volume.
      io_vol +=
        sa_reader->bytes_read() +
        sa_multiwriter->bytes_written();
      io_volume += io_vol;

      // Clean up.
      delete sa_multiwriter;
      delete sa_reader;

      // Print summary.
      long double compute_sa_subseq_time =
        utils::wclock() - compute_sa_subseq_start;
      fprintf(stderr, "time = %.2Lfs, I/O = %.2LfMiB/s, "
          "total I/O vol = %.2Lfbytes/input symbol\n",
          compute_sa_subseq_time,
          ((1.L * io_vol) / (1L << 20)) / compute_sa_subseq_time,
          (1.L * io_volume) / text_length);
    }

    // Compute subsequences containing LCP delta values.
    {

      // Print initial message, start the
      // timer and initialize I/O volme.
      fprintf(stderr, "    Compute LCP subsequences: ");
      long double compute_lcp_subseq_start = utils::wclock();
      std::uint64_t io_vol = 0;

      // Initialize reading of PLCP bitvector.
      typedef async_stream_reader<std::uint64_t>
        plcp_bitvector_reader_type;
      plcp_bitvector_reader_type *plcp_bitvector_reader =
        new plcp_bitvector_reader_type(B_filename);

      std::uint64_t bitbuf = plcp_bitvector_reader->read();
      std::uint64_t bitpos = 0;
      std::uint64_t cur_plcp = 1;

      // Allocate the array holding the block of PLCP.
      text_offset_type *plcp_block =
        utils::allocate_array<text_offset_type>(plcp_block_size);

      // Allocate buffers.
      static const std::uint64_t buffer_size = (1UL << 18);
      text_offset_type *sa_buf =
        utils::allocate_array<text_offset_type>(buffer_size);
      std::uint64_t *addr_buf =
        utils::allocate_array<std::uint64_t>(buffer_size);
      std::uint64_t *sparse_plcp_buf =
        utils::allocate_array<std::uint64_t>(buffer_size);
      std::uint64_t *plcp_buf =
        utils::allocate_array<std::uint64_t>(buffer_size);
      std::uint64_t *delta_buf =
        utils::allocate_array<std::uint64_t>(buffer_size);
      std::uint8_t *vbyte_slab =
        utils::allocate_array<std::uint8_t>(buffer_size * 9);

      // Process blocks left to right.
      for (std::uint64_t block_id = 0; block_id < n_plcp_blocks; ++block_id) {
        std::uint64_t block_beg = block_id * plcp_block_size;
        std::uint64_t block_end =
          std::min(block_beg + plcp_block_size, text_length);
        std::uint64_t block_size = block_end - block_beg;

        // Read a block of PLCP into RAM.
        for (std::uint64_t j = 0; j < block_size; ++j) {

          // Increment cur_plcp for every 0 in the bitvector.
          while ((bitbuf & (1UL << bitpos)) == 0) {
            ++cur_plcp;
            ++bitpos;
            if (bitpos == 64) {
              bitbuf = plcp_bitvector_reader->read();
              bitpos = 0;
            }
          }

          // We decrement last because cur_plcp is unsigned.
          --cur_plcp;
          plcp_block[j] = cur_plcp;

          // Skip the 1-bit in the bitvector.
          ++bitpos;
          if (bitpos == 64) {
            if (plcp_bitvector_reader->empty() == false)
              bitbuf = plcp_bitvector_reader->read();
            bitpos = 0;
          }
        }

        // Compute LCP subsequence and write to file.
        {

          // Initialize SA subsequence reader.
          typedef async_stream_reader<text_offset_type>
            sa_subseq_reader_type;
          sa_subseq_reader_type *sa_subseq_reader =
            new sa_subseq_reader_type(sa_subsequences_filenames[block_id]);

          // Initialize the LCP subsequence writer.
          typedef async_stream_writer<std::uint8_t> lcp_delta_write_type;
          lcp_delta_write_type *lcp_delta_writer = NULL;
          std::string filename = lcp_delta_subsequences_filenames[block_id];
          if (part_id == 0)
            lcp_delta_writer = new lcp_delta_write_type(filename, "w");
          else lcp_delta_writer = new lcp_delta_write_type(filename, "a");

          // Compute LCP subsequence.
          std::uint64_t subseq_size =
            utils::file_size(sa_subsequences_filenames[block_id]) /
            sizeof(text_offset_type);
          std::uint64_t items_processed = 0;
          while (items_processed < subseq_size) {
            std::uint64_t filled = std::min(buffer_size,
                subseq_size - items_processed);
            sa_subseq_reader->read(sa_buf, filled);

#ifdef _OPENMP
            #pragma omp parallel for
            for (std::uint64_t j = 0; j < filled; ++j) {
              std::uint64_t sa_val = sa_buf[j];
              std::uint64_t lcp_val = plcp_block[sa_val - block_beg];
              plcp_buf[j] = lcp_val;
            }

            #pragma omp parallel for
            for (std::uint64_t j = 0; j < filled; ++j) {
              std::uint64_t sa_val = sa_buf[j];
              std::uint64_t addr = sa_val / plcp_sampling_rate;
              addr_buf[j] = addr;
            }

            #pragma omp parallel for
            for (std::uint64_t j = 0; j < filled; ++j) {
              std::uint64_t addr = addr_buf[j];
              sparse_plcp_buf[j] = sparse_plcp[addr];
            }

            #pragma omp parallel for
            for (std::uint64_t j = 0; j < filled; ++j) {
              std::uint64_t sa_val = sa_buf[j];
              std::uint64_t plcp_val = plcp_buf[j];
              std::uint64_t sparse_plcp_addr = addr_buf[j];
              std::uint64_t sparse_plcp_idx =
                sparse_plcp_addr * plcp_sampling_rate;
              std::uint64_t sparse_plcp_val = sparse_plcp_buf[j];
              std::uint64_t plcp_lower_bound =
                std::max((std::int64_t)0,
                    (std::int64_t)sparse_plcp_val -
                    (std::int64_t)(sa_val - sparse_plcp_idx));
              delta_buf[j] = plcp_val - plcp_lower_bound;
            }
#else
            for (std::uint64_t j = 0; j < filled; ++j) {
              std::uint64_t sa_val = sa_buf[j];
              std::uint64_t lcp_val = plcp_block[sa_val - block_beg];
              plcp_buf[j] = lcp_val;
            }

            for (std::uint64_t j = 0; j < filled; ++j) {
              std::uint64_t sa_val = sa_buf[j];
              std::uint64_t addr = sa_val / plcp_sampling_rate;
              addr_buf[j] = addr;
            }

            for (std::uint64_t j = 0; j < filled; ++j) {
              std::uint64_t addr = addr_buf[j];
              sparse_plcp_buf[j] = sparse_plcp[addr];
            }

            for (std::uint64_t j = 0; j < filled; ++j) {
              std::uint64_t sa_val = sa_buf[j];
              std::uint64_t plcp_val = plcp_buf[j];
              std::uint64_t sparse_plcp_addr = addr_buf[j];
              std::uint64_t sparse_plcp_idx =
                sparse_plcp_addr * plcp_sampling_rate;
              std::uint64_t sparse_plcp_val = sparse_plcp_buf[j];
              std::uint64_t plcp_lower_bound =
                std::max((std::int64_t)0,
                    (std::int64_t)sparse_plcp_val -
                    (std::int64_t)(sa_val - sparse_plcp_idx));
              delta_buf[j] = plcp_val - plcp_lower_bound;
            }
#endif

            // Convert LCP deltas to vbyte encoding.
            std::uint64_t vbyte_slab_length =
              convert_to_vbyte_slab(delta_buf, filled, vbyte_slab);

            // Write LCP deltas to disk.
            lcp_delta_writer->write(vbyte_slab, vbyte_slab_length);

            // Update the number of processed items.
            items_processed += filled;
          }

          // Stop I/O threads.
          sa_subseq_reader->stop_reading();

          // Update I/O volume.
          io_vol +=
            sa_subseq_reader->bytes_read() +
            lcp_delta_writer->bytes_written();

          // Clean up.
          delete lcp_delta_writer;
          delete sa_subseq_reader;
        }

        // Delete the file with SA subsequence.
        // In later versions this will happen later.
        utils::file_delete(sa_subsequences_filenames[block_id]);
      }

      // Stop I/O threads.
      plcp_bitvector_reader->stop_reading();

      // Update I/O volume.
      io_vol +=
        plcp_bitvector_reader->bytes_read();
      io_volume += io_vol;

      // Clean up.
      utils::deallocate(vbyte_slab);
      utils::deallocate(delta_buf);
      utils::deallocate(plcp_buf);
      utils::deallocate(sparse_plcp_buf);
      utils::deallocate(addr_buf);
      utils::deallocate(sa_buf);
      utils::deallocate(plcp_block);
      delete plcp_bitvector_reader;

      // Print summary.
      long double compute_lcp_subseq_time =
        utils::wclock() - compute_lcp_subseq_start;
      fprintf(stderr, "time = %.2Lfs, I/O = %.2LfMiB/s, "
          "total I/O vol = %.2Lfbytes/input symbol\n",
          compute_lcp_subseq_time,
          ((1.L * io_vol) / (1L << 20)) / compute_lcp_subseq_time,
          (1.L * io_volume) / text_length);
    }
  }

  // Clean up.
  delete[] sa_subsequences_filenames;
  if (keep_plcp == false)
    utils::file_delete(B_filename);

  // Initialize the file with all LCP deltas in
  // lex order (stored in many physical files).
  std::string lcp_deltas_filename =
    output_filename + "." +
    utils::random_string_hash();

  // Rewrite files with LCP deltas into few physical
  // files that can be suitably deleted when reading.
  {

    // Print initial message, start the
    // timer and initialize I/O volume.
    fprintf(stderr, "  Preprocess LCP delta values: ");
    long double preprocess_lcp_delta_start = utils::wclock();
    std::uint64_t io_vol = 0;

    // Create multireader for LCP delta values.
    std::uint64_t total_buffers_ram = ram_left;
    std::uint64_t multi_reader_buffer_size =
      std::max((std::uint64_t)1,
          total_buffers_ram / n_plcp_blocks);
    typedef async_multi_stream_vbyte_reader
      lcp_delta_multireader_type;
    lcp_delta_multireader_type *lcp_delta_multireader =
      new lcp_delta_multireader_type(n_plcp_blocks,
          multi_reader_buffer_size);
    for (std::uint64_t i = 0; i < n_plcp_blocks; ++i)
      lcp_delta_multireader->add_file(
          lcp_delta_subsequences_filenames[i]);

    // Initialize SA reader.
    typedef async_stream_reader<text_offset_type> sa_reader_type;
    sa_reader_type *sa_reader = new sa_reader_type(sa_filename);

    // Initialize writer of preprocessed LCP delta values.
    typedef async_stream_writer_multipart_2<std::uint8_t>
      lcp_delta_writer_type;
    lcp_delta_writer_type *lcp_delta_writer =
      new lcp_delta_writer_type(lcp_deltas_filename);

    // Allocate buffers.
    // text_offset_type *sa_buf =
    //   utils::allocate_array<text_offset_type>(buffer_size);
    static const std::uint64_t buffer_size = (1L << 19);
    std::uint64_t *delta_buf =
      utils::allocate_array<std::uint64_t>(buffer_size);
    std::uint8_t *vbyte_slab =
      utils::allocate_array<std::uint8_t>(buffer_size * 9);

    // Compute the total size of files with LCP deltas.
    std::uint64_t total_lcp_delta_file_sizes = 0;
    for (std::uint64_t i = 0; i < n_plcp_blocks; ++i)
      total_lcp_delta_file_sizes +=
        utils::file_size(lcp_delta_subsequences_filenames[i]);

    // Initialize disk space counter.
    static const std::uint64_t max_allowed_extra_disk_space = (1UL << 20);
    std::uint64_t used_disk_space = total_lcp_delta_file_sizes;
    std::uint64_t cur_delta_file_size = 0;
    std::uint64_t lcp_deltas_filled = 0;

    for (std::uint64_t i = 0; i < text_length; ++i) {
      std::uint64_t sa_val = sa_reader->read();
      std::uint64_t plcp_block_id = sa_val / plcp_block_size;
      delta_buf[lcp_deltas_filled++] =
        lcp_delta_multireader->read_from_ith_file(plcp_block_id);
      used_disk_space += sizeof(text_offset_type);

      // XXX 32-bit and 48-bit integers?
      if (lcp_deltas_filled == buffer_size ||
          used_disk_space > text_length * sizeof(text_offset_type) +
          max_allowed_extra_disk_space) {

        // Convert LCP deltas to vbyte.
        std::uint64_t vbyte_slab_length =
          convert_to_vbyte_slab(delta_buf,
              lcp_deltas_filled, vbyte_slab);
        cur_delta_file_size += vbyte_slab_length;
        lcp_deltas_filled = 0;

        // Write LCP deltas to disk.
        lcp_delta_writer->write(vbyte_slab, vbyte_slab_length);

        // Check if need to end the current file.
        if (used_disk_space > text_length * sizeof(text_offset_type) +
            max_allowed_extra_disk_space) {
          lcp_delta_writer->end_current_file();

          // Simulate the space assuming we
          // now delete the current delta file.
          used_disk_space -= cur_delta_file_size;
          cur_delta_file_size = 0;
        }
      }
    }

    if (lcp_deltas_filled > 0) {

      // Convert lcp deltas to vbyte.
      std::uint64_t vbyte_slab_length =
        convert_to_vbyte_slab(delta_buf, lcp_deltas_filled, vbyte_slab);

      // Write lcp deltas to disk.
      lcp_delta_writer->write(vbyte_slab, vbyte_slab_length);
    }

    // Stop I/O threads.
    lcp_delta_multireader->stop_reading();
    sa_reader->stop_reading();

    // Update I/O volume.
    io_vol +=
      lcp_delta_multireader->bytes_read() +
      sa_reader->bytes_read() +
      lcp_delta_writer->bytes_written();
    io_volume += io_vol;

    // Clean up.
    utils::deallocate(vbyte_slab);
    utils::deallocate(delta_buf);
    delete lcp_delta_writer;
    delete sa_reader;
    delete lcp_delta_multireader;

    // Print summary.
    long double preprocess_lcp_delta_time =
      utils::wclock() - preprocess_lcp_delta_start;
    fprintf(stderr, "time = %.2Lfs, I/O = %.2LfMiB/s, "
        "total I/O vol = %.2Lfbytes/input symbol\n",
        preprocess_lcp_delta_time,
        ((1.L * io_vol) / (1L << 20)) / preprocess_lcp_delta_time,
        (1.L * io_volume) / text_length);
  }

  // Clean up.
  for (std::uint64_t i = 0; i < n_plcp_blocks; ++i) {
    std::string filename =
      lcp_delta_subsequences_filenames[i];
    if (utils::file_exists(filename))
      utils::file_delete(filename);
  }

  delete[] lcp_delta_subsequences_filenames;

  // Rewrite (in-place) LCP deltas files into final LCP
  // array with the help of the sparse PLCP array.
  {

    // Print initial message and start the timer.
    fprintf(stderr, "  Compute final LCP array: ");
    long double compute_final_lcp_start = utils::wclock();
    std::uint64_t io_vol = 0;

    // Initialize SA reader.
    typedef async_stream_reader<text_offset_type>
      sa_reader_type;
    sa_reader_type *sa_reader =
      new sa_reader_type(sa_filename);

    // Initialize the reader of LCP delta values.
    typedef async_stream_vbyte_reader_multipart
      lcp_delta_reader_type;
    lcp_delta_reader_type *lcp_delta_reader =
      new lcp_delta_reader_type(lcp_deltas_filename);

    // Initialize writer of the final LCP array.
    typedef async_stream_writer<text_offset_type>
      lcp_writer_type;
    lcp_writer_type *lcp_writer =
      new lcp_writer_type(output_filename);

    // Allocate buffers.
    static const std::uint64_t buffer_size = (1UL << 18);
    text_offset_type *sa_buf =
      utils::allocate_array<text_offset_type>(buffer_size);
    std::uint64_t *addr_buf =
      utils::allocate_array<std::uint64_t>(buffer_size);
    std::uint64_t *sparse_plcp_buf =
      utils::allocate_array<std::uint64_t>(buffer_size);
    std::uint64_t *lcp_buf =
      utils::allocate_array<std::uint64_t>(buffer_size);

    // Compute final LCP array.
    std::uint64_t items_processed = 0;
    while (items_processed < text_length) {
      std::uint64_t filled = std::min(buffer_size,
          text_length - items_processed);
      sa_reader->read(sa_buf, filled);

#ifdef _OPENMP
      #pragma omp parallel for
      for (std::uint64_t j = 0; j < filled; ++j) {
        std::uint64_t sa_val = sa_buf[j];
        std::uint64_t addr = sa_val / plcp_sampling_rate;
        addr_buf[j] = addr;
      }

      #pragma omp parallel for
      for (std::uint64_t j = 0; j < filled; ++j) {
        std::uint64_t addr = addr_buf[j];
        sparse_plcp_buf[j] = sparse_plcp[addr];
      }

      #pragma omp parallel for
      for (std::uint64_t j = 0; j < filled; ++j) {
        std::uint64_t sa_val = sa_buf[j];
        std::uint64_t sparse_plcp_addr = addr_buf[j];
        std::uint64_t sparse_plcp_idx =
          sparse_plcp_addr * plcp_sampling_rate;
        std::uint64_t sparse_plcp_val = sparse_plcp_buf[j];
        std::uint64_t plcp_lower_bound =
          std::max((std::int64_t)0,
              (std::int64_t)sparse_plcp_val -
              (std::int64_t)(sa_val - sparse_plcp_idx));
        lcp_buf[j] = plcp_lower_bound;
      }
#else
      for (std::uint64_t j = 0; j < filled; ++j) {
        std::uint64_t sa_val = sa_buf[j];
        std::uint64_t addr = sa_val / plcp_sampling_rate;
        addr_buf[j] = addr;
      }

      for (std::uint64_t j = 0; j < filled; ++j) {
        std::uint64_t addr = addr_buf[j];
        sparse_plcp_buf[j] = sparse_plcp[addr];
      }

      for (std::uint64_t j = 0; j < filled; ++j) {
        std::uint64_t sa_val = sa_buf[j];
        std::uint64_t sparse_plcp_addr = addr_buf[j];
        std::uint64_t sparse_plcp_idx =
          sparse_plcp_addr * plcp_sampling_rate;
        std::uint64_t sparse_plcp_val = sparse_plcp_buf[j];
        std::uint64_t plcp_lower_bound =
          std::max((std::int64_t)0,
              (std::int64_t)sparse_plcp_val -
              (std::int64_t)(sa_val - sparse_plcp_idx));
        lcp_buf[j] = plcp_lower_bound;
      }
#endif

      // Write the final LCP values to file.
      for (std::uint64_t i = 0; i < filled; ++i) {
        std::uint64_t lcp_lower_bound = lcp_buf[i];
        std::uint64_t lcp_delta = lcp_delta_reader->read();
        std::uint64_t lcp_value = lcp_lower_bound + lcp_delta;
        lcp_buf[i] = lcp_value;

        // Update statistics.
        local_max_lcp = std::max(local_max_lcp, lcp_value);
        local_lcp_sum += lcp_value;

        // Write LCP value to file.
        lcp_writer->write(lcp_value);
      }

      // Update the number of processed items.
      items_processed += filled;
    }

    // Stop I/O threads.
    sa_reader->stop_reading();
    lcp_delta_reader->stop_reading();

    // Update I/O volume.
    io_vol +=
      sa_reader->bytes_read() +
      lcp_delta_reader->bytes_read() +
      lcp_writer->bytes_written();
    io_volume += io_vol;

    // Clean up.
    utils::deallocate(lcp_buf);
    utils::deallocate(sparse_plcp_buf);
    utils::deallocate(addr_buf);
    utils::deallocate(sa_buf);
    delete lcp_writer;
    delete lcp_delta_reader;
    delete sa_reader;

    // Print summary.
    long double compute_final_lcp_time =
      utils::wclock() - compute_final_lcp_start;
    fprintf(stderr, "time = %.2Lfs, I/O = %.2LfMiB/s, "
        "total I/O vol = %.2Lfbytes/input symbol\n",
        compute_final_lcp_time,
        ((1.L * io_vol) / (1L << 20)) / compute_final_lcp_time,
        (1.L * io_volume) / text_length);
  }

  // Clean up.
  utils::deallocate(part_sizes);
  utils::deallocate(sparse_plcp);

  // Print summary.
  long double convert_plcp_to_lcp_time =
    utils::wclock() - convert_plcp_to_lcp_start;
  fprintf(stderr, "  Summary: time = %.2Lfs, "
      "total I/O vol = %.2Lfbytes/input symbol\n",
      convert_plcp_to_lcp_time,
      (1.L * io_volume) / text_length);

  // Update reference variables.
  max_lcp = local_max_lcp;
  lcp_sum = local_lcp_sum;
  global_io_volume += io_volume;
}

template<typename text_offset_type>
void compute_lcp_from_plcp(
    std::uint64_t text_length,
    std::uint64_t ram_use,
    std::uint64_t *B,
    std::string sa_filename,
    std::string output_filename,
    std::uint64_t &total_io_volume,
    std::uint64_t &max_lcp,
    std::uint64_t &lcp_sum) {

  // Write B to disk.
  std::string B_filename = output_filename +
    ".plcp." + utils::random_string_hash();
  {

    // Start the timer.
    fprintf(stderr, "Write PLCP bitvector to disk: ");
    long double write_plcp_start = utils::wclock();
    std::uint64_t io_volume = 0;

    // Write the data.
    std::uint64_t length_of_B_in_words = (2UL * text_length + 63) / 64;
    utils::write_to_file(B, length_of_B_in_words, B_filename);

    // Update I/O volume.
    io_volume += length_of_B_in_words * sizeof(std::uint64_t);
    total_io_volume += io_volume;
    long double write_plcp_time = utils::wclock() - write_plcp_start;
    fprintf(stderr, "time = %.2Lfs, I/O = %.2LfMiB/s, "
        "I/O vol = %.2Lfbytes/input symbol\n\n",
        write_plcp_time,
        ((1.L * io_volume) / (1L << 20)) / write_plcp_time,
        (1.L * io_volume) / text_length);
  }
  utils::deallocate(B);

  // Convert PLCP to LCP using EM method.
  compute_lcp_from_plcp<text_offset_type>(text_length,
      ram_use, sa_filename, output_filename, B_filename,
      total_io_volume, max_lcp, lcp_sum);
}

}  // namespace inplace_mode

template<typename text_offset_type>
void compute_lcp_from_plcp(
    std::string input_filename,
    std::string sa_filename,
    std::string output_filename,
    std::uint64_t ram_use,
    bool inplace_mode) {

  // Empty page cache and initialize
  // (pseudo)random number generator.
  srand(time(0) + getpid());
  utils::initialize_stats();
  utils::empty_page_cache(input_filename);
  utils::empty_page_cache(sa_filename);

  // Start the timer and initialize I/O volume.
  long double start = utils::wclock();
  std::uint64_t io_volume = 0;

  // Compute basic parameters.
  std::uint64_t sa_file_size = utils::file_size(sa_filename);
  std::uint64_t text_length = sa_file_size / sizeof(text_offset_type);

  // Check if the text is non-empty.
  if (text_length == 0) {
    fprintf(stderr, "Error: the input file is empty!\n");
    std::exit(EXIT_FAILURE);
  }

  // Sanity check.
  if (sa_file_size % sizeof(text_offset_type)) {
    fprintf(stderr, "\nError: size of SA file is not a "
        "multiple of sizeof(text_offset_type)!\n");
    std::exit(EXIT_FAILURE);
  }

  // Check if all types are sufficiently large.
  // XXX are these assumption correct. The
  // calculations are correct I'm pretty sure.
  {

    // text_offset_type must be able to hold values in range [0..text_length).
    std::uint64_t text_offset_type_max =
      std::numeric_limits<text_offset_type>::max();
    if (text_offset_type_max < text_length - 1) {
      fprintf(stderr, "\nError: text_offset_type is too small:\n"
          "\tnumeric_limits<text_offset_type>::max() = %lu\n"
          "\ttext length = %lu\n", text_offset_type_max, text_length);
      std::exit(EXIT_FAILURE);
    }
  }

  // Turn paths absolute.
  input_filename = utils::absolute_path(input_filename);
  sa_filename = utils::absolute_path(sa_filename);
  output_filename = utils::absolute_path(output_filename);

  // Print summary of basic parameters.
  fprintf(stderr, "Running EM-SuccinctIrreducible v0.2.0\n");
  fprintf(stderr, "Mode = convert PLCP bitvector into LCP array\n");
  fprintf(stderr, "Timestamp = %s", utils::get_timestamp().c_str());
  fprintf(stderr, "PLCP filename = %s\n", input_filename.c_str());
  fprintf(stderr, "SA filename = %s\n", sa_filename.c_str());
  fprintf(stderr, "Output (LCP) filename = %s\n", output_filename.c_str());
  fprintf(stderr, "Text length = %lu\n", text_length);
  fprintf(stderr, "RAM use = %lu bytes (%.2LfMiB)\n",
      ram_use, ram_use / (1024.L * 1024));
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

  std::uint64_t lcp_sum = 0;
  std::uint64_t max_lcp = 0;

  // Convert the PCLP array (bitvector representation) to LCP array.
  if (inplace_mode == true)
    inplace_mode::compute_lcp_from_plcp<text_offset_type>(text_length,
        ram_use, sa_filename, output_filename, input_filename, io_volume,
        max_lcp, lcp_sum, true);
  else
    normal_mode::compute_lcp_from_plcp<text_offset_type>(text_length,
        ram_use, sa_filename, output_filename, input_filename, io_volume,
        max_lcp, lcp_sum, true);

  // Print summary.
  long double total_time = utils::wclock() - start;
  long double avg_lcp = (long double)lcp_sum / text_length;
  fprintf(stderr, "\n\nComputation finished. Summary:\n");
  fprintf(stderr, "  Total time = %.2Lfs\n", total_time);
  fprintf(stderr, "  Relative time = %.2Lfus/input symbol\n",
      (1000000.0 * total_time) / text_length);
  fprintf(stderr, "  I/O volume = %lu bytes (%.2Lfbytes/input symbol)\n",
      io_volume, (1.L * io_volume) / text_length);

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

  fprintf(stderr, "  Sum of all LCPs = %lu\n", lcp_sum);
  fprintf(stderr, "  Average LCP = %.2Lf\n", avg_lcp);
  fprintf(stderr, "  Maximal LCP = %lu\n", max_lcp);
}

}  // namespace em_succinct_irreducible_private

#endif  // __SRC_EM_SUCCINCT_IRREDUCIBLE_SRC_COMPUTE_LCP_FROM_PLCP_HPP_INCLUDED
