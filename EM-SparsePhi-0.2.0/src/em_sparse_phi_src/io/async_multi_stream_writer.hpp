/**
 * @file    src/em_sparse_phi_src/io/async_multi_stream_writer.hpp
 * @section LICENCE
 *
 * This file is part of EM-SparsePhi v0.2.0
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

#ifndef __SRC_EM_SPARSE_PHI_SRC_IO_ASYNC_MULTI_STREAM_WRITER_HPP_INCLUDED
#define __SRC_EM_SPARSE_PHI_SRC_IO_ASYNC_MULTI_STREAM_WRITER_HPP_INCLUDED

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "../utils.hpp"


namespace em_sparse_phi_private {

template<typename value_type>
class async_multi_stream_writer {
  private:
    template<typename T>
    struct buffer {
      buffer(std::uint64_t size, T* const mem)
        : m_content(mem), m_size(size) {
        m_filled = 0;
      }

      void flush_to_file(std::FILE *f) {
        utils::write_to_file(m_content, m_filled, f);
        m_filled = 0;
      }

      inline bool empty() const {
        return (m_filled == 0);
      }

      inline bool full() const {
        return (m_filled == m_size);
      }

      inline std::uint64_t size_in_bytes() const {
        return sizeof(T) * m_filled;
      }

      inline std::uint64_t free_space() const {
        return m_size - m_filled;
      }

      T* const m_content;
      const std::uint64_t m_size;

      std::uint64_t m_filled;
    };

    template<typename T>
    struct circular_queue {
      private:
        std::uint64_t m_size;
        std::uint64_t m_filled;
        std::uint64_t m_head;
        std::uint64_t m_tail;
        T *m_data;

      public:
        circular_queue()
          : m_size(1),
            m_filled(0),
            m_head(0),
            m_tail(0),
            m_data(new T[m_size]) {}

        inline void push(T x) {
          m_data[m_head++] = x;
          if (m_head == m_size)
            m_head = 0;
          ++m_filled;
          if (m_filled == m_size)
            enlarge();
        }

        inline T &front() const {
          return m_data[m_tail];
        }

        inline void pop() {
          ++m_tail;
          if (m_tail == m_size)
            m_tail = 0;
          --m_filled;
        }

        inline bool empty() const {
          return (m_filled == 0);
        }

        inline std::uint64_t size() const {
          return m_filled;
        }

        ~circular_queue() {
          delete[] m_data;
        }

      private:
        void enlarge() {
          T *new_data = new T[2 * m_size];
          std::uint64_t left = m_filled;
          m_filled = 0;

          while (left > 0) {
            std::uint64_t tocopy = std::min(left, m_size - m_tail);
            std::copy(m_data + m_tail,
                m_data + m_tail + tocopy, new_data + m_filled);

            m_tail += tocopy;
            if (m_tail == m_size)
              m_tail = 0;
            left -= tocopy;
            m_filled += tocopy;
          }

          m_head = m_filled;
          m_tail = 0;
          m_size <<= 1;
          std::swap(m_data, new_data);
          delete[] new_data;
        }
    };

    template<typename buffer_type>
    struct request {
      request() {}
      request(buffer_type *buffer, std::uint64_t file_id) {
        m_buffer = buffer;
        m_file_id = file_id;
      }

      buffer_type *m_buffer;
      std::uint64_t m_file_id;
    };

    template<typename request_type>
    struct request_queue {
      request_queue()
        : m_no_more_requests(false) {}

      request_type get() {
        request_type ret = m_requests.front();
        m_requests.pop();
        return ret;
      }

      inline void add(request_type request) {
        std::lock_guard<std::mutex> lk(m_mutex);
        m_requests.push(request);
      }

      inline bool empty() const {
        return m_requests.empty();
      }

      circular_queue<request_type> m_requests;  // Must have FIFO property
      std::condition_variable m_cv;
      std::mutex m_mutex;
      bool m_no_more_requests;
    };

    template<typename buffer_type>
    struct buffer_collection {

      // Separate method to allow locking.
      inline void add(buffer_type *buffer) {
        std::lock_guard<std::mutex> lk(m_mutex);
        m_buffers.push_back(buffer);
      }

      buffer_type* get() {
        buffer_type *ret = m_buffers.back();
        m_buffers.pop_back();
        return ret;
      }

      inline bool empty() const {
        return m_buffers.empty();
      }

      std::vector<buffer_type*> m_buffers;
      std::condition_variable m_cv;
      std::mutex m_mutex;
    };

  private:
    template<typename T>
    static void async_io_thread_code(async_multi_stream_writer<T> *caller) {
      typedef buffer<T> buffer_type;
      typedef request<buffer_type> request_type;
      while (true) {

        // Wait for request or until 'no more requests' flag is set.
        std::unique_lock<std::mutex> lk(caller->m_write_requests.m_mutex);
        while (caller->m_write_requests.empty() &&
            !(caller->m_write_requests.m_no_more_requests))
          caller->m_write_requests.m_cv.wait(lk);

        if (caller->m_write_requests.empty() &&
            caller->m_write_requests.m_no_more_requests) {

          // No more requests -- exit.
          lk.unlock();
          break;
        }

        // Extract the buffer from the collection.
        request_type request = caller->m_write_requests.get();
        lk.unlock();

        // Process the request.
        request.m_buffer->flush_to_file(caller->m_files[request.m_file_id]);

        // Add the (now empty) buffer to the collection
        // of empty buffers and notify the waiting thread.
        caller->m_empty_buffers.add(request.m_buffer);
        caller->m_empty_buffers.m_cv.notify_one();
      }
    }

  private:
    typedef buffer<value_type> buffer_type;
    typedef request<buffer_type> request_type;

    std::uint64_t m_bytes_written;
    std::uint64_t m_items_per_buf;

    value_type *m_mem;
    value_type *m_mem_ptr;
    std::vector<std::FILE*> m_files;
    std::vector<buffer_type*> m_buffers;
    buffer_collection<buffer_type> m_empty_buffers;
    request_queue<request_type> m_write_requests;
    std::thread *m_io_thread;

    // Issue a request to write to buffer.
    void issue_write_request(std::uint64_t file_id) {
      request_type req(m_buffers[file_id], file_id);
      m_buffers[file_id] = NULL;
      m_write_requests.add(req);
      m_write_requests.m_cv.notify_one();
    }

    // Get a free buffer from the collection of free buffers.
    buffer_type* get_empty_buffer() {
      std::unique_lock<std::mutex> lk(m_empty_buffers.m_mutex);
      while (m_empty_buffers.empty())
        m_empty_buffers.m_cv.wait(lk);
      buffer_type *ret = m_empty_buffers.get();
      lk.unlock();
      return ret;
    }

  public:
    async_multi_stream_writer(
        std::uint64_t n_files,
        std::uint64_t buf_size_bytes = (std::uint64_t)(1 << 20),
        std::uint64_t n_empty_buffers = (std::uint64_t)4) {

      // Sanity check.
      if (n_files == 0) {
        fprintf(stderr, "\nError in async_multi_stream_writer: "
            "n_files == 0\n");
        std::exit(EXIT_FAILURE);
      }

      // Initialize basic parameters.
      // Works even with n_empty_buffers == 0.
      m_bytes_written = 0;
      m_items_per_buf = utils::disk_block_size<value_type>(buf_size_bytes);

      // Allocate buffers.
      std::uint64_t n_bufs = n_empty_buffers + n_files;
      m_mem = utils::allocate_array<value_type>(n_bufs * m_items_per_buf);
      m_mem_ptr = m_mem;
      for (std::uint64_t j = 0; j < n_empty_buffers; ++j) {
        m_empty_buffers.add(new buffer_type(m_items_per_buf, m_mem_ptr));
        m_mem_ptr += m_items_per_buf;
      }

      // Start the I/O thread.
      m_io_thread = new std::thread(async_io_thread_code<value_type>, this);
    }

    // The added file gets the next available ID (starting from 0).
    void add_file(std::string filename, std::string write_mode =
        std::string("w")) {
      m_buffers.push_back(new buffer_type(m_items_per_buf, m_mem_ptr));
      m_mem_ptr += m_items_per_buf;
      m_files.push_back(utils::file_open_nobuf(filename, write_mode));
    }

    // Write value to i-th file.
    inline void write_to_ith_file(std::uint64_t i, value_type value) {

      // We count I/O volume here (and not in the thread doing I/O) to
      // avoid the situation, where user call bytes_written(), but the
      // I/O thread is still writing the last buffer.
      m_bytes_written += sizeof(value_type);
      m_buffers[i]->m_content[m_buffers[i]->m_filled++] = value;
      if (m_buffers[i]->full()) {
        issue_write_request(i);
        m_buffers[i] = get_empty_buffer();
      }
    }

    // Write values[0..length) to i-th file.
    inline void write_to_ith_file(std::uint64_t i,
        const value_type *values, std::uint64_t length) {
      m_bytes_written += length * sizeof(value_type);
      while (length > 0) {
        std::uint64_t tocopy = std::min(length, m_buffers[i]->free_space());
        std::copy(values, values + tocopy,
            m_buffers[i]->m_content + m_buffers[i]->m_filled);
        m_buffers[i]->m_filled += tocopy;
        values += tocopy;
        length -= tocopy;
        if (m_buffers[i]->full()) {
          issue_write_request(i);
          m_buffers[i] = get_empty_buffer();
        }
      }
    }

    // Return performed I/O in bytes.
    inline std::uint64_t bytes_written() const {
      return m_bytes_written;
    }

    // Destructor.
    ~async_multi_stream_writer() {

      // Flush all buffers.
      std::uint64_t n_buffers = m_buffers.size();
      for (std::uint64_t file_id = 0; file_id < n_buffers; ++file_id) {
        if (!(m_buffers[file_id]->empty()))
          issue_write_request(file_id);
      }

      // Let the I/O thread know that there
      // won't be any more requests.
      std::unique_lock<std::mutex> lk(m_write_requests.m_mutex);
      m_write_requests.m_no_more_requests = true;
      lk.unlock();
      m_write_requests.m_cv.notify_one();

      // Wait for the I/O thread to finish.
      m_io_thread->join();
      delete m_io_thread;

      // Delete buffers and close files.
      for (std::uint64_t file_id = 0; file_id < n_buffers; ++file_id) {
        delete m_buffers[file_id];  // Can be NULL
        std::fclose(m_files[file_id]);
      }

      // Delete empty buffers.
      while (!(m_empty_buffers.empty())) {
        buffer_type *buf = m_empty_buffers.get();
        delete buf;
      }

      utils::deallocate(m_mem);
    }
};

}  // namespace em_sparse_phi_private

#endif  // __SRC_EM_SPARSE_PHI_SRC_IO_ASYNC_MULTI_STREAM_WRITER_HPP_INCLUDED
