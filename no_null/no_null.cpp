#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "mio.hpp"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cout << "Usage: no_null TEXT_PATH \n";
    return -1;
  }
  std::filesystem::path text_path = argv[1];

  if (!std::filesystem::exists(text_path)) {
    std::cout << "FILE " << text_path << " NOT FOUND. Usage: no_null TEXT_PATH. \n";
    return -2;
  }

  // First push back terminal if there is none.
  {
    std::fstream text(text_path, text.binary | text.in | text.out | text.app);
    text.seekg(-1, std::ios_base::end);
    std::cout << "Last character is 0b" << text.peek() << ". ";
    if (text.peek() > '\02') {
      std::cout << "Terminal byte added to " << text_path << ".\n";
      text.put('\02');
    }
  }

  // Then rewrite every terminal to '0x1'
  mio::mmap_sink text(text_path.c_str());
  if (text.size() == 0) {
    std::cout << "FILE " << text_path << " IS EMPTY TEXT. Usage: no_null TEXT_PATH. \n";
    return -3;
  }

  size_t counter_0 = 0;
  size_t counter_1 = 0;
  size_t counter_2 = 0;
  for (size_t i = 0; i < text.size() - 1; ++i) {
    switch (text[i])
    {
    case '\00':
      text[i] = '\03';
      ++counter_0; 
      break;
    case '\01':
      text[i] = '\03';
      ++counter_1;
    case '\02':
      text[i] = '\03';
      ++counter_2;
    default:
      break;
    }

    if ((i << (64 - 30)) == 0) {
      std::cout << "\r" << (i >> 30) << "GiB" << std::flush;
    }
  }
  text[text.size()-1] = '\02'; //Terminal

  std::cout << "\n #\\00 bytes=" << counter_0 << " #\\01 bytes=" << counter_1 << "\n";
  std::error_code error;
  text.sync(error);
  text.unmap();
}