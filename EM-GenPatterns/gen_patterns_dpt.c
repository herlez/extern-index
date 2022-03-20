#include "mio.hpp"
#include <iostream>
#include <string>
#include <filesystem>
#include <fstream>
#include <vector>

size_t get_number_of_patterns(std::string header) {
  size_t start_pos = header.find("number=");
  if (start_pos == std::string::npos || start_pos + 7 >= header.size()) {
    return -1;
  }
  start_pos += 7;

  size_t end_pos = header.substr(start_pos).find(" ");
  if (end_pos == std::string::npos) {
    return -2;
  }

  return std::atoi(header.substr(start_pos).substr(0, end_pos).c_str());
}

size_t get_patterns_length(std::string header) {
  size_t start_pos = header.find("length=");
  if (start_pos == std::string::npos || start_pos + 7 >= header.size()) {
    return -1;
  }

  start_pos += 7;

  size_t end_pos = header.substr(start_pos).find(" ");
  if (end_pos == std::string::npos) {
    return -2;
  }

  size_t n = std::atoi(header.substr(start_pos).substr(0, end_pos).c_str());

  return n;
}

std::vector<std::string> load_patterns(std::filesystem::path path, size_t num_patterns = std::numeric_limits<size_t>::max()) {
  std::vector<std::string> patterns;
  std::ifstream ifs(path);

  std::string header;
  std::getline(ifs, header);

  size_t n = std::min(get_number_of_patterns(header), num_patterns);
  size_t m = get_patterns_length(header);

  // extract patterns from file and search them in the index
  for (size_t i = 0; i < n; ++i) {
    std::string p = std::string();

    for (size_t j = 0; j < m; ++j) {
      char c;
      ifs.get(c);
      p += c;
    }
    patterns.push_back(p);
  }

  return patterns;
}



int main(int argc, char **argv) {

    if(argc < 2) {
        std::cout << "Usage: gen_patterns_dpt P&C_QUERIES_PATH\n";
        return -1; 
    }
    std::filesystem::path patterns_path = argv[1];
    std::cout << "Read from " << patterns_path << "\n";
    std::vector<std::string> patterns = load_patterns(patterns_path);

    std::filesystem::path dpt_patterns_path = patterns_path;
    dpt_patterns_path += "dpt";
    std::cout << "Write to " << dpt_patterns_path << "\n";

    std::ofstream of(dpt_patterns_path);
    for(auto const& pat : patterns) {
        of << pat << '\n';
    }

    return 0;

}