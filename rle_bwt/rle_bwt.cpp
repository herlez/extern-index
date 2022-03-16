#include <vector>
#include <iostream>
#include "mio.hpp"
#include <filesystem>
#include <fstream>

int main(int argc, char ** argv) {
    if(argc != 2) {
        std::cout << "Usage: bwt_rle TEXT_PATH \n";
        return -1;
    }
    std::filesystem::path text_path = argv[1];
    std::filesystem::path bwt_path = text_path;
    bwt_path += ".bwt";

    if(!std::filesystem::exists(bwt_path)) {
        std::cout << "FILE " << bwt_path << " NOT FOUND. Usage: bwt_rle BWT_PATH. \n";
        return -2;
    }

    mio::mmap_source bwt(bwt_path.c_str());
    if(bwt.size() == 0) {
        std::cout << "FILE " << bwt_path << " IS EMPTY TEXT. Usage: bwt_rle BWT_PATH. \n";
        return -3;
    }

    
    std::filesystem::path bwt_rlenc_path = bwt_path;
    bwt_rlenc_path.replace_extension("rlenc");
    if(std::filesystem::exists(bwt_rlenc_path)) {
        std::cout << "File " << bwt_rlenc_path << " already exists. Rewrite? (y or n)\n";
        char input;
        std::cin >> input;
        if(input != 'y') {
            return -1;
        }
    }
    
    std::filesystem::path bwt_rlength_path = bwt_path;
    bwt_rlength_path.replace_extension("rlength");
    if(std::filesystem::exists(bwt_rlength_path)) {
        std::cout << "File " << bwt_rlength_path << " already exists. Rewrite? (y or n)\n";
        char input;
        std::cin >> input;
        if(input != 'y') {
            return -1;
        }
    }


    std::ofstream bwt_rlenc(bwt_rlenc_path, std::ios_base::trunc);
    std::ofstream bwt_rlength(bwt_rlength_path, std::ios_base::trunc);

    
    char last_c = bwt[0];
    size_t run_length = 1;
    size_t run_count = 1;  

    
    for(size_t i = 1; i < bwt.size(); ++i) {
        char c = bwt[i];
        if(c == last_c) {
            ++run_length;
        } else {
            bwt_rlenc << last_c;
            bwt_rlength.write(reinterpret_cast<const char*>(&run_length), 5);

            last_c = c;
            run_length = 1;
            ++run_count;
        }
    }
    bwt_rlenc << last_c;
    bwt_rlength.write(reinterpret_cast<const char*>(&run_length), 5);


    std::filesystem::path info_path = bwt_path;
    info_path.replace_extension(".info");
    std::ofstream info(info_path);
    std::cout << "r=" << run_count 
              << " n/r=" << static_cast<float>(bwt.size()) / run_count 
              << '\n';
    info << "r=" << run_count 
              << " n/r=" << static_cast<float>(bwt.size()) / run_count 
              << '\n';


}