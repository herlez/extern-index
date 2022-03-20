cd EM-Sanitize;
g++ -std=c++20 -O3 sanitize.cpp -o sanitize;
mv sanitize ../;
cd ..

cd psascan;
make;
mv construct_sa ../;
cd ..;

cd pem-bwt;
make;
mv compute_bwt_sequential ../;
mv compute_bwt_parallel ../;
cd ..;

cd EM-SparsePhi-0.2.0;
make;
mv construct_lcp_sequential ../;
mv construct_lcp_parallel ../;
cd ..;

cd EM-RL_enc_bwt;
g++ -std=c++20 -O3 rle_bwt.cpp -o rle_bwt;
mv rle_bwt ../;
cd ../;

cd EM-GenPatterns;
g++ -std=c++20 -O3 gen_patterns.c -o gen_patterns;
g++ -std=c++20 -O3 gen_patterns_dpt.c -o gen_patterns_dpt;
mv gen_patterns ../;
mv gen_patterns_dpt ../;
cd ../;
