cd EM-Sanitize;
g++ -std=c++20 -O3 sanitize.cpp -o sanitize;
cp sanitize ../;
cd ..

cd psascan;
make;
cp construct_sa ../;
cd ..;

cd pem-bwt;
make;
cp compute_bwt_sequential ../;
cp compute_bwt_parallel ../;
cd ..;

cd EM-SparsePhi-0.2.0;
make;
cp construct_lcp_sequential ../;
cp construct_lcp_parallel ../;
cd ..;

cd EM-RL_enc_bwt;
g++ -std=c++20 -O3 rle_bwt.cpp -o rle_bwt;
cp rle_bwt ../;
cd ../;

cd EM-GenPatterns;
g++ -std=c++20 -O3 gen_patterns.c -o gen_patterns;
cp gen_patterns ../;
cd ../;
