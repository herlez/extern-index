
for F in ../abra/abra ../dna.50MB/dna.50MB ../dna/dna ../cere/cere #../dna.txt.16Gi/dna.txt.16Gi 
do
	./sanitize $F
	./psascan -m 400gi $F
	./compute_bwt_parallel $F
	./construct_lcp_parallel $F -m 400gi
	./rle_bwt $F
done

