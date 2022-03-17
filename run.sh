MEM="4Gi"


for F in ../dna.50MB/dna.50MB
#../abra/abra ../dna.50MB/dna.50MB ../dna/dna ../cere/cere #../dna.txt.16Gi/dna.txt.16Gi 
do
	./sanitize $F
	echo "./psascan -m  $F"
	./construct_sa -m $MEM $F
	./compute_bwt_parallel $F
	./construct_lcp_parallel -m $MEM $F
	./rle_bwt $F
	./gen_patterns $F 30 33554432 ${F}.qs
done

