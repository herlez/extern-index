
// Extracts random patterns from a file

#include <time.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include "mio.hpp"
#include <filesystem>

static int Seed;
#define ACMa 16807
#define ACMm 2147483647
#define ACMq 127773
#define ACMr 2836
#define hi (Seed / ACMq)
#define lo (Seed % ACMq)

static int fst = 1;

	/*
	 * returns a random integer in 0..top-1 
	 */

int64_t
aleat (int64_t top)
{
	long test;
	struct timeval t;
	if (fst)
	{
		srand(0);
		//gettimeofday (&t, NULL);
		//Seed = t.tv_sec * t.tv_usec;
		fst = 0;
	}
	{
		//Seed = ((test =
		//	 ACMa * lo - ACMr * hi) > 0) ? test : test + ACMm;
		//return ((double) Seed) * top / ACMm;
		int64_t result = rand();
		result <<= 32;
		result += rand();
		return result % top;
	}
}

void parse_forbid(unsigned char *forbid, unsigned char ** forbide) {

	int len, i, j;
	len = strlen((char*)forbid);
	
	*forbide = (unsigned char *) malloc((len+1)*sizeof(unsigned char));
	if (*forbide == NULL)
	{
		fprintf (stderr, "Error: cannot allocate %i bytes\n", len+1);
		fprintf (stderr, "errno = %i\n", errno);
		exit (1);
	}

	for(i = 0, j = 0; i < len; i++) {
		if(forbid[i] != '\\') {
			if(forbid[i] != '\n')
				(*forbide)[j++] = forbid[i];
		} else { 
			i++;
			if(i == len) {
				forbid[i-1] = '\0';
				(*forbide)[j] = '\0';
				fprintf (stderr, "Not correct forbidden string: only one \\\n");
				return;
			}
			switch (forbid[i]) {
				case'n':  (*forbide)[j++] = '\n'; break;
				case'\\': (*forbide)[j++] = '\\'; break;
				case'b':  (*forbide)[j++] = '\b'; break;				
				case'e':  (*forbide)[j++] = '\e'; break;
				case'f':  (*forbide)[j++] = '\f'; break;
				case'r':  (*forbide)[j++] = '\r'; break;
				case't':  (*forbide)[j++] = '\t'; break;
				case'v':  (*forbide)[j++] = '\v'; break;
				case'a':  (*forbide)[j++] = '\a'; break;
				case'c':  
						if(i+3 >= len) {
							forbid[i-1] = '\0';
							(*forbide)[j] = '\0';
							fprintf (stderr, "Not correct forbidden string: 3 digits after \\c\n");
							return;
						}
						(*forbide)[j++] = (forbid[i+1]-48)*100 +
										  (forbid[i+2]-48)*10 + (forbid[i+3]-48); 
						i+=3;
						break;					
				default:
				fprintf (stdout, "Unknown escape sequence '\\%c'in forbidden string\n", forbid[i]);
				break;
			}
		}
	}
	(*forbide)[j] = '\0';
}

int main (int argc, char **argv)
{
	int64_t n;
	int len, num, t;
	FILE *ofile;
	//unsigned char *buff;
	unsigned char *forbid, *forbide = NULL;

	if (argc < 5)
	{
		fprintf (stderr,
			 "Usage: genpatterns <file> <length> <number> <patterns file> <forbidden>\n"
			 "  randomly extracts <number> substrings of length <length> from <file>,\n"
			 "  avoiding substrings containing characters in <forbidden>.\n"
			 "  The output file, <patterns file> has a first line of the form:\n"
			 "    # number=<number> length=<length> file=<file> forbidden=<forbidden>\n"
			 "  and then the <number> patterns come successively without any separator.\n"
			 "  <forbidden> uses \\n, \\t, etc. for nonprintable chracters or \\cC\n"
			 "  where C is the ASCII code of the character written using 3 digits.\n\n");
		exit (1);
	}

	if (!std::filesystem::exists(argv[1]))
	{
		fprintf (stderr, "Error: cannot stat file %s\n", argv[1]);
		fprintf (stderr, " errno = %i\n", errno);
		exit (1);
	}
	n = std::filesystem::file_size(argv[1]);

	len = atoi (argv[2]);
	if ((len <= 0) || ((int64_t)len > n))
	{
		fprintf (stderr,
			 "Error: length must be >= 1 and <= file length"
			 " (%" PRId64 ")\n", n);
		exit (1);
	}

	num = atoi (argv[3]);
	if (num < 1)
	{
		fprintf (stderr, "Error: number of patterns must be >= 1\n");
		exit (1);
	}

	if (argc > 5) {
		forbid = (unsigned char*) argv[5];
		parse_forbid(forbid, &forbide);
	} else
		forbid = NULL;

	mio::mmap_sink buff(argv[1]);

	ofile = fopen (argv[4], "w");
	if (ofile == NULL)
	{
		fprintf (stderr, "Error: cannot open file %s for writing\n",
			 argv[4]);
		fprintf (stderr, " errno = %i\n", errno);
		exit (1);
	}

	if (fprintf (ofile, "# number=%i length=%i file=%s forbidden=%s\n",
		     num, len, argv[1],
		     forbid == NULL ? "" : (char *) forbid) <= 0)
	{
		fprintf (stderr, "Error: cannot write file %s\n", argv[4]);
		fprintf (stderr, " errno = %i\n", errno);
		exit (1);
	}

	for (t = 0; t < num; t++)
	{
		int64_t j;
		int l;
		if (!forbide)
			j = aleat (n - len + 1);
		else
		{
			do
			{
				j = aleat (n - len + 1);
				for (l = 0; l < len; l++)
					if (strchr ((char*)forbide, buff[j + l]))
						break;
			}
			while (l < len);
		}
		for (l = 0; l < len; l++)
			if (putc (buff[j + l], ofile) != buff[j + l])
			{
			/*	fprintf (stderr,
					 "Error: cannot write file %s\n",
					 argv[4]);
				fprintf (stderr, " errno = %i\n", errno);
				return -1;*/
			}
	}

	if (fclose (ofile) != 0)
	{
		fprintf (stderr, "Error: cannot write file %s\n", argv[4]);
		fprintf (stderr, " errno = %i\n", errno);
		return -1;
	}

	fprintf (stderr, "File %s successfully generated\n", argv[4]);
	free(forbide);
	return 0;
}
