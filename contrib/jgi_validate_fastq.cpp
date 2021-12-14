// module load sparshash
// g++ jgi_validate_fastq.cpp -o jgi_validate_fastq -O3 -lz -I${BOOST_ROOT}/include ${BOOST_ROOT}/lib/libboost_iostreams.a -fopenmp
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <boost/unordered_map.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/pool/pool.hpp>
#include <functional>
#include <string.h>
#include "omp.h"

#define LOG(msg) { std::stringstream s; s << "T" << omp_get_thread_num() << ": " << msg << std::endl; std::string str = s.str(); std::cerr << str; }
#define MAX_LEN 500
#define MAX_LOG 100

using namespace std;
struct StrHash : public std::unary_function<const char *, size_t> {
	size_t operator()(const char *_str) const {
		size_t len = strlen(_str);
		size_t hash = 0;
		const int64_t *str64 = (const int64_t*) _str;
		while (len > sizeof(int64_t)) {
			hash = std::hash<int64_t>{}(*(str64++)) ^ (hash << 1);
			len -= sizeof(int64_t);
		}
		const char *str = (const char*) str64;
		while (len--) {
			hash = std::hash<char>{}(*(str++)) ^ (hash << 1);
		}
		return hash;
	}
};
struct StrEqual {
	bool operator()(const char * a, const char * b) const {
		return strcmp(a, b) == 0;
	}
};
typedef boost::unordered_map<const char *, unsigned char, StrHash, StrEqual> NameMap;


string findGreatestCommonSubName(string gsn, string name) {
  if (gsn.empty())
    return name;
  int len = gsn.length(); 
  while (len > 0) {
    if (gsn.find(name.substr(0,len)) != string::npos) {
      return gsn.substr(0,len);
    }
    len--;
  }
  return string();
}

string parseLengths(std::vector<int64_t> &lengths) {
  std::stringstream s;
  for (int i = 0; i < lengths.size(); i++) {
    if (lengths[i] > 0) {
      s << i << ":\t" << ((i<=MAX_LEN) ? lengths[i] : exp(i-MAX_LEN)) << "\n";
    }
  } 
  return s.str();
}  

uint64_t getFileSize(istream &is) {
                assert( !is.eof() );
                assert( is.good() );
                assert( !is.fail() );
                fstream::streampos current = is.tellg();
                is.seekg(0, ios_base::end);
                uint64_t size = is.tellg();
                is.seekg(current);
                return size;
}


int main(int argc, char *argv[]) {

  int MAX_QUAL_SCORE=51;
  int MIN_QUAL = 33, MAX_QUAL;
  bool parseLen = true;

  const static char *USAGE = "Usage: jgi_validate_fastq [-h] [-64 | -32] [-max51] [-nolen] file.fastq[.gz] [...]\n\tChecks for proper 4 line records with sane quality scores (default 33), no duplicate record names\n\n";
  if (argc <= 1 || strcmp(argv[1], "-h") == 0) {
    cerr << USAGE << endl;
    exit( strcmp(argc<=1?"":argv[1], "-h") );
  }
  int firstarg = 1;
  int exit_status = 0;

  if (strcmp(argv[firstarg], "-64") == 0) {
    MIN_QUAL = 64;
    firstarg++;
  } else if (strcmp(argv[firstarg], "-32") == 0) {
    MIN_QUAL = 32;
    firstarg++;
  }
  if (strncmp(argv[firstarg], "-max", 4) == 0) {
    MAX_QUAL_SCORE = atoi(argv[firstarg]+4);
    firstarg++;
  }
  if (strncmp(argv[firstarg], "-nolen", 4) == 0) {
    parseLen = false;
    firstarg++;
  }
  MAX_QUAL = MIN_QUAL + MAX_QUAL_SCORE;
  cout << "Using MIN_QUAL=" << MIN_QUAL << " and MAX_QUAL=" << MAX_QUAL << " MAX_QUAL_SCORE=" << MAX_QUAL_SCORE << endl;


#pragma omp parallel for
  for(int i = firstarg; i < argc; i++) {
    string greatestCommonSubName;
    std::vector<int64_t> lengths;
    lengths.resize(MAX_LEN+MAX_LOG, 0);
    NameMap allNames;
    boost::pool<> nameAllocator(sizeof(char));
    ifstream file(argv[i]);
    uint64_t fileSize = getFileSize(file);
    int64_t paired = 0, read1 = 0, read2 = 0, unknown = 0, pairError = 0;
    boost::iostreams::filtering_istream in;
    if (string(argv[i]).find(".gz") != string::npos) {
      in.push(boost::iostreams::gzip_decompressor());
      LOG("Decompressing " << argv[i]);
    } else {
      LOG("Reading " << argv[i]);
    }

    in.push(file);

    int64_t countDuplicates = 0;
    int64_t count=0, bases=0;
    string s1,s2,s3,s4,message;
    while(true) {
       getline(in, s1);
       if (s1.size() != strlen(s1.c_str())) {
          LOG("Record at line " << count << " has different string lengths label (" << s1.size() << ", " << strlen(s1.c_str()) << ")... are there NULLs in the file?!");
       }

       if (s1.empty()) break;
       getline(in, s2);
       if (s2.size() != strlen(s2.c_str())) {
          LOG("Record at line " << count << " has different string lengths fasta (" << s2.size() << ", " << strlen(s2.c_str()) << ")... are there NULLs in the file?!");
       }
       getline(in, s3);
       if (s3.size() != strlen(s3.c_str())) {
          LOG("Record at line " << count << " has different string lengths qualabel (" << s3.size() << ", " << strlen(s3.c_str()) << ")... are there NULLs in the file?!");
       }
       getline(in, s4);
       if (s4.size() != strlen(s4.c_str())) {
          LOG("Record at line " << count << " has different string lengths quals (" << s4.size() << ", " << strlen(s4.c_str()) << ")... are there NULLs in the file?!");
       }

       bool seqPass = s2.find_first_not_of("ACGTNacgt") == string::npos;
       int seqlen = s2.size();
       if (seqlen > MAX_LEN) {
           seqlen = MAX_LEN + log(seqlen);
           if (seqlen > MAX_LEN + MAX_LOG) seqlen = MAX_LEN + MAX_LOG - 1;
       }
       lengths[seqlen]++;
       bool qualPass = true;
       bool qualPass2 = true;
       for(int j = 0; j < s4.length(); j++) {
          qualPass &= (int) s4[j] >= MIN_QUAL;
          qualPass &= (int) s4[j] <= MAX_QUAL;
       }
       if (!qualPass) {
          // once more with specifics
          for(int j = 0; j < s4.length(); j++) {
            if ((int) s4[j] < MIN_QUAL) message += string("qual '") + s4[j] + "' (" + to_string((int) s4[j]) + ") < '" + ((char) MIN_QUAL) + "' (" + to_string((int)MIN_QUAL) + ") at " + to_string(j) + ".";
            if ((int) s4[j] > MAX_QUAL) message += string("qual '") + s4[j] + "' (" + to_string((int) s4[j]) + ") > '" + ((char) MAX_QUAL) + "' (" + to_string((int)MAX_QUAL) + ") at " + to_string(j) + ".";
          }
 
       }
       if (s1.empty() || s2.empty() || s3.empty() || s4.empty() || s1[0] != '@' || s3[0] != '+' || s2.length() != s4.length() || (!seqPass) || (!qualPass)) {
         LOG(argv[i] << "\tRecord " << count << " line " << (count*4) << " pos " << in.tellg() << ":\n\n" << s1 << "\n" << s2 << "\n" << s3 << "\n" << s4);
         if (!seqPass) LOG(argv[i] << ": sequence characters wrong");
         if (!qualPass) {
             LOG(argv[i] << ": quality characters wrong... maybe this is 64 base?\n" << message);
         }
         exit_status++;
         break;
       }
       
       if (count == 20000) {
          long elem = allNames.size() * fileSize / file.tellg();
          allNames.reserve(4 * elem * 110 / 100 + 1000);
       }

       int namePos = s1.find_first_of(" \t");
       string old = s1;
       if (namePos != string::npos && s1.length() > namePos + 4 && s1[namePos+2] == ':' && s1[namePos+4] == ':' && s1[namePos+6] == ':' && (s1[namePos+1] == '1' || s1[namePos+1] == '2') && (s1[namePos+3] == 'Y' || s1[namePos+3] == 'N')) {
         // change from "@name [12]:[YN]:?:.*" to "@name/[12]	[YN]:?:.*"
         s1[namePos] = '/';
         namePos += 2;
         s1[namePos] = '\0';
       }
       int offset=1;
       if (namePos == string::npos) namePos = s1.length();
       string nameTemplate = s1.substr(offset, (s1[namePos-2] == '/' ? namePos-2 : namePos) - offset);
       NameMap::iterator it = allNames.find( nameTemplate.c_str() );
       if (it == allNames.end()) {
         char * storedName = (char*) nameAllocator.ordered_malloc(nameTemplate.size()+1);
         strcpy(storedName, nameTemplate.c_str());
         NameMap::value_type val(storedName, 0);
         it = allNames.insert(it, val);
       }
       // 0 first observation - no count yet
       // 1, 4, 5 - only pair1, only pair2, pair1 & pair2
       // 16 something else (@name/[^12])
       // 64 unpaired
       unsigned char &v = it->second;
       //printf("template: %s from %s (%s)\n", nameTemplate.c_str(), old.c_str(), s1.c_str());
       char pairIdx = s1[namePos-1];
       if (namePos > 3) {
         unsigned char x = s1[namePos-2] == '/' ? (s1[namePos-1] == '1' ? 1 : (s1[namePos-1] == '2' ? 4 : 16 )) : 64;
         if (x == 1) read1++;
         else if (x == 4) read2++;
         else if (x == 16) unknown++;
         v += x;
       } else {
         v += 64;
       }
       if (v == 5) paired++;
       if (v > 64) pairError++;
       if (v != 1 && v != 4 && v != 5 && v != 64) {
    	 if (countDuplicates++ == 0) {
    		 LOG("Record: (" << nameTemplate << ") " << s1 << " repeated with code " << (int) v << "\n");
                 exit_status++;
    		 break;
    	 }
       }
       greatestCommonSubName = findGreatestCommonSubName(greatestCommonSubName, s1);
       
       count++;
       bases += s2.size() - 1;
    }
    if (countDuplicates > 0) {
        exit_status++;
    	LOG(argv[i] << " had " << countDuplicates << " duplicated record");
    }
    if (read2 > 0 && read1 != read2) {
        exit_status++;
        LOG(argv[i] << " had different read1 & read2 counts (" << read1 << ", " << read2 << "). paired=" << paired);
    }
    LOG(argv[i] << " had subname: '" << greatestCommonSubName << "' with " << count << " reads and " << bases << " bases." << " paired: " << paired << " read1: " << read1 << " read2: " << read2 << " unknown: " << unknown << " pairError: " << pairError << "\n" << (parseLen ? parseLengths(lengths) : ""));
    allNames.clear();
    nameAllocator.release_memory();
  }

  if (exit_status != 0)
    LOG("There were " << exit_status << " files that were invalid");
  exit(exit_status);

}
