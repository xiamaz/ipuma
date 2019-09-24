#ifndef __UTILS_H
#define __UTILS_H

#include <sys/stat.h>
#include <unistd.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <limits>
#include <upcxx/upcxx.hpp>

#include "colors.h"

using std::string;
using std::stringstream;
using std::ostringstream;
using std::ostream;
using std::ifstream;
using std::ofstream;
using std::to_string;
using std::cout;
using std::cerr;
using std::min;

#ifndef NDEBUG
#define DEBUG
#endif

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define ONE_MB (1024*1024)
#define ONE_GB (ONE_MB*1024)

extern ofstream _logstream;
extern bool _verbose;

inline void init_logger() {
  if (!upcxx::rank_me()) _logstream.open("mhm.log");
}

inline void set_logger_verbose(bool verbose) {
  _verbose = verbose;
}
  
inline void logger(ostringstream &os) {}

template <typename T, typename... Params>
inline void logger(ostringstream &os, T first, Params... params) {
  os << first;
  logger(os, params ...);
}

template <typename T, typename... Params>
inline void logger(ostream &stream, bool fail, bool serial, bool flush, T first, Params... params) {
  if (serial && upcxx::rank_me()) return;
  ostringstream os;
  os << first;
  logger(os, params ...);
  if (fail) {
    std::cerr << "\n" << KNORM;
    throw std::runtime_error(os.str());
  }
  stream << os.str();
  if (flush) stream.flush();
}


#define SOUT(...) do {                                   \
    logger(cout, false, true, true, ##__VA_ARGS__);      \
  } while (0)
#define WARN(...)                                                       \
  logger(cerr, false, false, true, KRED, "[", upcxx::rank_me(), "] <", __FILENAME__, ":", __LINE__, "> WARNING: ", ##__VA_ARGS__, KNORM, "\n")
#define DIE(...)                                                        \
  logger(cerr, true, false, true, KLRED, "[", upcxx::rank_me(), "] <", __FILENAME__ , ":", __LINE__, "> ERROR: ", ##__VA_ARGS__, KNORM, "\n")
#define SWARN(...)                                                      \
  logger(cerr, false, true, true, KRED, "[", upcxx::rank_me(), "] <", __FILENAME__, ":", __LINE__, "> WARNING: ", ##__VA_ARGS__, KNORM, "\n")
#define SDIE(...)                                                       \
  logger(cerr, true, true, true, KLRED, "[", upcxx::rank_me(), "] <", __FILENAME__ , ":", __LINE__, "> ERROR: ", ##__VA_ARGS__, KNORM, "\n")


#define SLOG(...) do {                                              \
    logger(cout, false, true, true, ##__VA_ARGS__);                 \
    logger(_logstream, false, true, true, ##__VA_ARGS__);           \
  } while (0)

#define SLOG_VERBOSE(...) do {                                       \
    if (_verbose) logger(cout, false, true, true, ##__VA_ARGS__);    \
    logger(_logstream, false, true, true, ##__VA_ARGS__);           \
  } while (0)

#ifdef DEBUG
extern ofstream _dbgstream;
#define DBG(...) do {                                                   \
    if (_dbgstream) {                                                   \
      logger(_dbgstream, false, false, true, "<", __FILENAME__, ":", __LINE__, "> ", ##__VA_ARGS__); \
    }                                                                   \
  } while(0)
#else
#define DBG(...)
#endif

static double get_free_mem_gb(void) {
  string buf;
  ifstream f("/proc/meminfo");
  double mem_free = 0;
  while (!f.eof()) {
    getline(f, buf);
    if (buf.find("MemFree") == 0 || buf.find("Buffers") == 0 || buf.find("Cached") == 0) {
      stringstream fields;
      string units;
      string name;
      double mem;
      fields << buf;
      fields >> name >> mem >> units;
      if (units[0] == 'k') mem /= ONE_MB;
      mem_free += mem;
    }
  }
  return mem_free;
}

class IntermittentTimer {
  
  std::chrono::time_point<std::chrono::high_resolution_clock> t;
  double t_elapsed;
  string name;
public:
  IntermittentTimer(const string &name) {
    this->name = name;
    t_elapsed = 0;
  }
  
  ~IntermittentTimer() {
    SLOG_VERBOSE(KLCYAN, "--- Elapsed time for ", name, ": ", std::setprecision(2), std::fixed, t_elapsed, " s ---\n", KNORM);
    DBG("--- Elapsed time for ", name, ": ", std::setprecision(2), std::fixed, t_elapsed, " s ---\n");
  }

  void start() {
    t = std::chrono::high_resolution_clock::now();
  }
  
  void stop() {
    std::chrono::duration<double> interval = std::chrono::high_resolution_clock::now() - t;
    t_elapsed += interval.count();
  }
};


class Timer {
  std::chrono::time_point<std::chrono::high_resolution_clock> t;
  string name;
  double init_free_mem;
  bool always_show;
public:
  Timer(const string &name, bool always_show=false) : always_show(always_show) {
    t = std::chrono::high_resolution_clock::now();
    this->name = name;
    if (!upcxx::rank_me()) init_free_mem = get_free_mem_gb();
    if (always_show) SLOG(KLCYAN, "-- ", name, " (", init_free_mem, " GB free) --\n", KNORM);
    else SLOG_VERBOSE(KLCYAN, "-- ", name, " (", init_free_mem, " GB free) --\n", KNORM);
  }
  
  ~Timer() {
    std::chrono::duration<double> t_elapsed = std::chrono::high_resolution_clock::now() - t;
    DBG(KLCYAN, "-- ", name, " took ", std::setprecision(2), std::fixed, t_elapsed.count(), " s --\n", KNORM);
    upcxx::barrier();
    t_elapsed = std::chrono::high_resolution_clock::now() - t;
    if (always_show) {
      SLOG(KLCYAN, "-- ", name, " took ", std::setprecision(2), std::fixed, t_elapsed.count(), " s (used ",
           (init_free_mem - get_free_mem_gb()), " GB) --\n", KNORM);
    } else {
      SLOG_VERBOSE(KLCYAN, "-- ", name, " took ", std::setprecision(2), std::fixed, t_elapsed.count(), " s (used ",
                   (init_free_mem - get_free_mem_gb()), " GB) --\n", KNORM);
    }
  }
};


inline string tail(const string &s, int n) {
  return s.substr(s.size() - n);
}

inline string head(const string &s, int n) {
  return s.substr(0, n);
}

inline string perc_str(int64_t num, int64_t tot) {
  ostringstream os;
  os.precision(2);
  os << std::fixed; 
  os << num << " (" << 100.0 * num / tot << "%)";
  return os.str();
}

inline std::vector<string> split(const string &s, char delim) {
  std::vector<string> elems;
  std::stringstream ss(s);
  string token;
  while (std::getline(ss, token, delim)) elems.push_back(token);
  return elems;
}

inline bool file_exists(const string &fname) {
  ifstream ifs(fname, std::ios_base::binary);
  return ifs.good();
}

inline void replace_spaces(string &s) {
  for (int i = 0; i < s.size(); i++)
    if (s[i] == ' ') s[i] = '_';
}

inline string revcomp(const string &seq) {
  string seq_rc = "";
  seq_rc.reserve(seq.size());
  for (int i = seq.size() - 1; i >= 0; i--) {
    switch (seq[i]) {
      case 'A': seq_rc += 'T'; break;
      case 'C': seq_rc += 'G'; break;
      case 'G': seq_rc += 'C'; break;
      case 'T': seq_rc += 'A'; break;
      case 'N': seq_rc += 'N'; break;
      default:
        DIE("Illegal char in revcomp of '", seq, "'\n"); 
    }
  }
  return seq_rc;
}

inline char comp_nucleotide(char ch) {
  switch (ch) {
      case 'A': return 'T';
      case 'C': return 'G'; 
      case 'G': return 'C';
      case 'T': return 'A';
      case 'N': return 'N';
      case '0': return '0';
      default: DIE("Illegal char in revcomp of '", ch, "'\n"); 
  }
  return 0;
}

// returns 1 when it created the directory, 0 otherwise, -1 if there is an error
inline int check_dir(const char *path) {
  if (0 != access(path, F_OK)) {
    if (ENOENT == errno) {
      // does not exist
      // note: we make the directory to be world writable, so others can delete it later if we
      // crash to avoid cluttering up memory
      mode_t oldumask = umask(0000);
      if (0 != mkdir(path, 0777) && 0 != access(path, F_OK)) {
        umask(oldumask);
        fprintf(stderr, "Could not create the (missing) directory: %s (%s)", path, strerror(errno));
        return -1;
      }
      umask(oldumask);
    }
    if (ENOTDIR == errno) {
      // not a directory
      fprintf(stderr, "Expected %s was a directory!", path);
      return -1;
    }
  } else {
    return 0;
  }
  return 1;
}

// replaces the given path with a rank based path, inserting a rank-based directory
// example:  get_rank_path("path/to/file_output_data.txt", rank) -> "path/to/per_rank/<rankdir>/<rank>/file_output_data.txt"
// of if rank == -1, "path/to/per_rank/file_output_data.txt"
inline bool get_rank_path(string &fname, int rank) {
  char buf[MAX_FILE_PATH];
  strcpy(buf, fname.c_str());
  int pathlen = strlen(buf);
  char newPath[MAX_FILE_PATH*2+50];
  char *lastslash = strrchr(buf, '/');
  int checkDirs = 0;
  int thisDir;
  char *lastdir = NULL;

  if (pathlen + 25 >= MAX_FILE_PATH) {
    WARN("File path is too long (max: ", MAX_FILE_PATH, "): ", buf, "\n");
    return false;
  }
  if (lastslash) {
    *lastslash = '\0';
  }
  if (rank < 0) {
    if (lastslash) {
      snprintf(newPath, MAX_FILE_PATH*2+50, "%s/per_rank/%s", buf, lastslash + 1);
      checkDirs = 1;
    } else {
      snprintf(newPath, MAX_FILE_PATH*2+50, "per_rank/%s", buf);
      checkDirs = 1;
    }
  } else {
    if (lastslash) {
      snprintf(newPath, MAX_FILE_PATH*2+50, "%s/per_rank/%08d/%08d/%s", buf, rank / MAX_RANKS_PER_DIR, rank, lastslash + 1);
      checkDirs = 3;
    } else {
      snprintf(newPath, MAX_FILE_PATH*2+50, "per_rank/%08d/%08d/%s", rank / MAX_RANKS_PER_DIR, rank, buf);
      checkDirs = 3;
    }
  }
  strcpy(buf, newPath);
  while (checkDirs > 0) {
    strcpy(newPath, buf);
    thisDir = checkDirs;
    while (thisDir--) {
      lastdir = strrchr(newPath, '/');
      if (!lastdir) {
        WARN("What is happening here?!?!\n");
        return false;
      }
      *lastdir = '\0';
    }
    check_dir(newPath);
    checkDirs--;
  }
  fname = buf;
  return true;
}

inline std::vector<string> find_per_rank_files(string &fname_list, const string &ext) {
  std::vector<string> full_fnames;
  auto fnames = split(fname_list, ',');
  for (auto fname : fnames) {
    // first check for gzip file
    fname += ext;
    get_rank_path(fname, upcxx::rank_me());
    string gz_fname = fname + ".gz";
    struct stat stbuf;
    if (stat(gz_fname.c_str(), &stbuf) == 0) {
      // gzip file exists
      SLOG_VERBOSE("Found compressed file '", gz_fname, "'\n");
      fname = gz_fname;
    } else {
      // no gz file - look for plain file
      if (stat(fname.c_str(), &stbuf) != 0)
        SDIE("File '", fname, "' cannot be accessed (either .gz or not): ", strerror(errno), "\n");
    }
    full_fnames.push_back(fname);
  }
  return full_fnames;
}

inline string get_current_time() {
  auto t = std::time(nullptr);
  std::ostringstream os;
  os << std::put_time(localtime(&t), "%D %T");
  return os.str();
}

inline int hamming_dist(const string &s1, const string &s2, bool require_equal_len=true) {
  if (require_equal_len && s2.size() != s1.size())//abs((int)(s2.size() - s1.size())) > 1)
    DIE("Hamming distance substring lengths don't match, ", s1.size(), ", ", s2.size(), "\n");
  int d = 0;
  for (int i = 0; i < min(s1.size(), s2.size()); i++) 
    if (s1[i] != s2[i]) d++;
  return d;
}

inline bool is_overlap_mismatch(int dist, int overlap) {
  const int MISMATCH_THRES = 5;
  if (dist > MISMATCH_THRES || dist > overlap / 10) return true;
  return false;
}

inline std::pair<int, int> min_hamming_dist(const string &s1, const string &s2, int max_overlap, int expected_overlap=-1) {
  int min_dist = max_overlap;
  if (expected_overlap != -1) {
    int min_dist = hamming_dist(tail(s1, expected_overlap), head(s2, expected_overlap));
    if (!is_overlap_mismatch(min_dist, expected_overlap)) return {min_dist, expected_overlap};
  }
  for (int d = std::min(max_overlap, (int)std::min(s1.size(), s2.size())); d >= 10; d--) {
    int dist = hamming_dist(tail(s1, d), head(s2, d));
    if (dist < min_dist) {
      min_dist = dist;
      expected_overlap = d;
      if (dist == 0) break;
    }
  }
  return {min_dist, expected_overlap};
}

static bool has_ending (string const &full_string, string const &ending) {
  if (full_string.length() >= ending.length()) 
    return (0 == full_string.compare(full_string.length() - ending.length(), ending.length(), ending));
  return false;
}

static int does_file_exist(string fname) {
  struct stat s;
  if (stat(fname.c_str(), &s) != 0) return 0;
  return 1;
}

static void write_num_file(string fname, int64_t maxReadLength) {
  ofstream out(fname);
  if (!out) { std::string error("Could not write to " + fname); throw error; }
  out << std::to_string(maxReadLength);
  out.close();
}

static string get_size_str(int64_t sz) {
  if (sz >= ONE_GB * 1024l) return to_string(sz / (ONE_GB * 1024l)) + "TB";
  else if (sz >= ONE_GB) return to_string(sz / ONE_GB) + "GB";
  else if (sz >= ONE_MB) return to_string(sz / ONE_MB) + "MB";
  else if (sz >= 1024) return to_string(sz / 1024) + "KB";
  else return to_string(sz) + "B";
}

static string remove_file_ext(const string &fname) {
  size_t lastdot = fname.find_last_of(".");
  if (lastdot == std::string::npos) return fname;
  return fname.substr(0, lastdot); 
}

static string get_merged_reads_fname(const string &reads_fname) {
  string out_fname = remove_file_ext(reads_fname) + "-merged.fastq.gz";
  get_rank_path(out_fname, upcxx::rank_me());
  return out_fname;
}

static int64_t get_file_size(string fname) {
  struct stat s;
  if (stat(fname.c_str(), &s) != 0) return -1;
  return s.st_size;
}

static size_t get_uncompressed_file_size(string fname) {
  string uncompressedSizeFname = fname + ".uncompressedSize";
  ifstream f(uncompressedSizeFname, std::ios::binary);
  if (!f) DIE("Cannot get uncompressed size for file ", fname);
  size_t sz = 0;
  f.read((char*)&sz, sizeof(size_t));
  return sz;
}

static void write_uncompressed_file_size(string fname, size_t sz) {
  ofstream f(fname, std::ios::binary);
  f.write((char*)&sz, sizeof(size_t));
  f.close();
}


#endif
