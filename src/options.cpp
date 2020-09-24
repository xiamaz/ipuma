/*
 HipMer v 2.0, Copyright (c) 2020, The Regents of the University of California,
 through Lawrence Berkeley National Laboratory (subject to receipt of any required
 approvals from the U.S. Dept. of Energy).  All rights reserved."

 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

 (1) Redistributions of source code must retain the above copyright notice, this
 list of conditions and the following disclaimer.

 (2) Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

 (3) Neither the name of the University of California, Lawrence Berkeley National
 Laboratory, U.S. Dept. of Energy nor the names of its contributors may be used to
 endorse or promote products derived from this software without specific prior
 written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
 EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
 SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
 DAMAGE.

 You are under no obligation whatsoever to provide any bug fixes, patches, or upgrades
 to the features, functionality or performance of the source code ("Enhancements") to
 anyone; however, if you choose to make your Enhancements available either publicly,
 or directly to Lawrence Berkeley National Laboratory, without imposing a separate
 written license agreement for such Enhancements, then you hereby grant the following
 license: a  non-exclusive, royalty-free perpetual license to install, use, modify,
 prepare derivative works, incorporate into other computer software, distribute, and
 sublicense such enhancements or derivative works thereof, in binary and source code
 form.
*/

#include "options.hpp"

#include <sys/stat.h>
#include <unistd.h>

#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <upcxx/upcxx.hpp>

#include "CLI11.hpp"
#include "upcxx_utils/log.hpp"
#include "upcxx_utils/mem_profile.hpp"
#include "upcxx_utils/timers.hpp"
#include "utils.hpp"
#include "version.h"

using namespace upcxx_utils;
using namespace std;

#define YES_NO(X) ((X) ? "YES" : "NO")

vector<string> Options::splitter(string in_pattern, string &content) {
  vector<string> split_content;
  std::regex pattern(in_pattern);
  copy(std::sregex_token_iterator(content.begin(), content.end(), pattern, -1), std::sregex_token_iterator(),
       back_inserter(split_content));
  return split_content;
}

bool Options::extract_previous_lens(vector<unsigned> &lens, unsigned k) {
  for (size_t i = 0; i < lens.size(); i++) {
    if (lens[i] == k) {
      lens.erase(lens.begin(), lens.begin() + i + 1);
      return true;
    }
  }
  return false;
}

void Options::get_restart_options() {
  // read existing mhm2.log and get most recent completed stage
  ifstream log_file("mhm2.log");
  if (!log_file.good()) SDIE("Cannot open previous log file mhm2.log");
  string last_stage;
  while (log_file) {
    char buf[1000];
    log_file.getline(buf, 999);
    string line(buf);
    if (line.find("Completed contig") != string::npos)
      last_stage = line;
    else if (line.find("Completed scaffolding") != string::npos)
      last_stage = line;
    if (line.find("Finished in") != string::npos) {
      SDIE("Found the end of the previous run in mhm2.log, cannot restart");
      upcxx::barrier();
    }
  }
  if (!last_stage.empty()) {
    auto fields = split(last_stage, ' ');
    string stage_type = fields[3];
    unsigned k = std::stoi(fields[7]);
    if (stage_type == "contig") {
      if (k == kmer_lens.back()) {
        max_kmer_len = kmer_lens.back();
        kmer_lens = {};
        stage_type = "scaffolding";
        k = scaff_kmer_lens[0];
      } else {
        if (!extract_previous_lens(kmer_lens, k)) SDIE("Cannot find kmer length ", k, " in configuration: ", vec_to_str(kmer_lens));
        prev_kmer_len = k;
      }
      ctgs_fname = "contigs-" + to_string(k) + ".fasta";
    } else if (stage_type == "scaffolding") {
      max_kmer_len = kmer_lens.back();
      kmer_lens = {};
      if (k == scaff_kmer_lens.front()) {
        ctgs_fname = "contigs-" + to_string(k) + ".fasta";
      } else {
        if (k == scaff_kmer_lens.back()) k = scaff_kmer_lens[scaff_kmer_lens.size() - 2];
        ctgs_fname = "scaff-contigs-" + to_string(k) + ".fasta";
      }
      if (!extract_previous_lens(scaff_kmer_lens, k))
        SDIE("Cannot find kmer length ", k, " in configuration: ", vec_to_str(scaff_kmer_lens));
    } else {
      SDIE("Invalid previous stage ", stage_type, " in line '", last_stage, "', could not restart");
    }
    SLOG("\n*** Restarting from previous run at stage ", stage_type,
         " k = ", (stage_type == "contig" ? kmer_lens[0] : scaff_kmer_lens[0]), " ***\n\n");
    SLOG(KLBLUE, "Restart options:\n", "  kmer-lens =              ", vec_to_str(kmer_lens), "\n",
         "  scaff-kmer-lens =        ", vec_to_str(scaff_kmer_lens), "\n", "  prev-kmer-len =          ", prev_kmer_len, "\n",
         "  max-kmer-len =           ", max_kmer_len, "\n", "  contigs =                ", ctgs_fname, KNORM, "\n");
    if (!upcxx::rank_me() && !file_exists(ctgs_fname))
      SDIE("File ", ctgs_fname, " not found. Did the previous run have --checkpoint enabled?");
  } else {
    SLOG("No previously completed stage found, restarting from the beginning\n");
  }
}

void Options::setup_output_dir() {
  if (!upcxx::rank_me()) {
    // create the output directory and stripe it if not doing a restart
    if (restart) {
      if (access(output_dir.c_str(), F_OK) == -1) {
        SDIE("Output directory ", output_dir, " for restart does not exist");
      }
    } else {
      if (mkdir(output_dir.c_str(), S_IRWXU | S_IRWXG | S_IRWXO | S_ISGID /*use default mode/umask */) == -1) {
        // could not create the directory
        if (errno == EEXIST) {
          cerr << KLRED << "WARNING: " << KNORM << "Output directory " << output_dir
               << " already exists. May overwrite existing files\n";
        } else {
          ostringstream oss;
          oss << KLRED << "ERROR: " << KNORM << " Could not create output directory " << output_dir << ": " << strerror(errno)
              << endl;
          throw std::runtime_error(oss.str());
        }
      } else {
        // created the directory - now stripe it if possible
        auto status = std::system("which lfs 2>&1 > /dev/null");
        if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
          string cmd = "lfs setstripe -c -1 " + output_dir;
          auto status = std::system(cmd.c_str());
          if (WIFEXITED(status) && WEXITSTATUS(status) == 0)
            cout << "Set Lustre striping on the output directory\n";
          else
            cout << "Failed to set Lustre striping on output directory: " << WEXITSTATUS(status) << endl;

          // ensure per_thread dir exists and has stripe 1
          string per_thread = output_dir + "/per_thread";
          mkdir(per_thread.c_str(), S_IRWXU | S_IRWXG | S_IRWXO | S_ISGID /*use default mode/umask */);  // ignore any errors
          cmd = "lfs setstripe -c 1 " + per_thread;
          status = std::system(cmd.c_str());
          if (WIFEXITED(status) && WEXITSTATUS(status) == 0)
            cout << "Set Lustre striping on the per_thread output directory\n";
          else
            cout << "Failed to set Lustre striping on per_thread output directory: " << WEXITSTATUS(status) << endl;
          // this should avoid contention on the filesystem when ranks start racing to creating these top levels
          for (int i = 0; i < rank_n(); i += 1000) {
            char basepath[256];
            sprintf(basepath, "%s/%08d", per_thread.c_str(), i);
            auto ret = mkdir(basepath, S_IRWXU | S_IRWXG | S_IRWXO | S_ISGID /*use default mode/umask */);
            if (ret != 0) break;  // ignore any errors, just stop
          }
        }
      }
    }
  }

  upcxx::barrier();
  // after we change to the output directory, relative paths could be incorrect, so make sure we have the correct path of the
  // reads files
  char cwd_str[FILENAME_MAX];
  if (!getcwd(cwd_str, FILENAME_MAX)) SDIE("Cannot get current working directory: ", strerror(errno));
  for (auto &fname : reads_fnames) {
    if (fname[0] != '/') fname = string(cwd_str) + "/" + fname;
  }
  // all change to the output directory
  if (chdir(output_dir.c_str()) == -1 && !upcxx::rank_me()) {
    DIE("Cannot change to output directory ", output_dir, ": ", strerror(errno));
  }
  upcxx::barrier();
}

void Options::setup_log_file() {
  if (!upcxx::rank_me()) {
    // check to see if mhm2.log exists. If so, and not restarting, rename it
    if (file_exists("mhm2.log") && !restart) {
      string new_log_fname = "mhm2-" + get_current_time(true) + ".log";
      cerr << KLRED << "WARNING: " << KNORM << output_dir << "/mhm2.log exists. Renaming to " << output_dir << "/" << new_log_fname
           << endl;
      if (rename("mhm2.log", new_log_fname.c_str()) == -1) DIE("Could not rename mhm2.log: ", strerror(errno));
    } else if (!file_exists("mhm2.log") && restart) {
      DIE("Could not restart - missing mhm2.log in tis directory");
    }
  }
  upcxx::barrier();
}

Options::Options() {
  output_dir = string("mhm2-run-<reads_fname[0]>-n") + to_string(upcxx::rank_n()) + "-N" +
               to_string(upcxx::rank_n() / upcxx::local_team().rank_n()) + "-" + get_current_time(true);
}
Options::~Options() {
  flush_logger();
  cleanup();
}

void Options::cleanup() {
  // cleanup and close loggers that Options opened in load
  close_logger();
#ifdef DEBUG
  close_dbg();
#endif
}

bool Options::load(int argc, char **argv) {
  // MHM2 version v0.1-a0decc6-master (Release) built on 2020-04-08T22:15:40 with g++
  string full_version_str = "MHM2 version " + string(MHM2_VERSION) + "-" + string(MHM2_BRANCH) + " with upcxx-utils " +
                            string(UPCXX_UTILS_VERSION) + " built on " + string(MHM2_BUILD_DATE);
  CLI::App app(full_version_str);
  app.add_option("-r, --reads", reads_fnames, "Files containing merged and unmerged reads in FASTQ format (comma separated).")
      ->delimiter(',')
      ->check(CLI::ExistingFile);
  app.add_option("-p, --paired-reads", paired_fnames, "Pairs of files for the same insert in FASTQ format (comma separated).")
      ->delimiter(',')
      ->check(CLI::ExistingFile);
  /*
                 ->check([](const string &s) {
                   if (s == "lala") return s;
                   return string();
                 });
   */
  app.add_option("-i, --insert", insert_size, "Insert size (average:stddev)")
      ->delimiter(':')
      ->expected(2)
      ->check(CLI::Range(1, 10000));
  auto *kmer_lens_opt =
      app.add_option("-k, --kmer-lens", kmer_lens, "kmer lengths (comma separated)")->delimiter(',')->capture_default_str();
  app.add_option("--max-kmer-len", max_kmer_len, "Maximum kmer length (need to specify if only scaffolding)")
      ->capture_default_str()
      ->check(CLI::Range(0, 159));
  app.add_option("--prev-kmer-len", prev_kmer_len,
                 "Previous kmer length (need to specify if contigging and contig file is specified)")
      ->capture_default_str()
      ->check(CLI::Range(0, 159));
  auto *scaff_kmer_lens_opt = app.add_option("-s, --scaff-kmer-lens", scaff_kmer_lens,
                                             "kmer lengths for scaffolding (comma separated). 0 to disable scaffolding")
                                  ->delimiter(',')
                                  ->capture_default_str();
  app.add_option("-Q, --quality-offset", qual_offset, "Phred encoding offset")
      ->capture_default_str()
      ->check(CLI::IsMember({0, 33, 64}));
  app.add_option("-c, --contigs", ctgs_fname, "File with contigs used for restart");
  //                   ->check(CLI::ExistingFile);
  app.add_option("--dynamic-min-depth", dynamic_min_depth,
                 "Dynamic min. depth for DeBruijn graph traversal - set to 1.0 for a single genome")
      ->capture_default_str()
      ->check(CLI::Range(0.1, 1.0));
  app.add_option("--min-depth-thres", dmin_thres, "Absolute mininimum depth threshold for DeBruijn graph traversal")
      ->capture_default_str()
      ->check(CLI::Range(1, 100));
  app.add_option("--max-kmer-store", max_kmer_store_mb, "Maximum size for kmer store in MB per rank. 0 for auto 1% memory")
      ->capture_default_str()
      ->check(CLI::Range(0, 1000));
  app.add_option("--max-rpcs-in-flight", max_rpcs_in_flight, "Maximum number of RPCs in flight, per process (0 = unlimited)")
      ->capture_default_str()
      ->check(CLI::Range(0, 10000));
  app.add_flag("--use-heavy-hitters", use_heavy_hitters, "Activate the Heavy Hitter Streaming Store")->capture_default_str();
  app.add_option("--min-ctg-print-len", min_ctg_print_len, "Minimum length required for printing a contig in the final assembly")
      ->capture_default_str()
      ->check(CLI::Range(0, 100000));
  app.add_option("--break-scaff-Ns", break_scaff_Ns, "Number of Ns allowed before a scaffold is broken")
      ->capture_default_str()
      ->check(CLI::Range(0, 1000));
  app.add_option("--ranks-per-gpu", ranks_per_gpu,
                 "Override the automatic detction of ranks/gpu (i.e. local_team().rank_n() / devices).")
      ->capture_default_str()
      ->check(CLI::Range(0, (int)upcxx::local_team().rank_n() * 8));
  auto *output_dir_opt = app.add_option("-o,--output", output_dir, "Output directory")->capture_default_str();
  app.add_flag("--force-bloom", force_bloom, "Always use bloom filters")
      ->default_val(force_bloom ? "true" : "false")
      ->capture_default_str()
      ->multi_option_policy();
  app.add_flag("--checkpoint", checkpoint, "Checkpoint after each contig round")
      ->default_val(checkpoint ? "true" : "false")
      ->capture_default_str()
      ->multi_option_policy();
  app.add_flag("--use-kmer-depths", use_kmer_depths,
               "Use kmer depths for scaffolding decisions instead of alignment depths (the default)")
      ->capture_default_str();
  app.add_flag("--restart", restart, "Restart in previous directory where a run failed")->capture_default_str();
  app.add_flag("--pin", pin_by, "Restrict processes according to logical CPUs, cores (groups of hardware threads), "
                                "or NUMA domains (cpu, core, numa, none) - default is cpu ")
      ->capture_default_str()
      ->check(CLI::IsMember({"cpu", "core", "numa", "none"}));
  app.add_flag("--post-asm-align", post_assm_aln, "Align reads to final assembly")->capture_default_str();
  app.add_flag("--post-asm-abd", post_assm_abundances, "Compute and output abundances for final assembly (used by MetaBAT)")
      ->capture_default_str();
  app.add_flag("--write-gfa", dump_gfa, "Dump scaffolding contig graphs in GFA2 format")->capture_default_str();
  app.add_flag("--post-asm-only", post_assm_only, "Only run post assembly")->capture_default_str();
  app.add_flag("--progress", show_progress, "Show progress")->capture_default_str();
  app.add_flag("-v, --verbose", verbose, "Verbose output")->capture_default_str();

  auto *cfg_opt = app.set_config("--config", "", "Load options from a configuration file");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    if (upcxx::rank_me() == 0) app.exit(e);
    return false;
  }

  if (!paired_fnames.empty()) {
    // convert pairs to colon ':' separated single files for FastqReader to process
    if (paired_fnames.size() % 2 != 0) SDIE("Did not get pairs of files in -p: ", paired_fnames.size());
    while (paired_fnames.size() >= 2) {
      reads_fnames.push_back(paired_fnames[0] + ":" + paired_fnames[1]);
      paired_fnames.erase(paired_fnames.begin());
      paired_fnames.erase(paired_fnames.begin());
    }
  }

  // we can get restart incorrectly set in the config file from restarted runs
  if (*cfg_opt) {
    restart = false;
    app.get_option("--restart")->default_val("false");
  }

  if (!upcxx::rank_me() && !restart && reads_fnames.empty()) {
    // FIXME: there appears to be no way to do exactly this with a validator or ->required()
    DIE("Require read names if not restarting");
  }

  if (!upcxx::rank_me() && post_assm_only && ctgs_fname.empty()) {
    DIE("For running only post assembly analysis, require --contigs (e.g. pass final_assembly.fasta)");
  }

  upcxx::barrier();

  if (!*output_dir_opt) {
    string first_read_fname = remove_file_ext(get_basename(reads_fnames[0]));
    output_dir = "mhm2-run-" + first_read_fname + "-n" + to_string(upcxx::rank_n()) + "-N" +
                 to_string(upcxx::rank_n() / upcxx::local_team().rank_n()) + "-" + get_current_time(true);
    output_dir_opt->default_val(output_dir);
  }

  if (restart) {
    // this mucking about is to ensure we don't get multiple failures messages if the config file does not parse
    if (!upcxx::rank_me()) app.parse_config(output_dir + "/mhm2.config");
    upcxx::barrier();
    if (upcxx::rank_me()) app.parse_config(output_dir + "/mhm2.config");
  }

  // make sure we only use defaults for kmer lens if none of them were set by the user
  if (!*kmer_lens_opt && !*scaff_kmer_lens_opt) {
    kmer_lens = {21, 33, 55, 77, 99};
    scaff_kmer_lens = {99, 33};
    // set the option default strings so that the correct values are printed and saved to the .config file
    kmer_lens_opt->default_str("21 33 55 77 99");
    scaff_kmer_lens_opt->default_str("99 33");
  } else if (kmer_lens.size() && *kmer_lens_opt && !*scaff_kmer_lens_opt) {
    // use the last and second from kmer_lens for scaffolding
    auto n = kmer_lens.size();
    if (n == 1) {
      scaff_kmer_lens = kmer_lens;
      scaff_kmer_lens_opt->default_str(to_string(scaff_kmer_lens[0]));
    } else {
      scaff_kmer_lens = {kmer_lens[n - 1], kmer_lens[n == 2 ? 0 : 1]};
      scaff_kmer_lens_opt->default_str(to_string(scaff_kmer_lens[0]) + " " + to_string(scaff_kmer_lens[1]));
    }
  } else if (scaff_kmer_lens.size() == 1 && scaff_kmer_lens[0] == 0) {
    // disable scaffolding rounds
    scaff_kmer_lens.clear();
  }

  setup_output_dir();
  setup_log_file();

  if (max_kmer_store_mb == 0) {
    // use 1% of the minimum available memory
    max_kmer_store_mb = get_free_mem() / 1024 / 1024 / 100;
    max_kmer_store_mb = upcxx::reduce_all(max_kmer_store_mb / local_team().rank_n(), upcxx::op_fast_min).wait();
  }

  auto logger_t = chrono::high_resolution_clock::now();
  if (upcxx::local_team().rank_me() == 0) {
    // open 1 log per node
    // rank0 has mhm2.log in rundir, all others have logs in per_thread
    init_logger("mhm2.log", verbose, rank_me());
  }

  barrier();
  chrono::duration<double> logger_t_elapsed = chrono::high_resolution_clock::now() - logger_t;
  SLOG_VERBOSE("init_logger took ", setprecision(2), fixed, logger_t_elapsed.count(), " s at ", get_current_time(), "\n");

#ifdef DEBUG
  open_dbg("debug");
#endif

  SLOG(KLBLUE, full_version_str, KNORM, "\n");

  if (restart) get_restart_options();

  if (upcxx::rank_me() == 0) {
    // print out all compiler definitions
    SLOG_VERBOSE(KLBLUE, "_________________________", KNORM, "\n");
    SLOG_VERBOSE(KLBLUE, "Compiler definitions:", KNORM, "\n");
    std::istringstream all_defs_ss(ALL_DEFNS);
    vector<string> all_defs((std::istream_iterator<string>(all_defs_ss)), std::istream_iterator<string>());
    for (auto &def : all_defs) SLOG_VERBOSE("  ", def, "\n");
    SLOG_VERBOSE("_________________________\n");
    SLOG(KLBLUE, "Options:", KNORM, "\n");
    auto all_opts_strings = app.config_to_str_vector(true, false);
    for (auto &opt_str : all_opts_strings) SLOG(KLBLUE, opt_str, KNORM, "\n");
    SLOG(KLBLUE, "_________________________", KNORM, "\n");
  }
  auto num_nodes = upcxx::rank_n() / upcxx::local_team().rank_n();
  SLOG("Starting run with ", upcxx::rank_n(), " processes on ", num_nodes, " node", (num_nodes > 1 ? "s" : ""), " at ",
       get_current_time(), "\n");
#ifdef DEBUG
  SWARN("Running low-performance debug mode");
#endif
  if (!upcxx::rank_me()) {
    // write out configuration file for restarts
    ofstream ofs("mhm2.config");
    ofs << app.config_to_str(true, true);
  }
  upcxx::barrier();
  return true;
}