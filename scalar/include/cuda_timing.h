
#ifndef CUDA_TIMING_H_INCLUDED
#define CUDA_TIMING_H_INCLUDED
// Written by Gabor Daniel Balogh, Pazmany Peter Catholic University 2020
// TODO merge with timing.h

#include <chrono>
#include <map>
#include <vector>
#include <iostream>
#include "cutil_inline.h"
#include <nvToolsExt.h>

class Timing {
  using clock      = std::chrono::high_resolution_clock;
  using time_point = clock::time_point;

  struct LoopData {
    int index   = 0;
    int parent  = 0;
    double time = 0.0;
    time_point current;
    std::vector<cudaEvent_t> event_pairs;
  };
  std::string name;

  static std::map<std::string, LoopData> loops;
  static std::vector<int> stack;
  static int counter;

  static void sumCudaEvents();
  static void reportWithParent(int parent, const std::string &indentation);

public:
  explicit Timing(std::string &&name_p) : name(std::move(name_p)) {
    startTimer(name);
  }
  ~Timing() { stopTimer(name); }

  static void startTimer(const std::string &_name);
  static void stopTimer(const std::string &_name);
  static void startTimerCUDA(const std::string &_name, cudaStream_t stream);
  static void stopTimerCUDA(const std::string &_name, cudaStream_t stream);
  static void pushRange(const std::string &_name);
  static void popRange();
  static void markStart(const std::string &_name);
  static void report();
};

#if PROFILING
#  define BEGIN_PROFILING(name) Timing::startTimer(name)
#  define END_PROFILING(name)   Timing::stopTimer(name)

#  define BEGIN_PROFILING_CUDA(name,stream) Timing::startTimerCUDA(name,stream)
#  define END_PROFILING_CUDA(name,stream)   Timing::stopTimerCUDA(name,stream)

#  define PROFILE_SCOPE(name) Timing timer##__LINE__(name)
#  define PROFILE_FUNCTION()  PROFILE_SCOPE(__FUNCTION__)
#  define PROFILE_REPORT()    Timing::report()
#  if PROFILING > 1
#    define BEGIN_PROFILING2(name)      BEGIN_PROFILING(name)
#    define END_PROFILING2(name)        END_PROFILING(name)
#    define BEGIN_PROFILING_CUDA2(name,stream) BEGIN_PROFILING_CUDA(name,stream)
#    define END_PROFILING_CUDA2(name,stream)   END_PROFILING_CUDA(name,stream)
#  else
#    define BEGIN_PROFILING2(name)      Timing::pushRange(name)
#    define END_PROFILING2(name)        Timing::popRange()
#    define BEGIN_PROFILING_CUDA2(name,stream) Timing::markStart(name)
#    define END_PROFILING_CUDA2(name,stream)
#  endif
#else
#  define BEGIN_PROFILING(name)
#  define END_PROFILING(name)

#  define BEGIN_PROFILING_CUDA(name,stream)
#  define END_PROFILING_CUDA(name,stream)

#  define BEGIN_PROFILING2(name)
#  define END_PROFILING2(name)
#  define BEGIN_PROFILING_CUDA2(name,stream)
#  define END_PROFILING_CUDA2(name,stream)

#  define PROFILE_SCOPE(name)
#  define PROFILE_FUNCTION()
#  define PROFILE_REPORT()
#endif

#endif /* ifndef CUDA_TIMING_H_INCLUDED */
