/**
 * Load profiler and the experimental faster_loading boosts, see
 * include/rex/kernel/xboxkrnl/load_profile.h.
 */

#include <rex/kernel/xboxkrnl/load_profile.h>

#include <algorithm>
#include <cstring>
#include <ctime>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>

#include <rex/cvar.h>
#include <rex/logging.h>
#include <rex/chrono/clock.h>
#include <rex/system/xthread.h>

#ifdef _WIN32
#include <windows.h>
#include <tlhelp32.h>
#include <dbghelp.h>
#endif

REXCVAR_DEFINE_BOOL(faster_loading, false, "Kernel",
                    "EXPERIMENTAL, off by default. Shortens game loads by running the guest "
                    "clock faster for short moments (idle_boost while the game only waits on "
                    "its own timers, stall_boost while no frames are presented), so the "
                    "game's timed waits end sooner. In tests this saved about 4-6 s of a 22 s "
                    "load. Side effects: game time jumps forward a few seconds in total during "
                    "a load, so anything timed in game time can briefly run fast, and a small "
                    "background thread plus per-read/per-sleep bookkeeping runs. Logs "
                    "[idle-boost] and [stall-boost] lines.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(diag_load_profile, false, "Kernel",
                    "Diagnostic: once a second, log guest disk reads and, per thread, CPU use, "
                    "sleeps, waits and yields ([load-prof] lines), long waits ([wait]) and "
                    "stalls ([stall]). For finding what loading time is spent on. Costs CPU.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_STRING(diag_load_profile_dump, "", "Kernel",
                      "Diagnostic: with diag_load_profile_sample, append every sampled call "
                      "chain (24 deep) to this file once a second, as \"tid count rva rva ...\" "
                      "lines (rva = offset in fable_2.exe), for offline profiles. Empty = off.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(diag_load_profile_sample, false, "Kernel",
                    "Diagnostic: with diag_load_profile, sample where the busiest guest threads "
                    "spend their time and log the hottest functions and call chains once a "
                    "second ([load-prof-hot] and [load-prof-chain] lines). Uses fable_2.pdb "
                    "next to the exe for names.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(load_timer, false, "Kernel",
                    "Diagnostic: time loads and log [load-timer] lines (also runs while "
                    "faster_loading is on). A load is a stretch of heavy disk reading "
                    "(load_timer_start_mb in one second) that ends after load_timer_quiet_s "
                    "seconds without reads.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_INT32(load_timer_start_mb, 20, "Kernel",
                     "MB read within one second that start a load (load_timer, faster_loading).")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_INT32(load_timer_quiet_s, 5, "Kernel",
                     "Seconds without reads that end a load (load_timer, faster_loading).")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_DOUBLE(stall_boost, 8.0, "Kernel",
                      "faster_loading: while no frame is presented for diag_stall_ms during a "
                      "load, run the guest clock this many times faster, so a guest-time "
                      "timeout in the game (about 5 s near the end of the Continue load) ends "
                      "sooner. 1 = off, max 16.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_DOUBLE(idle_boost, 16.0, "Kernel",
                      "faster_loading: early in a load, while the GameThread mostly sleeps and "
                      "almost nothing is read from disk, run the guest clock this many times "
                      "faster so the game's timed waits end sooner. 1 = off, max 16 (higher "
                      "values are capped).")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_INT32(idle_boost_window_s, 30, "Kernel",
                     "idle_boost: only within this many seconds after a load starts.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_INT32(stall_boost_max_s, 15, "Kernel",
                     "faster_loading: give up and return to normal speed after a boost has "
                     "run this many seconds.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_INT32(diag_wait_min_ms, 100, "Kernel",
                     "diag_load_profile: guest waits and sleeps at least this long are logged "
                     "as [wait] lines with the game functions that made them.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_INT32(diag_stall_ms, 250, "Kernel",
                     "A gap of this many ms between presented frames is a stall: it starts "
                     "stall_boost (faster_loading) and, with diag_load_profile, is logged as "
                     "a [stall] with what every game thread is doing during it.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

namespace rex::kernel::xboxkrnl::load_profile {

namespace {

// diag_load_profile: the detailed profiler (reporter, sampler, wait logs).
bool Diag() { return REXCVAR_GET(diag_load_profile); }
// faster_loading: the experimental boosts.
bool BoostOn() { return REXCVAR_GET(faster_loading); }
// Either one needs the per-thread read/sleep bookkeeping and the monitor thread.
bool ActiveNow() { return Diag() || BoostOn(); }
std::atomic<bool> g_control_started{false};
// NowUs() until which a load_timer session counts as running (set by ControlMain).
std::atomic<uint64_t> g_load_session_until_us{0};
// NowUs() of the current [load-timer] START (0 = none yet).
std::atomic<uint64_t> g_load_start_us{0};
void ControlMain();

struct ThreadStats {
  uint32_t guest_tid = 0;
  uint32_t host_tid = 0;
  std::atomic<uint64_t> read_calls{0};
  std::atomic<uint64_t> read_bytes{0};
  std::atomic<uint64_t> read_us{0};
  std::atomic<uint64_t> delay_calls{0};
  std::atomic<uint64_t> delay_us{0};
  std::atomic<uint64_t> delay_requested_us{0};
  std::atomic<uint64_t> wait_calls{0};
  std::atomic<uint64_t> wait_us{0};
  std::atomic<uint64_t> yields{0};
  // The wait in progress (wait_start_us != 0), for the wait monitor.
  std::atomic<uint64_t> wait_seq{0};
  std::atomic<uint64_t> wait_start_us{0};
  std::atomic<uint32_t> wait_kind{0};
  std::atomic<uint32_t> wait_object{0};
  std::atomic<uint32_t> wait_count{0};
  std::atomic<const char*> wait_what{nullptr};
  std::atomic<uint64_t> wait_end_seq{0};
  std::atomic<uint64_t> wait_end_us{0};
};

// Presented frames (VdSwap), for the stall detector.
std::atomic<uint64_t> g_last_swap_us{0};
std::atomic<uint64_t> g_swap_count{0};
std::atomic<uint64_t> g_max_swap_gap_us{0};
std::atomic<bool> g_monitor_started{false};
void WaitMonitorMain();

struct Snapshot {
  uint64_t read_calls = 0, read_bytes = 0, read_us = 0;
  uint64_t delay_calls = 0, delay_us = 0, delay_requested_us = 0;
  uint64_t wait_calls = 0, wait_us = 0;
  uint64_t yields = 0;
};

std::mutex g_threads_mutex;
std::vector<ThreadStats*> g_threads;
std::atomic<bool> g_reporter_started{false};
thread_local ThreadStats* t_stats = nullptr;

void ReporterMain();

ThreadStats* Stats() {
  if (!t_stats) {
    auto* stats = new ThreadStats();
    stats->guest_tid = rex::system::XThread::IsInThread()
                           ? rex::system::XThread::GetCurrentThreadId()
                           : 0;
#ifdef _WIN32
    stats->host_tid = GetCurrentThreadId();
#endif
    {
      std::lock_guard<std::mutex> lock(g_threads_mutex);
      g_threads.push_back(stats);
    }
    t_stats = stats;
    bool expected = false;
    if (g_reporter_started.compare_exchange_strong(expected, true)) {
      std::thread(ReporterMain).detach();
    }
  }
  return t_stats;
}

Snapshot Take(const ThreadStats& stats) {
  Snapshot s;
  s.read_calls = stats.read_calls.load(std::memory_order_relaxed);
  s.read_bytes = stats.read_bytes.load(std::memory_order_relaxed);
  s.read_us = stats.read_us.load(std::memory_order_relaxed);
  s.delay_calls = stats.delay_calls.load(std::memory_order_relaxed);
  s.delay_us = stats.delay_us.load(std::memory_order_relaxed);
  s.delay_requested_us = stats.delay_requested_us.load(std::memory_order_relaxed);
  s.wait_calls = stats.wait_calls.load(std::memory_order_relaxed);
  s.wait_us = stats.wait_us.load(std::memory_order_relaxed);
  s.yields = stats.yields.load(std::memory_order_relaxed);
  return s;
}

struct HostThreadCpu {
  uint64_t cpu_100ns = 0;
  std::string name;
};

// CPU time (kernel + user) and description of every thread of this process.
std::unordered_map<uint32_t, HostThreadCpu> SampleHostThreads() {
  std::unordered_map<uint32_t, HostThreadCpu> result;
#ifdef _WIN32
  using GetThreadDescriptionFn = HRESULT(WINAPI*)(HANDLE, PWSTR*);
  static GetThreadDescriptionFn get_thread_description = reinterpret_cast<GetThreadDescriptionFn>(
      GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "GetThreadDescription"));
  HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
  if (snapshot == INVALID_HANDLE_VALUE) {
    return result;
  }
  DWORD process_id = GetCurrentProcessId();
  THREADENTRY32 entry;
  entry.dwSize = sizeof(entry);
  if (Thread32First(snapshot, &entry)) {
    do {
      if (entry.th32OwnerProcessID != process_id) {
        continue;
      }
      HANDLE thread = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, entry.th32ThreadID);
      if (!thread) {
        continue;
      }
      FILETIME creation, exit, kernel, user;
      if (GetThreadTimes(thread, &creation, &exit, &kernel, &user)) {
        HostThreadCpu cpu;
        cpu.cpu_100ns = ((uint64_t(kernel.dwHighDateTime) << 32) | kernel.dwLowDateTime) +
                        ((uint64_t(user.dwHighDateTime) << 32) | user.dwLowDateTime);
        if (get_thread_description) {
          PWSTR description = nullptr;
          if (SUCCEEDED(get_thread_description(thread, &description)) && description) {
            for (const wchar_t* c = description; *c && cpu.name.size() < 40; ++c) {
              cpu.name.push_back(*c < 128 ? char(*c) : '?');
            }
            LocalFree(description);
          }
        }
        result.emplace(uint32_t(entry.th32ThreadID), std::move(cpu));
      }
      CloseHandle(thread);
    } while (Thread32Next(snapshot, &entry));
  }
  CloseHandle(snapshot);
#endif
  return result;
}


// ---------------------------------------------------------------------------
// Code sampler (diag_load_profile_sample): suspends the busiest guest threads
// about every 2 ms, records the instruction pointer and the first return
// address inside the game executable found on the stack, and reports the
// hottest functions once a second.
constexpr size_t kChainDepth = 24;

struct SampleBuckets {
  uint64_t samples = 0;
  std::unordered_map<uint64_t, uint32_t> leaf;
  std::unordered_map<uint64_t, uint32_t> game_frame;
  // Call chains: key = the first (up to 8) addresses inside the game executable
  // found on the stack, innermost first, as raw bytes.
  std::unordered_map<std::string, uint32_t> chain;
  // diag_load_profile_dump only: the leaf and the stack values inside any
  // module of kDumpModules (host code too), innermost first.
  std::unordered_map<std::string, uint32_t> host_chain;
};

std::mutex g_sample_mutex;
std::vector<uint32_t> g_sample_targets;
std::unordered_map<uint32_t, SampleBuckets> g_sample_buckets;
std::atomic<bool> g_sampler_started{false};

#ifdef _WIN32
uintptr_t g_exe_begin = 0;
uintptr_t g_exe_end = 0;

// Modules whose addresses the dump records (index = the "m" in "m:rva").
const wchar_t* const kDumpModules[] = {nullptr, L"rexgpu-xenos.dll", L"rexruntime.dll",
                                       L"ntdll.dll", L"KernelBase.dll", L"nvwgf2umx.dll",
                                       L"d3d12core.dll", L"kernel32.dll"};
constexpr size_t kDumpModuleCount = sizeof(kDumpModules) / sizeof(kDumpModules[0]);
uintptr_t g_module_begin[kDumpModuleCount] = {};
uintptr_t g_module_end[kDumpModuleCount] = {};

void InitModuleRange(size_t index) {
  auto base = reinterpret_cast<uintptr_t>(GetModuleHandleW(kDumpModules[index]));
  if (!base) {
    return;
  }
  auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
  auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS*>(base + dos->e_lfanew);
  g_module_begin[index] = base;
  g_module_end[index] = base + nt->OptionalHeader.SizeOfImage;
}

int DumpModuleOf(uint64_t address) {
  for (size_t i = 0; i < kDumpModuleCount; ++i) {
    if (address >= g_module_begin[i] && address < g_module_end[i]) {
      return int(i);
    }
  }
  return -1;
}

void InitExeRange() {
  for (size_t i = 0; i < kDumpModuleCount; ++i) {
    InitModuleRange(i);
  }
  g_exe_begin = g_module_begin[0];
  g_exe_end = g_module_end[0];
}

void SamplerMain() {
  InitExeRange();
  std::unordered_map<uint32_t, HANDLE> handles;
  uint64_t stack[1024];
  for (;;) {
    if (!REXCVAR_GET(diag_load_profile) || !REXCVAR_GET(diag_load_profile_sample)) {
      Sleep(100);
      continue;
    }
    Sleep(2);
    const bool dump_on = !REXCVAR_GET(diag_load_profile_dump).empty();
    std::vector<uint32_t> targets;
    {
      std::lock_guard<std::mutex> lock(g_sample_mutex);
      targets = g_sample_targets;
    }
    for (uint32_t tid : targets) {
      auto handle_it = handles.find(tid);
      if (handle_it == handles.end()) {
        HANDLE opened = OpenThread(
            THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, tid);
        handle_it = handles.emplace(tid, opened).first;
      }
      HANDLE thread = handle_it->second;
      if (!thread) {
        continue;
      }
      if (SuspendThread(thread) == DWORD(-1)) {
        continue;
      }
      // No allocation or locking while the thread is suspended.
      CONTEXT context = {};
      context.ContextFlags = CONTEXT_CONTROL;
      bool have_context = GetThreadContext(thread, &context) != FALSE;
      SIZE_T stack_bytes = 0;
      if (have_context) {
        if (!ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(context.Rsp),
                               stack, sizeof(stack), &stack_bytes)) {
          // Near the top of the stack: try a smaller window.
          if (!ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(context.Rsp),
                                 stack, 256, &stack_bytes)) {
            stack_bytes = 0;
          }
        }
      }
      ResumeThread(thread);
      if (!have_context) {
        continue;
      }
      uint64_t leaf = context.Rip;
      uint64_t game_frame = 0;
      if (leaf >= g_exe_begin && leaf < g_exe_end) {
        game_frame = leaf;
      } else {
        for (size_t i = 0; i < stack_bytes / sizeof(uint64_t); ++i) {
          if (stack[i] >= g_exe_begin && stack[i] < g_exe_end) {
            game_frame = stack[i];
            break;
          }
        }
      }
      // Call chain: addresses inside the game executable, innermost first.
      std::string chain_key;
      {
        uint64_t previous = 0;
        size_t count = 0;
        auto add = [&](uint64_t address) {
          if (count >= kChainDepth || address < g_exe_begin || address >= g_exe_end || address == previous) {
            return;
          }
          chain_key.append(reinterpret_cast<const char*>(&address), sizeof(address));
          previous = address;
          ++count;
        };
        add(leaf);
        for (size_t i = 0; i < stack_bytes / sizeof(uint64_t) && count < kChainDepth; ++i) {
          add(stack[i]);
        }
      }
      std::string host_key;
      if (dump_on) {
        uint64_t previous = 0;
        size_t count = 0;
        auto add = [&](uint64_t address) {
          if (count >= kChainDepth || address == previous || DumpModuleOf(address) < 0) {
            return;
          }
          host_key.append(reinterpret_cast<const char*>(&address), sizeof(address));
          previous = address;
          ++count;
        };
        add(leaf);
        if (host_key.empty()) {
          // Leaf outside the known modules (another DLL): keep a marker.
          uint64_t unknown = 0;
          host_key.append(reinterpret_cast<const char*>(&unknown), sizeof(unknown));
          ++count;
        }
        for (size_t i = 0; i < stack_bytes / sizeof(uint64_t) && count < kChainDepth; ++i) {
          add(stack[i]);
        }
      }
      std::lock_guard<std::mutex> lock(g_sample_mutex);
      SampleBuckets& buckets = g_sample_buckets[tid];
      if (!host_key.empty()) {
        ++buckets.host_chain[host_key];
      }
      ++buckets.samples;
      ++buckets.leaf[leaf];
      if (game_frame) {
        ++buckets.game_frame[game_frame];
      }
      if (!chain_key.empty()) {
        ++buckets.chain[chain_key];
      }
    }
  }
}

// Function name for an address (fable_2.pdb via DbgHelp), else module+offset.
// DbgHelp is single-threaded: callers go through Symbolize().
std::string SymbolizeUnlocked(uint64_t address) {
  static std::unordered_map<uint64_t, std::string> cache;
  static bool initialized = false;
  using SymInitializeFn = BOOL(WINAPI*)(HANDLE, PCSTR, BOOL);
  using SymSetOptionsFn = DWORD(WINAPI*)(DWORD);
  using SymFromAddrFn = BOOL(WINAPI*)(HANDLE, DWORD64, PDWORD64, PSYMBOL_INFO);
  static SymFromAddrFn sym_from_addr = nullptr;
  if (!initialized) {
    initialized = true;
    HMODULE dbghelp = LoadLibraryW(L"dbghelp.dll");
    if (dbghelp) {
      auto sym_set_options =
          reinterpret_cast<SymSetOptionsFn>(GetProcAddress(dbghelp, "SymSetOptions"));
      auto sym_initialize =
          reinterpret_cast<SymInitializeFn>(GetProcAddress(dbghelp, "SymInitialize"));
      sym_from_addr = reinterpret_cast<SymFromAddrFn>(GetProcAddress(dbghelp, "SymFromAddr"));
      if (sym_set_options) {
        sym_set_options(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
      }
      if (!sym_initialize || !sym_initialize(GetCurrentProcess(), nullptr, TRUE)) {
        sym_from_addr = nullptr;
      }
    }
  }
  auto cached = cache.find(address);
  if (cached != cache.end()) {
    return cached->second;
  }
  std::string name;
  if (sym_from_addr) {
    alignas(SYMBOL_INFO) char buffer[sizeof(SYMBOL_INFO) + 256];
    auto* symbol = reinterpret_cast<SYMBOL_INFO*>(buffer);
    std::memset(buffer, 0, sizeof(buffer));
    symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
    symbol->MaxNameLen = 255;
    DWORD64 displacement = 0;
    if (sym_from_addr(GetCurrentProcess(), address, &displacement, symbol)) {
      name.assign(symbol->Name, symbol->NameLen);
    }
  }
  if (name.empty()) {
    HMODULE module = nullptr;
    char path[MAX_PATH] = {};
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCSTR>(address), &module) &&
        GetModuleFileNameA(module, path, MAX_PATH)) {
      std::string module_name(path);
      size_t slash = module_name.find_last_of("\\/");
      if (slash != std::string::npos) {
        module_name = module_name.substr(slash + 1);
      }
      name = fmt::format("{}+{:#x}", module_name,
                         address - reinterpret_cast<uint64_t>(module));
    } else {
      name = fmt::format("{:#x}", address);
    }
  }
  cache.emplace(address, name);
  return name;
}
std::string Symbolize(uint64_t address) {
  static std::mutex symbol_mutex;
  std::lock_guard<std::mutex> lock(symbol_mutex);
  return SymbolizeUnlocked(address);
}
#else
void SamplerMain() {}
std::string Symbolize(uint64_t address) { return fmt::format("{:#x}", address); }
#endif

std::string TopFunctions(const std::unordered_map<uint64_t, uint32_t>& counts, uint64_t total) {
  std::unordered_map<std::string, uint32_t> by_name;
  for (const auto& [address, count] : counts) {
    by_name[Symbolize(address)] += count;
  }
  std::vector<std::pair<std::string, uint32_t>> sorted(by_name.begin(), by_name.end());
  std::sort(sorted.begin(), sorted.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
  std::string text;
  for (size_t i = 0; i < sorted.size() && i < 8; ++i) {
    text += fmt::format("{}{} {:.0f}%", i ? ", " : "", sorted[i].first,
                        total ? 100.0 * sorted[i].second / double(total) : 0.0);
  }
  return text;
}

struct Row {
  uint32_t host_tid = 0;
  bool guest = false;
  std::string label;
  double cpu_pct = 0.0;
  Snapshot delta;
};

void ReporterMain() {
  std::unordered_map<const ThreadStats*, Snapshot> previous;
  std::unordered_map<uint32_t, uint64_t> previous_cpu;
  auto last_time = std::chrono::steady_clock::now();
  for (;;) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    auto now = std::chrono::steady_clock::now();
    double elapsed_ms =
        std::chrono::duration<double, std::milli>(now - last_time).count();
    last_time = now;

    if (!Diag()) {
      // faster_loading only needs the monitor thread, not the once-a-second report.
      if (ActiveNow()) {
        bool expected_monitor = false;
        if (g_monitor_started.compare_exchange_strong(expected_monitor, true)) {
          std::thread(WaitMonitorMain).detach();
        }
      }
      continue;
    }
    std::unordered_map<uint32_t, HostThreadCpu> host = SampleHostThreads();
    std::unordered_map<uint32_t, double> cpu_pct;
    for (const auto& [tid, cpu] : host) {
      auto it = previous_cpu.find(tid);
      if (it != previous_cpu.end() && cpu.cpu_100ns >= it->second && elapsed_ms > 0.0) {
        cpu_pct[tid] = double(cpu.cpu_100ns - it->second) / 10000.0 / elapsed_ms * 100.0;
      }
    }
    previous_cpu.clear();
    for (const auto& [tid, cpu] : host) {
      previous_cpu[tid] = cpu.cpu_100ns;
    }

    std::vector<ThreadStats*> threads;
    {
      std::lock_guard<std::mutex> lock(g_threads_mutex);
      threads = g_threads;
    }

    std::vector<Row> rows;
    Snapshot total;
    std::unordered_map<uint32_t, bool> guest_host_tids;
    for (const ThreadStats* stats : threads) {
      Snapshot current = Take(*stats);
      Snapshot& before = previous[stats];
      Row row;
      row.delta.read_calls = current.read_calls - before.read_calls;
      row.delta.read_bytes = current.read_bytes - before.read_bytes;
      row.delta.read_us = current.read_us - before.read_us;
      row.delta.delay_calls = current.delay_calls - before.delay_calls;
      row.delta.delay_us = current.delay_us - before.delay_us;
      row.delta.delay_requested_us = current.delay_requested_us - before.delay_requested_us;
      row.delta.wait_calls = current.wait_calls - before.wait_calls;
      row.delta.wait_us = current.wait_us - before.wait_us;
      row.delta.yields = current.yields - before.yields;
      before = current;
      total.read_calls += row.delta.read_calls;
      total.read_bytes += row.delta.read_bytes;
      total.read_us += row.delta.read_us;
      guest_host_tids[stats->host_tid] = true;
      auto cpu_it = cpu_pct.find(stats->host_tid);
      row.cpu_pct = cpu_it != cpu_pct.end() ? cpu_it->second : 0.0;
      bool active = row.cpu_pct >= 2.0 || row.delta.read_calls || row.delta.delay_us >= 20000 ||
                    row.delta.wait_us >= 20000 || row.delta.yields >= 1000;
      if (!active) {
        continue;
      }
      row.host_tid = stats->host_tid;
      row.guest = stats->guest_tid != 0;
      row.label = stats->guest_tid ? fmt::format("guest {:X}", stats->guest_tid)
                                   : fmt::format("host {}", stats->host_tid);
      auto host_it = host.find(stats->host_tid);
      if (host_it != host.end() && !host_it->second.name.empty()) {
        row.label += " '" + host_it->second.name + "'";
      }
      rows.push_back(std::move(row));
    }
    // Busy host threads that never called into the kernel (GPU, audio, ...).
    for (const auto& [tid, pct] : cpu_pct) {
      if (pct < 10.0 || guest_host_tids.count(tid)) {
        continue;
      }
      Row row;
      row.host_tid = tid;
      row.label = fmt::format("host {}", tid);
      auto host_it = host.find(tid);
      if (host_it != host.end() && !host_it->second.name.empty()) {
        row.label += " '" + host_it->second.name + "'";
      }
      row.cpu_pct = pct;
      rows.push_back(std::move(row));
    }

    bool expected_monitor = false;
    if (g_monitor_started.compare_exchange_strong(expected_monitor, true)) {
      std::thread(WaitMonitorMain).detach();
    }
    double guest_cpu_total = 0.0;
    for (const Row& row : rows) {
      if (row.guest) {
        guest_cpu_total += row.cpu_pct;
      }
    }
    uint64_t frames = g_swap_count.exchange(0);
    uint64_t max_gap_us = g_max_swap_gap_us.exchange(0);
    std::sort(rows.begin(), rows.end(),
              [](const Row& a, const Row& b) { return a.cpu_pct > b.cpu_pct; });
    if (rows.size() > 10) {
      rows.resize(10);
    }
    std::string line = fmt::format(
        "[load-prof] {:.0f} ms: frames {} (longest gap {:.0f} ms), game threads cpu {:.0f}%, "
        "reads {} ({:.1f} MB, {:.0f} ms in reads)",
        elapsed_ms, frames, double(max_gap_us) / 1000.0, guest_cpu_total, total.read_calls,
        double(total.read_bytes) / (1024.0 * 1024.0), double(total.read_us) / 1000.0);
    for (const Row& row : rows) {
      line += fmt::format(" | {} cpu {:.0f}%", row.label, row.cpu_pct);
      if (row.delta.read_calls) {
        line += fmt::format(" read {}x {:.1f}MB {:.0f}ms", row.delta.read_calls,
                            double(row.delta.read_bytes) / (1024.0 * 1024.0),
                            double(row.delta.read_us) / 1000.0);
      }
      if (row.delta.delay_calls) {
        line += fmt::format(" sleep {}x {:.0f}ms (asked {:.0f}ms)", row.delta.delay_calls,
                            double(row.delta.delay_us) / 1000.0,
                            double(row.delta.delay_requested_us) / 1000.0);
      }
      if (row.delta.wait_calls) {
        line += fmt::format(" wait {}x {:.0f}ms", row.delta.wait_calls,
                            double(row.delta.wait_us) / 1000.0);
      }
      if (row.delta.yields) {
        line += fmt::format(" yield {}x", row.delta.yields);
      }
    }
    REXKRNL_WARN("{}", line);

    if (!REXCVAR_GET(diag_load_profile_sample)) {
      continue;
    }
    bool expected_sampler = false;
    if (g_sampler_started.compare_exchange_strong(expected_sampler, true)) {
      std::thread(SamplerMain).detach();
    }
    std::unordered_map<uint32_t, SampleBuckets> buckets;
    std::unordered_map<uint32_t, std::string> labels;
    {
      std::lock_guard<std::mutex> lock(g_sample_mutex);
      buckets.swap(g_sample_buckets);
      // Next second: the (up to) 4 busiest guest threads.
      g_sample_targets.clear();
      for (const Row& row : rows) {
        bool gpu_thread = !row.guest && (row.label.find("GPU Commands") != std::string::npos ||
                                          row.label.find("Command Recorder") != std::string::npos);
        if ((row.guest || gpu_thread) && row.cpu_pct >= (gpu_thread ? 15.0 : 30.0) && g_sample_targets.size() < 6) {
          g_sample_targets.push_back(row.host_tid);
        }
      }
    }
    for (const Row& row : rows) {
      labels[row.host_tid] = row.label;
    }
    if (!REXCVAR_GET(diag_load_profile_dump).empty()) {
      if (FILE* dump = std::fopen(REXCVAR_GET(diag_load_profile_dump).c_str(), "a")) {
        for (const auto& [tid, sampled] : buckets) {
          auto label_it = labels.find(tid);
          std::fprintf(dump, "# %u %s\n", tid,
                       label_it != labels.end() ? label_it->second.c_str() : "?");
          for (const auto& [key, count] : sampled.host_chain) {
            std::fprintf(dump, "%u %u", tid, count);
            for (size_t offset = 0; offset + sizeof(uint64_t) <= key.size();
                 offset += sizeof(uint64_t)) {
              uint64_t address = 0;
              std::memcpy(&address, key.data() + offset, sizeof(address));
              int module = DumpModuleOf(address);
              std::fprintf(dump, " %d:%llx", module,
                           module < 0 ? 0ull
                                      : (unsigned long long)(address - g_module_begin[module]));
            }
            std::fprintf(dump, "\n");
          }
        }
        std::fclose(dump);
      }
    }
    for (const auto& [tid, sampled] : buckets) {
      if (!sampled.samples) {
        continue;
      }
      auto label_it = labels.find(tid);
      REXKRNL_WARN("[load-prof-hot] {} ({} samples) | at: {} | in game code: {}",
                   label_it != labels.end() ? label_it->second : fmt::format("host {}", tid),
                   sampled.samples, TopFunctions(sampled.leaf, sampled.samples),
                   TopFunctions(sampled.game_frame, sampled.samples));
      // Most common call chains (innermost first; "a < b" = a was called by b).
      std::unordered_map<std::string, uint32_t> by_text;
      for (const auto& [key, count] : sampled.chain) {
        std::string text;
        std::string previous_name;
        size_t shown = 0;
        for (size_t offset = 0; offset + sizeof(uint64_t) <= key.size() && shown < 6;
             offset += sizeof(uint64_t)) {
          uint64_t address = 0;
          std::memcpy(&address, key.data() + offset, sizeof(address));
          std::string name = Symbolize(address);
          if (name == previous_name) {
            continue;
          }
          text += (shown ? " < " : "") + name;
          previous_name = name;
          ++shown;
        }
        by_text[text] += count;
      }
      std::vector<std::pair<std::string, uint32_t>> chains(by_text.begin(), by_text.end());
      std::sort(chains.begin(), chains.end(),
                [](const auto& a, const auto& b) { return a.second > b.second; });
      std::string chain_line;
      for (size_t i = 0; i < chains.size() && i < 3; ++i) {
        chain_line += fmt::format("{}{:.0f}%: {}", i ? " ;; " : "",
                                  100.0 * chains[i].second / double(sampled.samples),
                                  chains[i].first);
      }
      if (!chain_line.empty()) {
        REXKRNL_WARN("[load-prof-chain] {} | {}",
                     label_it != labels.end() ? label_it->second : fmt::format("host {}", tid),
                     chain_line);
      }
    }
  }
}


// ---------------------------------------------------------------------------
uint64_t TotalReadBytes() {
  uint64_t total = 0;
  std::lock_guard<std::mutex> lock(g_threads_mutex);
  for (const ThreadStats* stats : g_threads) {
    total += stats->read_bytes.load(std::memory_order_relaxed);
  }
  return total;
}

std::string WallClock() {
  auto now = std::chrono::system_clock::now();
  std::time_t now_time = std::chrono::system_clock::to_time_t(now);
  std::tm local_tm{};
#ifdef _WIN32
  localtime_s(&local_tm, &now_time);
#else
  localtime_r(&now_time, &local_tm);
#endif
  uint32_t ms = uint32_t(
      std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() %
      1000);
  return fmt::format("{:02}:{:02}:{:02}.{:03}", local_tm.tm_hour, local_tm.tm_min,
                     local_tm.tm_sec, ms);
}

// ---------------------------------------------------------------------------
// Wait monitor and stall detector (diag_load_profile). Every 25 ms it looks at
// each game thread's wait in progress. A wait that passes diag_wait_min_ms has
// its thread suspended once to read the stack, so the game functions that
// made the wait are known; finished long waits are summed per second into
// [wait] lines. A gap of diag_stall_ms between presented frames is a stall:
// [stall] START/END lines, with a [stall-thread] line per game thread (every
// second while it lasts) saying what each one is waiting on.

std::string WallClockAgo(uint64_t us_ago) {
  auto at = std::chrono::system_clock::now() - std::chrono::microseconds(us_ago);
  std::time_t at_time = std::chrono::system_clock::to_time_t(at);
  std::tm local_tm{};
#ifdef _WIN32
  localtime_s(&local_tm, &at_time);
#else
  localtime_r(&at_time, &local_tm);
#endif
  uint32_t ms = uint32_t(
      std::chrono::duration_cast<std::chrono::milliseconds>(at.time_since_epoch()).count() %
      1000);
  return fmt::format("{:02}:{:02}:{:02}.{:03}", local_tm.tm_hour, local_tm.tm_min,
                     local_tm.tm_sec, ms);
}

const char* WaitKindName(uint32_t kind) {
  switch (kind) {
    case kWaitKeSingle:
      return "KeWaitForSingleObject";
    case kWaitNtSingle:
      return "NtWaitForSingleObjectEx";
    case kWaitKeMultiple:
      return "KeWaitForMultipleObjects";
    case kWaitNtMultiple:
      return "NtWaitForMultipleObjectsEx";
    case kWaitSignalAndWait:
      return "NtSignalAndWaitForSingleObjectEx";
    case kWaitSleep:
      return "KeDelayExecutionThread";
    default:
      return "wait";
  }
}

// Recompiled game functions are named <name>_8XXXXXXX (their Xbox address).
bool IsGameFunctionName(const std::string& name) {
  if (name.size() < 10 || name[name.size() - 9] != '_' || name[name.size() - 8] != '8') {
    return false;
  }
  for (size_t i = name.size() - 7; i < name.size(); ++i) {
    char c = name[i];
    if (!((c >= '0' && c <= '9') || (c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f'))) {
      return false;
    }
  }
  return true;
}

#ifdef _WIN32
// Game functions on a thread's stack, innermost first ("a < b < c" = a was
// called by b). Found by scanning the stack for return addresses, so an odd
// stale entry is possible.
std::string GameStack(uint32_t host_tid, std::unordered_map<uint32_t, HANDLE>& handles) {
  if (!g_exe_begin) {
    InitExeRange();
  }
  auto handle_it = handles.find(host_tid);
  if (handle_it == handles.end()) {
    HANDLE opened = OpenThread(
        THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, host_tid);
    handle_it = handles.emplace(host_tid, opened).first;
  }
  HANDLE thread = handle_it->second;
  if (!thread) {
    return "(no access)";
  }
  static uint64_t stack[4096];
  SIZE_T stack_bytes = 0;
  if (SuspendThread(thread) == DWORD(-1)) {
    return "(suspend failed)";
  }
  // No allocation or locking while the thread is suspended.
  CONTEXT context = {};
  context.ContextFlags = CONTEXT_CONTROL;
  bool have_context = GetThreadContext(thread, &context) != FALSE;
  if (have_context) {
    for (SIZE_T size = sizeof(stack); size >= 256; size /= 4) {
      if (ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(context.Rsp),
                            stack, size, &stack_bytes)) {
        break;
      }
      stack_bytes = 0;
    }
  }
  ResumeThread(thread);
  if (!have_context) {
    return "(no context)";
  }
  std::vector<std::string> frames;
  auto consider = [&](uint64_t address) {
    if (address < g_exe_begin || address >= g_exe_end || frames.size() >= 8) {
      return;
    }
    std::string name = Symbolize(address);
    if (IsGameFunctionName(name) && (frames.empty() || frames.back() != name)) {
      frames.push_back(std::move(name));
    }
  };
  consider(context.Rip);
  for (size_t i = 0; i < stack_bytes / sizeof(uint64_t) && frames.size() < 8; ++i) {
    consider(stack[i]);
  }
  if (frames.empty()) {
    return "(no game functions on stack)";
  }
  std::string text;
  for (size_t i = 0; i < frames.size(); ++i) {
    text += (i ? " < " : "") + frames[i];
  }
  return text;
}

std::string ThreadName(uint32_t host_tid) {
  static std::unordered_map<uint32_t, std::string> names;
  auto it = names.find(host_tid);
  if (it != names.end()) {
    return it->second;
  }
  using GetThreadDescriptionFn = HRESULT(WINAPI*)(HANDLE, PWSTR*);
  static auto get_thread_description = reinterpret_cast<GetThreadDescriptionFn>(
      GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "GetThreadDescription"));
  std::string name;
  HANDLE thread = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, host_tid);
  if (thread && get_thread_description) {
    PWSTR description = nullptr;
    if (SUCCEEDED(get_thread_description(thread, &description)) && description) {
      for (const wchar_t* c = description; *c && name.size() < 40; ++c) {
        name.push_back(*c < 128 ? char(*c) : '?');
      }
      LocalFree(description);
    }
  }
  if (thread) {
    CloseHandle(thread);
  }
  names.emplace(host_tid, name);
  return name;
}
#else
std::string GameStack(uint32_t, std::unordered_map<uint32_t, void*>&) { return "(n/a)"; }
std::string ThreadName(uint32_t) { return ""; }
#endif

void WaitMonitorMain() {
#ifdef _WIN32
  std::unordered_map<uint32_t, HANDLE> handles;
#else
  std::unordered_map<uint32_t, void*> handles;
#endif
  struct Record {
    uint64_t seq = 0;
    uint64_t start_us = 0;
    uint32_t kind = 0, object = 0, count = 0;
    const char* what = nullptr;
    std::string stack;
    bool open = false;
  };
  struct Sum {
    uint32_t count = 0;
    uint64_t total_us = 0, max_us = 0;
  };
  std::unordered_map<const ThreadStats*, Record> records;
  // Long waits that finished this second: key = thread | kind | object type | stack.
  std::unordered_map<std::string, Sum> finished;
  auto next_report = std::chrono::steady_clock::now() + std::chrono::seconds(1);
  bool in_stall = false;
  uint64_t stall_last_swap = 0;
  uint32_t stall_count = 0;
  bool boosted = false;
  auto boost_since = std::chrono::steady_clock::now();
  uint64_t boost_guest_start = 0;
  // idle_boost: the boost is owned by the idle detector rather than a stall.
  bool idle_owned = false;
  const ThreadStats* game_thread = nullptr;
  uint64_t idle_prev_delay_us = 0, idle_prev_reads = 0, idle_prev_us = 0;
  uint32_t idle_hits = 0;
  auto end_boost = [&](const char* reason) {
    if (!boosted) {
      return;
    }
    (void)rex::chrono::Clock::QueryGuestTickCount();
    rex::chrono::Clock::set_guest_time_scalar(1.0);
    boosted = false;
    double real_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - boost_since)
                        .count();
    double guest_s = double(rex::chrono::Clock::QueryGuestTickCount() - boost_guest_start) /
                     double(rex::chrono::Clock::guest_tick_frequency());
    REXKRNL_WARN("[{}] OFF ({}) after {:.2f} s real, {:.2f} s of game time",
                 idle_owned ? "idle-boost" : "stall-boost", reason, real_s, guest_s);
    idle_owned = false;
  };
  auto next_stall_dump = std::chrono::steady_clock::now();

  auto label = [](const ThreadStats* stats) {
    std::string text = stats->guest_tid ? fmt::format("guest {:X}", stats->guest_tid)
                                        : fmt::format("host {}", stats->host_tid);
    std::string name = ThreadName(stats->host_tid);
    if (!name.empty()) {
      text += " '" + name + "'";
    }
    return text;
  };
  auto describe = [](const Record& r) {
    std::string text = fmt::format("{} {}", WaitKindName(r.kind), r.what ? r.what : "?");
    if (r.kind == kWaitSleep) {
      text += fmt::format(" (asked {} ms)", r.object);
    } else {
      text += fmt::format(" {:#x}", r.object);
      if (r.count > 1) {
        text += fmt::format(" (+{} more objects)", r.count - 1);
      }
    }
    return text;
  };

  for (;;) {
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
    if (!ActiveNow()) {
      records.clear();
      finished.clear();
      in_stall = false;
      end_boost("faster_loading and diag_load_profile off");
      continue;
    }
    if (boosted && !BoostOn()) {
      end_boost("faster_loading off");
    }
    const bool diag = Diag();
    const uint64_t now = NowUs();
    const uint64_t min_us = uint64_t(std::max(REXCVAR_GET(diag_wait_min_ms), 1)) * 1000;
    std::vector<ThreadStats*> threads;
    {
      std::lock_guard<std::mutex> lock(g_threads_mutex);
      threads = g_threads;
    }

    for (ThreadStats* stats : threads) {
      if (!diag) {
        break;  // Long-wait logging is only for diag_load_profile.
      }
      Record& record = records[stats];
      uint64_t seq = stats->wait_seq.load(std::memory_order_acquire);
      uint64_t start = stats->wait_start_us.load(std::memory_order_acquire);
      // Close the previous long wait once it has ended.
      if (record.open && (start == 0 || seq != record.seq)) {
        uint64_t end_seq = stats->wait_end_seq.load(std::memory_order_acquire);
        uint64_t end_us = stats->wait_end_us.load(std::memory_order_acquire);
        uint64_t duration =
            end_seq == record.seq && end_us > record.start_us ? end_us - record.start_us
                                                               : now - record.start_us;
        std::string key = label(stats) + " | " + describe(record) + " | " + record.stack;
        Sum& sum = finished[key];
        ++sum.count;
        sum.total_us += duration;
        sum.max_us = std::max(sum.max_us, duration);
        record.open = false;
      }
      // A wait in progress that just passed the threshold: read its stack once.
      if (start != 0 && now > start && now - start >= min_us && !(record.open && record.seq == seq)) {
        record.seq = seq;
        record.start_us = start;
        record.kind = stats->wait_kind.load(std::memory_order_relaxed);
        record.object = stats->wait_object.load(std::memory_order_relaxed);
        record.count = stats->wait_count.load(std::memory_order_relaxed);
        record.what = stats->wait_what.load(std::memory_order_relaxed);
        record.stack = stats->host_tid ? GameStack(stats->host_tid, handles) : "(no thread)";
        // The wait may have ended while the stack was read; it still counts.
        record.open = true;
      }
    }

    // Stall detector.
    const uint64_t last_swap = g_last_swap_us.load(std::memory_order_acquire);
    const uint64_t stall_us = uint64_t(std::max(REXCVAR_GET(diag_stall_ms), 1)) * 1000;
    auto steady_now = std::chrono::steady_clock::now();
    if (!in_stall && last_swap && now > last_swap && now - last_swap >= stall_us) {
      in_stall = true;
      stall_last_swap = last_swap;
      ++stall_count;
      if (diag) {
        REXKRNL_WARN("[stall] #{} START: no frame presented since {} ({} ms ago)", stall_count,
                     WallClockAgo(now - last_swap), (now - last_swap) / 1000);
      }
      next_stall_dump = steady_now;
      double boost = std::min(REXCVAR_GET(stall_boost), 16.0);
      if (boosted && idle_owned && boost > 1.0) {
        // An idle boost is running; the stall keeps it until the stall ends.
        idle_owned = false;
        REXKRNL_WARN("[stall-boost] stall #{} takes over the idle boost", stall_count);
      }
      if (boost > 1.0 && BoostOn() && !boosted &&
          now < g_load_session_until_us.load(std::memory_order_relaxed)) {
        boost_guest_start = rex::chrono::Clock::QueryGuestTickCount();
        rex::chrono::Clock::set_guest_time_scalar(boost);
        boost_since = steady_now;
        boosted = true;
        REXKRNL_WARN("[stall-boost] ON x{:.1f} for stall #{}", boost, stall_count);
      }
    } else if (in_stall && last_swap != stall_last_swap) {
      in_stall = false;
      if (diag) {
        REXKRNL_WARN("[stall] #{} END at {}: {:.0f} ms without a frame", stall_count,
                     WallClockAgo(now - last_swap), double(last_swap - stall_last_swap) / 1000.0);
      }
      end_boost("stall ended");
    }
    // Idle detector (idle_boost), evaluated every 250 ms.
    if (!game_thread) {
      for (ThreadStats* stats : threads) {
        if (stats->guest_tid && stats->host_tid &&
            ThreadName(stats->host_tid).find("GameThread") != std::string::npos) {
          game_thread = stats;
          break;
        }
      }
    }
    if (game_thread && now - idle_prev_us >= 250000) {
      uint64_t delay_us = game_thread->delay_us.load(std::memory_order_relaxed);
      uint64_t reads = 0;
      for (ThreadStats* stats : threads) {
        reads += stats->read_bytes.load(std::memory_order_relaxed);
      }
      if (idle_prev_us) {
        double window_us = double(now - idle_prev_us);
        // delay_us counts sleeps that already ended; good enough over 250 ms.
        // Sleeps are counted when they end, so a window can hold more than 100%.
        double sleep_frac = std::min(1.0, double(delay_us - idle_prev_delay_us) / window_us);
        double read_mb_s = double(reads - idle_prev_reads) / (1024.0 * 1024.0) / (window_us / 1e6);
        uint64_t load_start = g_load_start_us.load(std::memory_order_relaxed);
        double idle_boost = std::min(REXCVAR_GET(idle_boost), 16.0);
        bool in_window =
            load_start && now >= load_start &&
            now - load_start < uint64_t(std::max(REXCVAR_GET(idle_boost_window_s), 1)) * 1000000ull;
        bool idle = sleep_frac >= 0.6 && read_mb_s < 8.0;
        if (idle_boost > 1.0 && BoostOn() && in_window && idle && !in_stall) {
          // Two windows in a row (0.5 s) before switching on.
          if (++idle_hits >= 2 && !boosted) {
            boost_guest_start = rex::chrono::Clock::QueryGuestTickCount();
            rex::chrono::Clock::set_guest_time_scalar(idle_boost);
            boost_since = steady_now;
            boosted = true;
            idle_owned = true;
            REXKRNL_WARN("[idle-boost] ON x{:.1f} at +{:.1f} s (GameThread asleep {:.0f}%, "
                         "reads {:.1f} MB/s)",
                         idle_boost, double(now - load_start) / 1e6, sleep_frac * 100.0, read_mb_s);
          }
        } else {
          idle_hits = 0;
          if (boosted && idle_owned) {
            end_boost(!in_window ? "window over"
                      : read_mb_s >= 8.0 ? "disk reads started"
                      : in_stall ? "stall"
                                 : "GameThread busy");
          }
        }
      }
      idle_prev_us = now;
      idle_prev_delay_us = delay_us;
      idle_prev_reads = reads;
    }
    if (boosted && steady_now - boost_since >
                       std::chrono::seconds(std::max(REXCVAR_GET(stall_boost_max_s), 1))) {
      end_boost("time limit");
    }
    if (diag && in_stall && steady_now >= next_stall_dump) {
      next_stall_dump = steady_now + std::chrono::seconds(1);
      uint64_t stalled_ms = (now - stall_last_swap) / 1000;
      for (ThreadStats* stats : threads) {
        if (!stats->guest_tid) {
          continue;
        }
        uint64_t start = stats->wait_start_us.load(std::memory_order_acquire);
        Record& record = records[stats];
        std::string state;
        if (start == 0) {
          state = "not waiting (running, spinning or in a short wait) | " +
                  (stats->host_tid ? GameStack(stats->host_tid, handles) : std::string("?"));
        } else {
          Record current;
          current.kind = stats->wait_kind.load(std::memory_order_relaxed);
          current.object = stats->wait_object.load(std::memory_order_relaxed);
          current.count = stats->wait_count.load(std::memory_order_relaxed);
          current.what = stats->wait_what.load(std::memory_order_relaxed);
          std::string stack = record.open && record.start_us == start
                                  ? record.stack
                                  : (stats->host_tid ? GameStack(stats->host_tid, handles)
                                                     : std::string("?"));
          state = fmt::format("waiting {} ms in {} | {}", (now > start ? now - start : 0) / 1000,
                              describe(current), stack);
        }
        REXKRNL_WARN("[stall-thread] #{} +{} ms {} {}", stall_count, stalled_ms, label(stats),
                     state);
      }
    }

    if (diag && steady_now >= next_report) {
      next_report = steady_now + std::chrono::seconds(1);
      std::vector<std::pair<std::string, Sum>> rows(finished.begin(), finished.end());
      finished.clear();
      std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
        return a.second.total_us > b.second.total_us;
      });
      for (size_t i = 0; i < rows.size() && i < 8; ++i) {
        const Sum& sum = rows[i].second;
        REXKRNL_WARN("[wait] {}x total {:.0f} ms (longest {:.0f} ms): {}", sum.count,
                     double(sum.total_us) / 1000.0, double(sum.max_us) / 1000.0, rows[i].first);
      }
      if (rows.size() > 8) {
        REXKRNL_WARN("[wait] ... {} more kinds of long wait this second", rows.size() - 8);
      }
    }
  }
}

// Watches the read rate to find loads: a load starts when load_timer_start_mb are
// read within one second and ends after load_timer_quiet_s seconds without reads.
// While a load runs it keeps the "load session" open, which is what lets the
// boosts act (they never run outside a load). Logs [load-timer] lines.
void ControlMain() {
  // Read-rate window: bytes seen at each 250 ms step, last 4 s.
  std::vector<uint64_t> read_history;
  std::vector<std::chrono::steady_clock::time_point> history_times;
  auto last_sample = std::chrono::steady_clock::now();

  bool timing = false;
  uint32_t load_count = 0;
  std::chrono::steady_clock::time_point load_start, last_read, last_progress;
  std::string load_start_wall, last_read_wall;
  uint64_t load_start_bytes = 0, load_last_bytes = 0;

  for (;;) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const bool timer_on = REXCVAR_GET(load_timer) || BoostOn() || Diag();
    if (!timer_on) {
      read_history.clear();
      history_times.clear();
      timing = false;
      continue;
    }
    auto now = std::chrono::steady_clock::now();
    if (now - last_sample >= std::chrono::milliseconds(250)) {
      last_sample = now;
      uint64_t total_bytes = TotalReadBytes();
      read_history.push_back(total_bytes);
      history_times.push_back(now);
      if (read_history.size() > 17) {
        read_history.erase(read_history.begin());
        history_times.erase(history_times.begin());
      }
      size_t n = read_history.size();
      // Bytes read in the last second.
      uint64_t last_1s = n >= 5 ? read_history[n - 1] - read_history[n - 5] : 0;
      bool read_now = n >= 2 && read_history[n - 1] - read_history[n - 2] >= 64 * 1024;
      uint64_t start_bytes = uint64_t(std::max(REXCVAR_GET(load_timer_start_mb), 1)) * 1024 * 1024;
      double quiet_s = double(std::max(REXCVAR_GET(load_timer_quiet_s), 1));
      const bool verbose = REXCVAR_GET(load_timer) || Diag();

      if (!timing && last_1s >= start_bytes) {
        // Begin at the oldest sample of the last second that saw reads.
        size_t first = n >= 5 ? n - 5 : 0;
        while (first + 1 < n && read_history[first + 1] == read_history[first]) {
          ++first;
        }
        timing = true;
        ++load_count;
        load_start = history_times[first];
        load_start_wall = WallClock();
        last_read = now;
        last_read_wall = load_start_wall;
        last_progress = now;
        load_start_bytes = read_history[first];
        load_last_bytes = total_bytes;
        g_load_start_us.store(NowUs(), std::memory_order_relaxed);
        REXKRNL_WARN("[load-timer] #{} START at {} ({:.1f} MB read in 1 s){}", load_count,
                     load_start_wall, double(last_1s) / (1024.0 * 1024.0),
                     BoostOn() ? fmt::format(" faster_loading: idle_boost={} stall_boost={}",
                                             REXCVAR_GET(idle_boost), REXCVAR_GET(stall_boost))
                               : std::string());
      } else if (timing) {
        if (read_now) {
          last_read = now;
          last_read_wall = WallClock();
          load_last_bytes = total_bytes;
        }
        if (verbose && std::chrono::duration<double>(now - last_progress).count() >= 5.0) {
          last_progress = now;
          REXKRNL_WARN("[load-timer] #{} ... {:.1f} s so far, {:.0f} MB read", load_count,
                       std::chrono::duration<double>(now - load_start).count(),
                       double(total_bytes - load_start_bytes) / (1024.0 * 1024.0));
        }
        if (std::chrono::duration<double>(now - last_read).count() >= quiet_s) {
          timing = false;
          REXKRNL_WARN("[load-timer] #{} END: {:.2f} s ({} to {}), {:.0f} MB read", load_count,
                       std::chrono::duration<double>(last_read - load_start).count(),
                       load_start_wall, last_read_wall,
                       double(load_last_bytes - load_start_bytes) / (1024.0 * 1024.0));
        }
      }
    }
    if (timing) {
      // The load session stays open for 15 s after the last check.
      g_load_session_until_us.store(NowUs() + 15000000, std::memory_order_relaxed);
    }
  }
}
}  // namespace

void StartControl() {
  if (!(ActiveNow() || REXCVAR_GET(load_timer))) {
    return;
  }
  bool expected = false;
  if (!g_control_started.load(std::memory_order_relaxed) &&
      g_control_started.compare_exchange_strong(expected, true)) {
    std::thread(ControlMain).detach();
  }
}

bool ReadsTracked() {
  StartControl();
  return ActiveNow() || REXCVAR_GET(load_timer);
}

bool Active() {
  StartControl();
  return ActiveNow();
}

bool Enabled() {
  StartControl();
  return Diag();
}

uint64_t NowUs() {
  return uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::steady_clock::now().time_since_epoch())
                      .count());
}

void AddRead(uint64_t bytes, uint64_t elapsed_us) {
  ThreadStats* stats = Stats();
  stats->read_calls.fetch_add(1, std::memory_order_relaxed);
  stats->read_bytes.fetch_add(bytes, std::memory_order_relaxed);
  stats->read_us.fetch_add(elapsed_us, std::memory_order_relaxed);
}

void AddDelay(uint64_t elapsed_us, int64_t requested_interval) {
  ThreadStats* stats = Stats();
  stats->delay_calls.fetch_add(1, std::memory_order_relaxed);
  stats->delay_us.fetch_add(elapsed_us, std::memory_order_relaxed);
  if (requested_interval < 0) {
    stats->delay_requested_us.fetch_add(uint64_t(-requested_interval) / 10,
                                        std::memory_order_relaxed);
  }
}

void AddWait(uint64_t elapsed_us) {
  ThreadStats* stats = Stats();
  stats->wait_calls.fetch_add(1, std::memory_order_relaxed);
  stats->wait_us.fetch_add(elapsed_us, std::memory_order_relaxed);
}

void AddYield() { Stats()->yields.fetch_add(1, std::memory_order_relaxed); }

void BeginWait(uint32_t kind, uint32_t object, const char* what, uint32_t count) {
  ThreadStats* stats = Stats();
  stats->wait_kind.store(kind, std::memory_order_relaxed);
  stats->wait_object.store(object, std::memory_order_relaxed);
  stats->wait_count.store(count, std::memory_order_relaxed);
  stats->wait_what.store(what, std::memory_order_relaxed);
  stats->wait_seq.fetch_add(1, std::memory_order_release);
  stats->wait_start_us.store(NowUs(), std::memory_order_release);
}

void EndWait() {
  ThreadStats* stats = Stats();
  stats->wait_end_us.store(NowUs(), std::memory_order_relaxed);
  stats->wait_end_seq.store(stats->wait_seq.load(std::memory_order_relaxed),
                            std::memory_order_release);
  stats->wait_start_us.store(0, std::memory_order_release);
}

void NoteSwap() {
  if (!ActiveNow()) {
    return;
  }
  uint64_t now = NowUs();
  uint64_t previous = g_last_swap_us.exchange(now, std::memory_order_acq_rel);
  g_swap_count.fetch_add(1, std::memory_order_relaxed);
  if (previous && now > previous) {
    uint64_t gap = now - previous;
    uint64_t current = g_max_swap_gap_us.load(std::memory_order_relaxed);
    while (gap > current &&
           !g_max_swap_gap_us.compare_exchange_weak(current, gap, std::memory_order_relaxed)) {
    }
  }
}

}  // namespace rex::kernel::xboxkrnl::load_profile
