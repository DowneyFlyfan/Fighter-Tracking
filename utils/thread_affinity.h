#pragma once

#include <pthread.h>
#include <sched.h>

// Pin the calling thread to a specific CPU core so it doesn't migrate
// between cores (which kills CPU cache + branch predictor warmups for
// tight inner loops like the per-frame tracker). Use core 0 for the
// main thread; the OS keeps decoder/writer threads off it.
inline void bind_thread_to_core(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_t current_thread = pthread_self();
    pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
}
