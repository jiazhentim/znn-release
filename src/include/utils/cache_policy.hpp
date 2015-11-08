//
// Copyright (C) 2012-2015  Aleksandar Zlateski <zlateski@mit.edu>
// ---------------------------------------------------------------
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#pragma once

#include <memory>
#include <condition_variable>
#include <mutex>

#include "callable.hpp"
#include "../types.hpp"
#include "scheduling_policy.hpp"

#ifndef ZNN_XEON_PHI
#error "Cache policy for scheduling on implemented for Xeon Phis"
#endif

namespace znn {
namespace v4 {

// Runs tasks according to their priorities. Considers the memory size of a
// computation before running it - if the total L2 cache space cannot handle
// it, then the task is not run.
//
// This class employes a "throttle-parallelism policy" - if no tasks of the top
// priority can fit in the cache, then threads wait.
//
// There are multiple alterations to this approach.
// - If there are low priority tasks at all that can run in the current cache
// space, then run them.
// - An invariant is maintained where there is either enough space to run all
// present tasks, or there is a task being currently run (that will eventually
// finish) which will free space for the cheapest high-priority task. So long
// as this invariant is being maintained, a lower-priority task fitting in cache
// space may be run.
//
// There are additional notions of cache space which may be refined: Consider
// adding the per-cpu L2 local cache that threads have fast access to as another
// element. Even still, we can use thread affinities to pin threads to CPUs
// and solve a generic knapsack problem for scheduling.
//
// Thread-compatible.
class cache_policy : public scheduling_policy {
public:
  cache_policy(std::size_t concurrency)
      : scheduling_policy(concurrency)
      , cache_used_by_thread_(concurrency, 0)
      , largest_running_task_(0) {
      tot_cache_ = 512 * 1024 * 60; // 60 CPUs (1 for OS). 512KB ea.
      avail_cache_ = tot_cache_;
  }
  ~cache_policy() override {}

  void schedule(int priority, std::shared_ptr<task> t) override {
      weighted_priority key{priority, static_cast<int>(t->memsize())};
      tasks_.insert(std::make_pair(key, std::move(t)));
  }

  void get_next(std::size_t tid, std::shared_ptr<task>* t) override {
      if (tasks_.empty()) {
          *t = nullptr;
          return;
      }

      auto it = tasks_.rbegin();
      auto top_prio = it->first.priority_;
      auto lowest_top_prio_memsize = it->first.mem_;
      for (; it != tasks_.rend() && it->first.priority_ == top_prio; ++it) {
          if (!it->second->valid()) {
              tasks_.erase(std::next(it).base());
              continue;
          }

          int sz = it->first.mem_;
          lowest_top_prio_memsize = sz;
          ZI_ASSERT(sz >= 0);
          if (sz > tot_cache_ || sz <= avail_cache_) {
              // In the event we get a very large task, it'll use up the
              // "remainder" of the cache, with some guaranteed misses.
              sz = std::min(avail_cache_, sz);
              *t = std::move(it->second);
              tasks_.erase(std::next(it).base());
              cache_used_by_thread_[tid] = sz;
              largest_running_task_ = std::max(largest_running_task_, sz);
              avail_cache_ -= sz;
              return;
          }
      }

      for (; it != tasks_.rend(); ++it) {
          if (!it->second->valid()) {
              tasks_.erase(std::next(it).base());
              continue;
          }

          int sz = it->first.mem_;
          ZI_ASSERT(sz >= 0);
          if (avail_cache_ - sz + largest_running_task_ >=
              lowest_top_prio_memsize) {
              sz = std::min(avail_cache_, sz);
              *t = std::move(it->second);
              tasks_.erase(std::next(it).base());
              largest_running_task_ = std::max(largest_running_task_, sz);
              avail_cache_ -= sz;
              cache_used_by_thread_[tid] = sz;
              return;
          }
      }

      *t = nullptr;
  }

  void notify_finished(std::size_t tid) {
      auto sz = cache_used_by_thread_[tid];
      avail_cache_ += sz;
      cache_used_by_thread_[tid] = 0;
      if (cache_used_by_thread_[tid] == largest_running_task_) {
          largest_running_task_ =
              *std::min_element(cache_used_by_thread_.begin(),
                                cache_used_by_thread_.end());
      }
  }

private:
    struct weighted_priority {
        int priority_;
        int mem_;
        bool operator<(const weighted_priority& x) const {
            if (priority_ == x.priority_) return mem_ < x.mem_;
            return priority_ > x.priority_;
        }
    };

    multimap<weighted_priority, std::shared_ptr<task>> tasks_;
    int tot_cache_, avail_cache_, largest_running_task_;
    vector<int> cache_used_by_thread_;

    std::mutex mutex_;
    std::condition_variable feasible_tasks_cv_;
};

}  // namespace v4
}  // namespace znn
