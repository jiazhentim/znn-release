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

#include <functional>
#include <thread>
#include <fstream>
#include <atomic>
#include <map>
#include <set>
#include <list>
#include <mutex>
#include <memory>
#include <condition_variable>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

#include <zi/utility/singleton.hpp>
#include <zi/time.hpp>

#include "callable.hpp"
#include "scheduling_policy.hpp"
#include "../types.hpp"

namespace znn {
namespace v4 {

  // task_manager accepts generic scheduling policies and lets waiting
  // threads steal tasks as well.
  //
  // Class is thread-safe.
  class task_manager {
  private:
    // We use atomic booleans to transparently mark tombstone tasks,
    // or tasks that are stolen by external threads.
    //
    // Thread-safe.
    class task : public scheduling_policy::task {
    private:
      // Invariants:
      //   callable_ valid iff !has_started_
      //   has_started_ ==> has_finished_
      unique_ptr<callable> callable_;
      std::atomic<bool> has_started_;
      std::atomic<bool> has_finished_;

    public:
      task(unique_ptr<callable> c)
        : callable_(std::move(c))
        , has_started_(false)
        , has_finished_(false) {
      }

      // Should be called externally after caller runs get()
      void notify_finished() {
          has_finished_.store(true);
      }

      bool valid() const override { return !has_started_.load(); }

      // Caller responsible for evaluating the function while this is still in
      // scope. Returns nullptr if not valid.
      unique_ptr<callable> get() {
        if (has_started_.exchange(true)) return nullptr;
        return std::move(callable_);
      }

      std::size_t memsize() override {
        if (has_started_.load()) return 0;
        return callable_->memsize_;
      }

      void finish() {
        // In the unlikely event we caught the thread in the middle of the
        // computation, yield.
        while (!has_finished_.load()) std::this_thread::yield();
        // TODO smarter dependencies (mutex/cond var),
        // or provide api task_manager support for it.
      }
    };

  public:
    typedef std::shared_ptr<task> task_handle;

  private:
    std::size_t spawned_threads_;
    std::size_t concurrency_    ;
    std::size_t idle_threads_   ;

    // Coarse-grained class mutex
    std::mutex mutex_;

    // Waiting condition for the workers, that there is work to do.
    std::condition_variable workers_cv_;

    std::unique_ptr<scheduling_policy> policy_;
    std::vector<std::thread> threads_;

  private:
    void worker_loop(std::size_t tid) {
      {
        std::lock_guard<std::mutex> g(mutex_);
        ZI_ASSERT(spawned_threads_ < concurrency_);
        ++spawned_threads_;
      }

      while (true) {
        std::shared_ptr<scheduling_policy::task> f;
        {
          std::unique_lock<std::mutex> g(mutex_);
          ++idle_threads_;
          workers_cv_.wait(g, [&]() {
              policy_->get_next(tid, &f);
              bool shutting_down = concurrency_ < spawned_threads_;
              return f || shutting_down;
            });
          --idle_threads_;

          if (concurrency_ < spawned_threads_) {
            --spawned_threads_;
            return;
          }
        }

        auto t = static_cast<task*>(f.get());
        auto fn = t->get();
        if (fn) {
            fn->closure_();
            t->notify_finished();
        }
        {
            std::lock_guard<std::mutex> g(mutex_);
            policy_->notify_finished(tid);
            if (idle_threads_ > 0) workers_cv_.notify_one();
        }
      }
    }

  public:
    task_manager(std::unique_ptr<scheduling_policy> policy)
      : spawned_threads_{0}
      , concurrency_{0}
      , idle_threads_{0}
      , policy_(std::move(policy)) {
        concurrency_ = policy_->concurrency();

        for (std::size_t i = 0; i < concurrency_; ++i) {
          threads_.emplace_back(&task_manager::worker_loop, this, i);
        }
      }

    task_manager(const task_manager&) = delete;
    task_manager& operator=(const task_manager&) = delete;

    task_manager(task_manager&& other) = delete;
    task_manager& operator=(task_manager&&) = delete;

    ~task_manager() {
      {
        std::lock_guard<std::mutex> g(mutex_);
        concurrency_ = 0;
        workers_cv_.notify_all();
      }
      for (auto& t : threads_) t.join();
    }

    std::size_t get_concurrency() {
      std::lock_guard<std::mutex> g(mutex_);
      return concurrency_;
    }

    std::size_t idle_threads() {
      std::lock_guard<std::mutex> g(mutex_);
      return idle_threads_;
    }

    std::size_t active_threads() {
      std::lock_guard<std::mutex> g(mutex_);
      return concurrency_ - idle_threads_;
    }

  public:
    // The parameter callable must have a valid closure.
    // Additional callable fields only need to be filled in if the
    // manager's policy requires it.
    //
    // If out parameter pointer-to-shared-pointer is not null, saves
    // the handle there.
    void schedule(int priority, unique_ptr<callable> fn,
                  task_handle* out) {
      auto handle = zmake_shared<task>(std::move(fn));
      if (out) *out = handle;
      std::lock_guard<std::mutex> g(mutex_);
      policy_->schedule(priority, std::move(handle));
      if (idle_threads_ > 0) workers_cv_.notify_one();
    }

    // Steals task if not being executed.
    void require_done(const task_handle& t) {
      if (!t) return;

      if (t->valid()) {
          t->get()->closure_();
          t->notify_finished();
      } else {
          t->finish();
      }
    }
  }; // class task_manager

}  // namespace v4
}  // namespace znn
