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
#include <stack>

#include "callable.hpp"
#include "../types.hpp"
#include "scheduling_policy.hpp"

namespace znn {
namespace v4 {

// Runs tasks according to their priorities. High priority tasks
// will get run before low priority tasks when both are in the wait
// queue.
//
// Thread-compatible.
class dfs_policy : public scheduling_policy {
public:
  dfs_policy(std::size_t concurrency)
    : scheduling_policy(concurrency), locals_(concurrency) {
  }
  ~dfs_policy() override {}

  // Presumes only worker_loop threads call schedule()
  void schedule(int /*priority*/, std::shared_ptr<task> t) override {
    regular_task r{std::move(t), tid_, {}};
    locals_[tid_].push_front(std::move(r));
    auto local = locals_[tid_].begin();
    globals_.push_front(local);
    local->global_list_it_ = globals_.begin();
  }

  void get_next(std::size_t tid, std::shared_ptr<task>* t) override {
    tid_ = tid;

    if (globals_.empty()) return;

    auto reg_task = !locals_[tid_].empty() ?
      locals_[tid_].begin() : globals_.front();
    *t = std::move(reg_task->task_);
    globals_.erase(reg_task->global_list_it_);
    locals_[reg_task->owner_tid_].erase(reg_task);
  }

private:
  struct regular_task;
  typedef list<regular_task> task_list;
  typedef list<task_list::iterator> task_list_iter_list;
  struct regular_task {
    std::shared_ptr<task> task_;
    std::size_t owner_tid_;
    task_list_iter_list::iterator global_list_it_;
  };

  static thread_local std::size_t tid_;

  vector<task_list> locals_;
  task_list_iter_list globals_;
};

thread_local std::size_t znn::v4::dfs_policy::tid_ = 0;

}  // namespace v4
}  // namespace znn
