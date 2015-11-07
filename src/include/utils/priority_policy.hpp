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
class priority_policy : public scheduling_policy {
public:
  priority_policy(std::size_t concurrency)
    : scheduling_policy(concurrency) {}
  ~priority_policy() override {}

  void schedule(int priority, std::shared_ptr<task> t) override {
    tasks_[priority].push(std::move(t));
  }

  void get_next(std::size_t /*tid*/, std::shared_ptr<task>* t) override {
    do {
      if (tasks_.empty()) {
          *t = nullptr;
          return;
      }
      auto& top_stack = tasks_.rbegin()->second;
      *t = std::move(top_stack.top());
      top_stack.pop();
      if (top_stack.empty()) {
        tasks_.erase(tasks_.rbegin()->first);
      }
    } while (!(*t)->valid());
  }

private:
  map<int, std::stack<std::shared_ptr<task>,
                      vector<std::shared_ptr<task>>>> tasks_;
};

}  // namespace v4
}  // namespace znn
