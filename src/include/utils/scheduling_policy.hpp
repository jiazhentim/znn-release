// TODO copyright
#pragma once

#include <memory>

#include "callable.hpp"
#include "../types.hpp"

namespace znn {
namespace v4 {

// Class is thread-compatible (TODO google classification link).
class scheduling_policy {
public:
  class task {
    virtual std::size_t memsize() = 0;
  };

  scheduling_policy(std::size_t concurrency)
    : concurrency_(concurrency) {}
  virtual ~scheduling_policy() {}

  std::size_t concurrency() const { return concurrency_; }

  virtual void schedule(int priority, std::shared_ptr<task> t) = 0;

  // Assuming this is thread tid of the threads [0, concurrency)
  // which are running, this returns via 't' the next task for that thread.
  // If no task should be assigned, t remains unchanged.
  virtual void get_next(std::size_t tid, std::shared_ptr<task>* t) = 0;

private:
  const std::size_t concurrency_;

};

}  // namespace v4
}  // namespace znn
