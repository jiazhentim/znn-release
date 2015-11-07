// TODO copyright
#pragma once

#include <memory>

#include "callable.hpp"
#include "../types.hpp"

namespace znn {
namespace v4 {

// Class is thread-compatible (const-methods may be called
// on multiple threads, but mutations must be externally synchronized)
class scheduling_policy {
public:
  class task {
  public:
    virtual std::size_t memsize() = 0;
    virtual bool valid() const = 0;
  };

  scheduling_policy(std::size_t concurrency)
    : concurrency_(concurrency) {}
  virtual ~scheduling_policy() {}

  std::size_t concurrency() const { return concurrency_; }

  virtual void schedule(int priority, std::shared_ptr<task> t) = 0;

  // Assuming this is thread tid of the threads [0, concurrency)
  // which are running, this returns via 't' the next task for that thread.
  // If no task should be assigned, t set to null.
  // Method only returns valid() tasks. Invalid tasks are discarded.
  // valid() may be called multiple times.
  virtual void get_next(std::size_t tid, std::shared_ptr<task>* t) = 0;

  // Should be called to notify the task this thread was working on was
  // finished.
  virtual void notify_finished(std::size_t /*tid*/) {}

private:
  const std::size_t concurrency_;

};

}  // namespace v4
}  // namespace znn
