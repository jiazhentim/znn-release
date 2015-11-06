// TODO: copyright notice

#pragma once

#include <functional>

namespace znn {
namespace v4 {

typedef std::function<void()> closure;

struct callable
{
  callable(closure closure, std::string name, size_t memsize)
    : closure_(closure), name_(name), memsize_(memsize) {}
  closure closure_;
  std::string name_;
  size_t memsize_;
  explicit operator bool() const { return !!closure_; }
};

}  // namespace v4
}  // namespace znn
