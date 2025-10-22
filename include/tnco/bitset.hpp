// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <boost/dynamic_bitset.hpp>
#include <functional>
#include <string>

namespace tnco::bitset {

// Rename pybind11
namespace py = pybind11;

// Use _a literal for py::arg
using namespace py::literals;

template <typename... T>
struct Bitset : boost::dynamic_bitset<T...> {
  using base_type = boost::dynamic_bitset<T...>;

  // Use base_type constructors
  using base_type::base_type;

  operator std::string() const {
    std::string bits;
    for (size_t i_ = 0; i_ < std::size(*this); ++i_) {
      bits += this->test(i_) ? "1" : "0";
    }
    return bits;
  }

  Bitset(base_type bits) : base_type{std::move(bits)} {}

  template <typename Vector,
            std::enable_if_t<
                std::is_integral_v<typename std::decay_t<Vector>::value_type>,
                bool> = true>
  Bitset(const Vector &pos, size_t size) : base_type(size) {
    std::for_each(std::begin(pos), std::end(pos),
                  [this, n = std::size(*this)](auto &&x) -> auto {
                    if (x >= n) {
                      throw std::invalid_argument("'size' is too small.");
                    }
                    this->set(x);
                  });
  }

  Bitset(std::string bits)
      : base_type{std::string{std::make_move_iterator(std::rbegin(bits)),
                              std::make_move_iterator(std::rend(bits))}} {}

  template <typename Callback>
  auto visit(const Callback &callback) const {
    for (size_t p_ = this->find_first(); p_ != Bitset<T...>::npos;
         p_ = this->find_next(p_)) {
      callback(p_);
    }
  }

  auto positions() const {
    std::vector<size_t> pos;
    visit([&pos](auto &&p) -> auto { pos.push_back(p); });
    return pos;
  }

  auto operator&(const Bitset &other) const {
    auto new_{*this};
    new_ &= other;
    return new_;
  }
  auto operator|(const Bitset &other) const {
    auto new_{*this};
    new_ |= other;
    return new_;
  }
  auto operator^(const Bitset &other) const {
    auto new_{*this};
    new_ ^= other;
    return new_;
  }
  auto operator-(const Bitset &other) const {
    auto new_{*this};
    new_ -= other;
    return new_;
  }
  auto operator~() const -> Bitset {
    return ~(*static_cast<const base_type *>(this));
  }

  template <
      typename Stream,
      std::enable_if_t<
          std::is_base_of_v<std::basic_ostream<typename Stream::char_type,
                                               typename Stream::traits_type>,
                            Stream>,
          bool> = true>
  friend auto operator<<(Stream &os, const Bitset &bs) -> Stream & {
    os << std::string(bs);
    return os;
  }
};

template <typename... T>
void init(py::module &m, const std::string &name) {
  using self_type = Bitset<T...>;
  py::class_<self_type>(m, name.c_str())
      .def(py::init<>())
      .def(py::init<const std::vector<size_t> &, size_t>(), "positions"_a,
           "size"_a)
      .def(py::init<const std::string &>())
      .def("__and__", &self_type::operator&)
      .def("__or__", &self_type::operator|)
      .def("__xor__", &self_type::operator^)
      .def("__sub__", &self_type::operator-)
      .def("__eq__",
           [](const self_type &self, const self_type &other) -> auto {
             return self == other;
           })
      .def("isdisjoint",
           [](const self_type &self, const self_type &other) -> auto {
             return !self.intersects(other);
           })
      .def("issubset",
           [](const self_type &self, const self_type &other) -> auto {
             return self.is_subset_of(other);
           })
      .def("issuperset",
           [](const self_type &self, const self_type &other) -> auto {
             return other.is_subset_of(self);
           })
      .def("__le__",
           [](const self_type &self, const self_type &other) -> auto {
             return self.is_subset_of(other);
           })
      .def("__ge__",
           [](const self_type &self, const self_type &other) -> auto {
             return other.is_subset_of(self);
           })
      .def("__lt__",
           [](const self_type &self, const self_type &other) -> auto {
             return self.is_proper_subset_of(other);
           })
      .def("__gt__",
           [](const self_type &self, const self_type &other) -> auto {
             return other.is_proper_subset_of(self);
           })
      .def("__invert__", [](const self_type &self) -> auto { return ~self; })
      .def("__len__",
           [](const self_type &self) -> auto { return std::size(self); })
      .def("__getitem__",
           [](const self_type &self, const size_t pos) -> auto {
             if (pos >= std::size(self)) {
               throw std::out_of_range("Index out of range.");
             }
             return self.test(pos);
           })
      .def("__str__",
           [](const self_type &self) -> auto { return std::string(self); })
      .def("__repr__",
           [](const self_type &self) -> auto {
             return "Bitset(" + std::string(self) + ")";
           })
      .def("count", [](const self_type &self) -> auto { return self.count(); })
      .def("visit", &self_type::template visit<std::function<void(size_t)>>)
      .def("positions", &self_type::positions)
      .def(py::pickle(
          [](const self_type &self) -> auto { return std::string(self); },
          [](const std::string &state) -> auto { return self_type{state}; }));
}

}  // namespace tnco::bitset
