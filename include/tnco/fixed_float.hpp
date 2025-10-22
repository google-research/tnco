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

#include <mpfr.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>
#include <regex>
#include <sstream>

namespace tnco::fixed_float {

// Rename pybind11
namespace py = pybind11;

// Use _a literal for py::arg
using namespace py::literals;

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-array-to-pointer-decay)

template <mpfr_prec_t Precision, mpfr_rnd_t Rounding = MPFR_RNDN>
struct FixedFloat {
  static constexpr mpfr_prec_t prec = Precision;
  static constexpr mpfr_rnd_t rounding = MPFR_RNDN;

  explicit operator double() const { return mpfr_get_d(_x, rounding); }

  explicit operator long double() const { return mpfr_get_ld(_x, rounding); }

  FixedFloat() : _x{} { mpfr_init2(_x, prec); }

  FixedFloat(long double x) : FixedFloat() { mpfr_set_ld(_x, x, rounding); }

  FixedFloat(const std::string &x, const int base = 10) : FixedFloat() {
    mpfr_set_str(_x, x.data(), base, rounding);
  }

  FixedFloat(const mpfr_t &x) : FixedFloat() { mpfr_set(_x, x, rounding); }

  FixedFloat(FixedFloat &&x) noexcept : FixedFloat() { mpfr_swap(_x, x._x); }

  FixedFloat(const FixedFloat &x) : FixedFloat() {
    mpfr_set(_x, x._x, rounding);
  }

  ~FixedFloat() { mpfr_clear(_x); }

  [[nodiscard]] auto digits10() const -> size_t {
    // Log10(2) * precision
    return 0.30102999566 * prec;
  }

  auto operator=(FixedFloat &&x) noexcept -> FixedFloat & {
    mpfr_swap(_x, x._x);
    return *this;
  }

  auto operator=(const FixedFloat &x) -> FixedFloat & {
    mpfr_set(_x, x._x, rounding);
    return *this;
  }

  template <typename T>
  auto operator=(const T &x) -> FixedFloat & {
    *this = FixedFloat(x);
    return *this;
  }

  auto operator+() const -> const FixedFloat & { return *this; }

  auto operator-() const -> FixedFloat {
    auto out = FixedFloat();
    mpfr_neg(out._x, _x, rounding);
    return out;
  }

  auto operator+=(const FixedFloat &x) -> FixedFloat & {
    mpfr_add(_x, _x, x._x, rounding);
    return *this;
  }

  auto operator-=(const FixedFloat &x) -> FixedFloat & {
    mpfr_sub(_x, _x, x._x, rounding);
    return *this;
  }

  auto operator*=(const FixedFloat &x) -> FixedFloat & {
    mpfr_mul(_x, _x, x._x, rounding);
    return *this;
  }

  auto operator/=(const FixedFloat &x) -> FixedFloat {
    mpfr_div(_x, _x, x._x, rounding);
    return *this;
  }

  auto operator+(const FixedFloat &x) const -> FixedFloat {
    auto out = FixedFloat();
    mpfr_add(out._x, _x, x._x, rounding);
    return out;
  }

  auto operator-(const FixedFloat &x) const -> FixedFloat {
    auto out = FixedFloat();
    mpfr_sub(out._x, _x, x._x, rounding);
    return out;
  }

  auto operator*(const FixedFloat &x) const -> FixedFloat {
    auto out = FixedFloat();
    mpfr_mul(out._x, _x, x._x, rounding);
    return out;
  }

  auto operator/(const FixedFloat &x) const -> FixedFloat {
    auto out = FixedFloat();
    mpfr_div(out._x, _x, x._x, rounding);
    return out;
  }

  auto operator==(const FixedFloat &x) const -> bool {
    return mpfr_cmp(_x, x._x) == 0;
  }

  auto operator!=(const FixedFloat &x) const -> bool {
    return mpfr_cmp(_x, x._x) != 0;
  }

  auto operator>(const FixedFloat &x) const -> bool {
    return mpfr_cmp(_x, x._x) > 0;
  }

  auto operator>=(const FixedFloat &x) const -> bool {
    return mpfr_cmp(_x, x._x) >= 0;
  }

  auto operator<(const FixedFloat &x) const -> bool {
    return mpfr_cmp(_x, x._x) < 0;
  }

  auto operator<=(const FixedFloat &x) const -> bool {
    return mpfr_cmp(_x, x._x) <= 0;
  }

  template <typename T>
  friend auto operator+(const T &x, const FixedFloat &y) -> FixedFloat {
    return FixedFloat(x) + y;
  }

  template <typename T>
  friend auto operator-(const T &x, const FixedFloat &y) -> FixedFloat {
    return FixedFloat(x) - y;
  }

  template <typename T>
  friend auto operator*(const T &x, const FixedFloat &y) -> FixedFloat {
    return FixedFloat(x) * y;
  }

  template <typename T>
  friend auto operator/(const T &x, const FixedFloat &y) -> FixedFloat {
    return FixedFloat(x) / y;
  }

  template <typename T>
  friend auto operator==(const T &x, const FixedFloat &y) -> bool {
    return FixedFloat(x) == y;
  }

  template <typename T>
  friend auto operator!=(const T &x, const FixedFloat &y) -> bool {
    return FixedFloat(x) != y;
  }

  template <typename T>
  friend auto operator>(const T &x, const FixedFloat &y) -> bool {
    return FixedFloat(x) > y;
  }

  template <typename T>
  friend auto operator>=(const T &x, const FixedFloat &y) -> bool {
    return FixedFloat(x) >= y;
  }

  template <typename T>
  friend auto operator<(const T &x, const FixedFloat &y) -> bool {
    return FixedFloat(x) < y;
  }

  template <typename T>
  friend auto operator<=(const T &x, const FixedFloat &y) -> bool {
    return FixedFloat(x) <= y;
  }

  friend auto abs(const FixedFloat &x) -> FixedFloat {
    auto out = FixedFloat();
    mpfr_abs(out._x, x._x, rounding);
    return out;
  }

  friend auto log(const FixedFloat &x) -> FixedFloat {
    auto out = FixedFloat();
    mpfr_log(out._x, x._x, rounding);
    return out;
  }

  friend auto log2(const FixedFloat &x) -> FixedFloat {
    auto out = FixedFloat();
    mpfr_log2(out._x, x._x, rounding);
    return out;
  }

  friend auto pow(const FixedFloat &x, const FixedFloat &p) -> FixedFloat {
    auto out = FixedFloat();
    mpfr_pow(out._x, x._x, p._x, rounding);
    return out;
  }

  template <typename T>
  friend auto pow(const FixedFloat &x, const T &p) -> FixedFloat {
    return pow(x, FixedFloat(p));
  }

  friend auto isnan(const FixedFloat &x) -> bool { return mpfr_nan_p(x._x); }

  friend auto isinf(const FixedFloat &x) -> bool { return mpfr_inf_p(x._x); }

  friend auto signbit(const FixedFloat &x) -> bool {
    return mpfr_signbit(x._x);
  }

  friend auto to_string(const FixedFloat &x,
                        const std::optional<std::string> &fmt = std::nullopt)
      -> std::string {
    // Default format
    if (std::empty(fmt.value_or(""))) {
      std::ostringstream mpfr_fmt;
      mpfr_fmt << "1." << x.digits10() << "f";
      return to_string(x, mpfr_fmt.str());
    }

    // Python format specifier regex:
    // [[fill]align][sign][#][0][width][,][.precision][type]
    const std::regex re(
        // Group 1, 2, 3: [[fill]align]
        "((.)?([<>^=]))?"
        // Group 4: [sign]
        "([+\\- ])?"
        // Group 5: [#]
        "(#)?"
        // Group 6: [0]
        "(0)?"
        // Group 7: [width]
        "(\\d+)?"
        // Group 8: [,]
        "(,)?"
        // Group 9: [.precision]
        "(\\.\\d+)?"
        // Group 10: [type]
        "([fFeEgG%])?");

    std::smatch match;
    if (!std::regex_match(fmt.value(), match, re)) {
      throw std::runtime_error("Error: Invalid Python format string '" +
                               fmt.value() + "'.");
    }

    // --- 1. Extract components from the regex match ---
    char fill_char = match[2].matched ? match[2].str()[0] : ' ';
    char align_char = match[3].matched ? match[3].str()[0] : '>';
    std::string sign = match[4].str();
    bool alt_form = match[5].matched;
    bool zero_pad = match[6].matched;
    int width = match[7].matched ? std::stoi(match[7].str()) : 0;
    bool thousands = match[8].matched;
    std::string prec = match[9].matched ? match[9].str() : "";
    std::string type = match[10].matched ? match[10].str() : "g";

    // Default precision for %
    if (type == "%") {
      prec = prec.empty() ? ".6" : prec;
    }

    // If alignment is '=', it's like zero-padding for numbers.
    if (align_char == '=') {
      zero_pad = true;
    }

    // --- 2. Build the core mpfr_asprintf format string (without
    // alignment/width) ---
    std::ostringstream mpfr_fmt;
    mpfr_fmt << "%";
    if (!sign.empty()) {
      mpfr_fmt << sign;
    }
    if (alt_form) {
      mpfr_fmt << "#";
    }
    if (zero_pad && align_char != '<' && align_char != '^') {
      // '0' is only a printf flag for right-alignment. We handle others
      // manually.
      mpfr_fmt << "0";
    }
    if (thousands) {
      mpfr_fmt << "'";
    }
    if (!prec.empty()) {
      mpfr_fmt << prec;
    }

    // Append the MPFR type specifier (using Round to Nearest, sensible default)
    mpfr_fmt << "RN"
             << (type == "%" ? "f" : type);  // Use 'f' for percentage type

    // --- 3. Get the formatted number string from MPFR ---
    char *formatted_num_c_str = nullptr;

    // For percentage, multiply by 100 first
    if (type == "%") {
      mpfr_t tmp_;
      mpfr_init2(tmp_, mpfr_get_prec(x._x));
      mpfr_mul_ui(tmp_, x._x, 100, MPFR_RNDN);
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
      mpfr_asprintf(&formatted_num_c_str, mpfr_fmt.str().c_str(), tmp_);
      mpfr_clear(tmp_);
    } else {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
      mpfr_asprintf(&formatted_num_c_str, mpfr_fmt.str().c_str(), x._x);
    }

    // Convert to string
    std::string formatted_num(formatted_num_c_str);
    if (type == "%") {
      formatted_num += "%";
    }
    mpfr_free_str(formatted_num_c_str);

    // --- 4. Manually apply padding and alignment ---
    if (static_cast<int>(formatted_num.length()) >= width) {
      return formatted_num;
    }

    // Get padding
    int padding = width - static_cast<int>(formatted_num.length());

    // Left align
    if (align_char == '<') {
      return formatted_num + std::string(padding, fill_char);
    }

    // Center align
    if (align_char == '^') {
      int left_pad = padding / 2;
      int right_pad = padding - left_pad;
      return std::string(left_pad, fill_char) + formatted_num +
             std::string(right_pad, fill_char);
    }

    // Right align ('>' or '=')
    return std::string(padding, fill_char) + formatted_num;
  }

  template <
      typename Stream,
      std::enable_if_t<
          std::is_base_of_v<std::basic_ostream<typename Stream::char_type,
                                               typename Stream::traits_type>,
                            Stream>,
          bool> = true>
  friend auto operator<<(Stream &os, const FixedFloat &x) -> Stream & {
    os << to_string(x);
    return os;
  }

  friend auto dump(const FixedFloat &x) -> std::string {
    char *buffer = nullptr;
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg,-warnings-as-errors)
    mpfr_asprintf(&buffer, "%Ra", x._x);
    std::string out(buffer);
    mpfr_free_str(buffer);
    return out;
  }

 private:
  mpfr_t _x;
};

// NOLINTEND(cppcoreguidelines-pro-bounds-array-to-pointer-decay)

template <typename T>
struct is_FixedFloat : std::false_type {};

template <mpfr_prec_t Precision, mpfr_rnd_t Rounding>
struct is_FixedFloat<FixedFloat<Precision, Rounding>> : std::true_type {};

template <typename T>
constexpr bool is_FixedFloat_v = is_FixedFloat<T>::value;

template <mpfr_prec_t Precision>
void init(py::module &m, const std::string &name) {
  using self_type = FixedFloat<Precision>;
  py::class_<self_type>(m, name.data())
      .def(py::init<>())
      .def(py::init<long double>())
      .def(py::init<const std::string &>())
      .def(py::init<const self_type &>())
      .def_property_readonly("digits10", &self_type::digits10)
      .def_property_readonly(
          "prec", [](const self_type &self) -> auto { return self.prec; })
      .def("__format__",
           [](const self_type &self, const std::string &fmt) -> auto {
             return to_string(self, fmt);
           })
      .def("__float__",
           [](const self_type &self) -> auto { return (long double)(self); })
      .def("__repr__",
           [](const self_type &self) -> auto { return to_string(self); })
      .def("__str__",
           [](const self_type &self) -> auto { return to_string(self); })
      .def("__add__",
           [](const self_type &self, const self_type &other) -> auto {
             return self + other;
           })
      .def("__add__",
           [](const self_type &self, const long double &other) -> auto {
             return self + other;
           })
      .def("__iadd__",
           [](self_type &self, const self_type &other) -> auto {
             self += other;
             return self;
           })
      .def("__iadd__",
           [](self_type &self, const long double &other) -> auto {
             self += other;
             return self;
           })
      .def("__radd__",
           [](const self_type &self, const long double &other) -> auto {
             return other + self;
           })
      .def("__sub__",
           [](const self_type &self, const self_type &other) -> auto {
             return self - other;
           })
      .def("__sub__",
           [](const self_type &self, const long double &other) -> auto {
             return self - other;
           })
      .def("__isub__",
           [](self_type &self, const self_type &other) -> auto {
             self -= other;
             return self;
           })
      .def("__isub__",
           [](self_type &self, const long double &other) -> auto {
             self -= other;
             return self;
           })
      .def("__rsub__",
           [](const self_type &self, const long double &other) -> auto {
             return other - self;
           })
      .def("__mul__",
           [](const self_type &self, const self_type &other) -> auto {
             return self * other;
           })
      .def("__mul__",
           [](const self_type &self, const long double &other) -> auto {
             return self * other;
           })
      .def("__imul__",
           [](self_type &self, const self_type &other) -> auto {
             self *= other;
             return self;
           })
      .def("__imul__",
           [](self_type &self, const long double &other) -> auto {
             self *= other;
             return self;
           })
      .def("__rmul__",
           [](const self_type &self, const long double &other) -> auto {
             return other * self;
           })
      .def("__truediv__",
           [](const self_type &self, const self_type &other) -> auto {
             return self / other;
           })
      .def("__truediv__",
           [](const self_type &self, const long double &other) -> auto {
             return self / other;
           })
      .def("__itruediv__",
           [](self_type &self, const self_type &other) -> auto {
             self /= other;
             return self;
           })
      .def("__itruediv__",
           [](self_type &self, const long double &other) -> auto {
             self /= other;
             return self;
           })
      .def("__rtruediv__",
           [](const self_type &self, const long double &other) -> auto {
             return other / self;
           })
      .def("__pos__", [](const self_type &self) -> auto { return +self; })
      .def("__neg__", [](const self_type &self) -> auto { return -self; })
      .def("__abs__", [](const self_type &self) -> auto { return abs(self); })
      .def("__pow__",
           [](const self_type &self, const long double &p) -> auto {
             return pow(self, p);
           })
      .def("__eq__",
           [](const self_type &self, const self_type &other) -> auto {
             return self == other;
           })
      .def("__eq__",
           [](const self_type &self, const long double &other) -> auto {
             return self == other;
           })
      .def("__ne__",
           [](const self_type &self, const self_type &other) -> auto {
             return self != other;
           })
      .def("__ne__",
           [](const self_type &self, const long double &other) -> auto {
             return self != other;
           })
      .def("__lt__",
           [](const self_type &self, const self_type &other) -> auto {
             return self < other;
           })
      .def("__lt__",
           [](const self_type &self, const long double &other) -> auto {
             return self < other;
           })
      .def("__le__",
           [](const self_type &self, const self_type &other) -> auto {
             return self <= other;
           })
      .def("__le__",
           [](const self_type &self, const long double &other) -> auto {
             return self <= other;
           })
      .def("__gt__",
           [](const self_type &self, const self_type &other) -> auto {
             return self > other;
           })
      .def("__gt__",
           [](const self_type &self, const long double &other) -> auto {
             return self > other;
           })
      .def("__ge__",
           [](const self_type &self, const self_type &other) -> auto {
             return self >= other;
           })
      .def("__ge__",
           [](const self_type &self, const long double &other) -> auto {
             return self >= other;
           })
      .def("isinf", [](const self_type &self) -> auto { return isnan(self); })
      .def("isnan", [](const self_type &self) -> auto { return isnan(self); })
      .def("signbit", [](const self_type &self) -> auto { return isnan(self); })
      .def(py::pickle(
          [](const self_type &self) -> std::string { return dump(self); },
          [](const std::string &state) -> auto {
            return self_type(state, 0);
          }));
}

}  // namespace tnco::fixed_float
