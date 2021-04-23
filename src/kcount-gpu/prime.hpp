#pragma once

//---------------------------------------------------------------------------
// Pre-computed list of prime numbers, approx. 5% apart, with cheap div/modulo
// computation. Suitable for efficient hash tables
//
// SPDX-License-Identifier: MIT
// (c) 2020 Thomas Neumann
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice (including the next
// paragraph) shall be included in all copies or substantial portions of the
// Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//---------------------------------------------------------------------------
#include <cstdint>
//---------------------------------------------------------------------------
namespace primes {
//---------------------------------------------------------------------------
struct Number {
  uint64_t value, magic, shift;
};

/// A pre-computed prime number, with efficient division and multiplication logic
class Prime {
 private:
  Number number = {0, 0, 0};

 public:
  /// Get the value of the prime number
  uint64_t get();
  /// Divide the argument by the prime number
  uint64_t div(uint64_t x);
  /// Return the remainder after division
  uint64_t mod(uint64_t x);
  /// Pick a suitable prime number larger than the argument
  void set(uint64_t desiredSize);
};
//---------------------------------------------------------------------------
}  // namespace primes
