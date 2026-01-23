#pragma once

#include "infinicore/ops/signbit.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_signbit(py::module &m) {
    m.def("signbit",
          &op::signbit,
          py::arg("input"),
          R"doc(Tests if each element of input has its sign bit set or not.)doc");

    m.def("signbit_",
          &op::signbit_,
          py::arg("input"),
          py::arg("output"),
          R"doc(In-place signbit.)doc");
}

} // namespace infinicore::ops