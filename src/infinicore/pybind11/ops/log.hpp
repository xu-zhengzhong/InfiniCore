#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/log.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_log(py::module &m) {
    m.def("log",
          &op::log,
          py::arg("input"),
          R"doc(Natural logarithm function.)doc");

    m.def("log_",
          &op::log_,
          py::arg("output"),
          py::arg("input"),
          R"doc(In-place natural logarithm function.)doc");
}

} // namespace infinicore::ops
