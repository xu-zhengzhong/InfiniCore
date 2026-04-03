#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/asum.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_asum(py::module &m) {
    m.def("asum",
          &op::asum,
          py::arg("x"),
          R"doc(Computes the sum of the absolute values of the elements of vector x.)doc");
}

} // namespace infinicore::ops
