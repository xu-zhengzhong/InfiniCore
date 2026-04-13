#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/dot.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_dot(py::module &m) {
    m.def("dot",
          &op::dot,
          py::arg("x"),
          py::arg("y"),
          R"doc(BLAS Level1 dot function.)doc");
}

} // namespace infinicore::ops