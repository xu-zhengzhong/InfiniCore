#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/axpy.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_axpy(py::module &m) {
    m.def("axpy",
          &op::axpy,
          py::arg("y"),
          py::arg("x"),
          py::arg("alpha"),
          R"doc(BLAS Level1 axpy function.)doc");

    m.def("axpy_",
          &op::axpy_,
          py::arg("y"),
          py::arg("x"),
          py::arg("alpha"),
          py::arg("out"),
          R"doc(BLAS Level1 in-place axpy function.)doc");
}

} // namespace infinicore::ops
