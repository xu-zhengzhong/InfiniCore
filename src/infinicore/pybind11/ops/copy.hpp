#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/copy.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_copy(py::module &m) {
    m.def("copy",
          &op::copy,
          py::arg("x"),
          py::arg("y"),
          R"doc(BLAS Level1 copy function.)doc");

    m.def("copy_",
          &op::copy_,
          py::arg("x"),
          py::arg("y"),
          py::arg("out"),
          R"doc(BLAS Level1 in-place copy function.)doc");
}

} // namespace infinicore::ops