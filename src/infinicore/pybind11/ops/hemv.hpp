#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/hemv.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_hemv(py::module &m) {
    m.def("hemv_",
          &op::hemv_,
          py::arg("alpha"),
          py::arg("a"),
          py::arg("x"),
          py::arg("beta"),
          py::arg("y"),
          py::arg("uplo"),
          R"doc(In-place BLAS level-2 Hermitian matrix-vector multiply.)doc");
}

} // namespace infinicore::ops
