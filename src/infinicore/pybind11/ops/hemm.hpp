#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/hemm.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_hemm(py::module &m) {
    m.def("hemm_",
          &op::hemm_,
          py::arg("a"),
          py::arg("b"),
          py::arg("alpha"),
          py::arg("beta"),
          py::arg("c"),
          py::arg("side"),
          py::arg("uplo"),
          R"doc(In-place BLAS level-3 Hermitian matrix-matrix multiply.)doc");
}

} // namespace infinicore::ops
