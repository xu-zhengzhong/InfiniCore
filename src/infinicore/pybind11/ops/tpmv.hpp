#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/tpmv.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_tpmv(py::module &m) {
    m.def("tpmv_",
          &op::tpmv_,
          py::arg("ap"),
          py::arg("x"),
          py::arg("uplo"),
          py::arg("trans"),
          py::arg("diag"),
          R"doc(In-place BLAS level-2 packed triangular matrix-vector multiply.)doc");
}

} // namespace infinicore::ops
