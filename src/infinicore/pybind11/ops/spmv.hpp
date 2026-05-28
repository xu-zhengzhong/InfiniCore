#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/spmv.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_spmv(py::module &m) {
    m.def("spmv_",
          &op::spmv_,
          py::arg("alpha"),
          py::arg("ap"),
          py::arg("x"),
          py::arg("beta"),
          py::arg("y"),
          py::arg("uplo"),
          R"doc(In-place BLAS level-2 symmetric packed matrix-vector multiply.)doc");
}

} // namespace infinicore::ops
