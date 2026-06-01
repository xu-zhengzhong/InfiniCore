#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/tbsv.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_tbsv(py::module &m) {
    m.def("tbsv_",
          &op::tbsv_,
          py::arg("a"),
          py::arg("x"),
          py::arg("uplo"),
          py::arg("trans"),
          py::arg("diag"),
          py::arg("k"),
          R"doc(In-place BLAS level-2 triangular band solve.)doc");
}

} // namespace infinicore::ops
