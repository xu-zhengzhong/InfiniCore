#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/syr2.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_syr2(py::module &m) {
    m.def("syr2_",
          &op::syr2_,
          py::arg("alpha"),
          py::arg("x"),
          py::arg("y"),
          py::arg("a"),
          py::arg("uplo"),
          R"doc(In-place BLAS level-2 symmetric rank-2 update.)doc");
}

} // namespace infinicore::ops
