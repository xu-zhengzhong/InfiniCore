#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/syr.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_syr(py::module &m) {
    m.def("syr_",
          &op::syr_,
          py::arg("alpha"),
          py::arg("x"),
          py::arg("a"),
          py::arg("uplo"),
          R"doc(In-place BLAS level-2 symmetric rank-1 update.)doc");
}

} // namespace infinicore::ops
