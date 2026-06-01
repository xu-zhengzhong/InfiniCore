#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/spr.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_spr(py::module &m) {
    m.def("spr_",
          &op::spr_,
          py::arg("alpha"),
          py::arg("x"),
          py::arg("ap"),
          py::arg("uplo"),
          R"doc(In-place BLAS level-2 symmetric packed rank-1 update.)doc");
}

} // namespace infinicore::ops
