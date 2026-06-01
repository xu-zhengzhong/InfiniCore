#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/spr2.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_spr2(py::module &m) {
    m.def("spr2_",
          &op::spr2_,
          py::arg("alpha"),
          py::arg("x"),
          py::arg("y"),
          py::arg("ap"),
          py::arg("uplo"),
          R"doc(In-place BLAS level-2 symmetric packed rank-2 update.)doc");
}

} // namespace infinicore::ops
