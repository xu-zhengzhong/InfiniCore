#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/swap.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_swap(py::module &m) {
    m.def("swap_",
          &op::swap_,
          py::arg("x"),
          py::arg("y"),
          R"doc(BLAS Level1 in-place swap function.)doc");
}

} // namespace infinicore::ops