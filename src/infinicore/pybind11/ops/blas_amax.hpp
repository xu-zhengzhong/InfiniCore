#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/blas_amax.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_blas_amax(py::module &m) {
    m.def("blas_amax",
          &op::blas_amax,
          py::arg("x"),
          R"doc(Finds the (smallest) index of the element of the maximum magnitude.)doc");
}

} // namespace infinicore::ops
