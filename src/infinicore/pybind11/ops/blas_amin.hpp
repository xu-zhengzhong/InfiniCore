#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/blas_amin.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_blas_amin(py::module &m) {
    m.def("blas_amin",
          &op::blas_amin,
          py::arg("x"),
          R"doc(Finds the (smallest) index of the element of the minimum magnitude.)doc");
}

} // namespace infinicore::ops