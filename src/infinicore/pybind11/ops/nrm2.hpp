#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/nrm2.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_nrm2(py::module &m) {
    m.def("nrm2",
          &op::nrm2,
          py::arg("x"),
          R"doc(Computes the Euclidean norm of the vector x.)doc");
}

} // namespace infinicore::ops
