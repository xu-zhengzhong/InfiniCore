#pragma once

#include <pybind11/pybind11.h>
#include "infinicore/ops/gather.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_gather(py::module &m) {
    m.def("gather",
          &op::gather,
          py::arg("input"),
          py::arg("index"),
          py::arg("dim"),
          R"doc(Gathers values along an axis specified by dim. Output has the same shape as index.)doc");

    m.def("gather_",
          &op::gather_,
          py::arg("output"),
          py::arg("input"),
          py::arg("index"),
          py::arg("dim"),
          R"doc(Explicit output gather operation. Writes the result into output.)doc");
}

} // namespace infinicore::ops