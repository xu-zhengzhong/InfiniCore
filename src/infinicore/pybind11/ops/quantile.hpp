#pragma once

#include "infinicore/ops/quantile.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_quantile(py::module &m) {
    // Bind InterpolationMode enum
    py::enum_<op::InterpolationMode>(m, "InterpolationMode")
        .value("LINEAR", op::InterpolationMode::LINEAR)
        .value("LOWER", op::InterpolationMode::LOWER)
        .value("HIGHER", op::InterpolationMode::HIGHER)
        .value("NEAREST", op::InterpolationMode::NEAREST)
        .value("MIDPOINT", op::InterpolationMode::MIDPOINT)
        .export_values();
    
    m.def("quantile",
          [](Tensor input, Tensor q, py::object dim, bool keepdim, const std::string &interpolation) {
              std::optional<int64_t> dim_opt;
              if (!dim.is_none()) {
                  dim_opt = py::cast<int64_t>(dim);
              }
              auto mode = op::parse_interpolation_mode(interpolation);
              return op::quantile(input, q, dim_opt, keepdim, mode);
          },
          py::arg("input"),
          py::arg("q"),
          py::arg("dim") = py::none(),
          py::arg("keepdim") = false,
          py::arg("interpolation") = "linear",
          R"doc(Computes the q-th quantiles of each row of the input tensor along the dimension dim.)doc");

    m.def("quantile_",
          [](Tensor input, Tensor q, Tensor output, py::object dim, bool keepdim, const std::string &interpolation) {
              std::optional<int64_t> dim_opt;
              if (!dim.is_none()) {
                  dim_opt = py::cast<int64_t>(dim);
              }
              auto mode = op::parse_interpolation_mode(interpolation);
              op::quantile_(input, q, output, dim_opt, keepdim, mode);
          },
          py::arg("input"),
          py::arg("q"),
          py::arg("output"),
          py::arg("dim") = py::none(),
          py::arg("keepdim") = false,
          py::arg("interpolation") = "linear",
          R"doc(In-place quantile.)doc");
}

} // namespace infinicore::ops