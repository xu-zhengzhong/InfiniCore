#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/rot.hpp"
#include "../../../utils/custom_types.h"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_rot(py::module &m) {
    m.def("rot",
          [](Tensor x, Tensor y, double c, double s) {
              switch (x->dtype()) {
              case DataType::F16: {
                  fp16_t c_value = utils::cast<fp16_t>(static_cast<float>(c));
                  fp16_t s_value = utils::cast<fp16_t>(static_cast<float>(s));
                  return op::rot(x, y, &c_value, &s_value);
              }
              case DataType::BF16: {
                  bf16_t c_value = utils::cast<bf16_t>(static_cast<float>(c));
                  bf16_t s_value = utils::cast<bf16_t>(static_cast<float>(s));
                  return op::rot(x, y, &c_value, &s_value);
              }
              case DataType::F32: {
                  float c_value = static_cast<float>(c);
                  float s_value = static_cast<float>(s);
                  return op::rot(x, y, &c_value, &s_value);
              }
              case DataType::F64: {
                  double c_value = c;
                  double s_value = s;
                  return op::rot(x, y, &c_value, &s_value);
              }
              default:
                  throw std::runtime_error("rot only supports floating dtypes.");
              }
          },
          py::arg("x"),
          py::arg("y"),
          py::arg("c"),
          py::arg("s"),
          R"doc(BLAS Level1 rot function.)doc");

    m.def("rot_",
          [](Tensor x, Tensor y, Tensor out_x, Tensor out_y, double c, double s) {
              switch (x->dtype()) {
              case DataType::F16: {
                  fp16_t c_value = utils::cast<fp16_t>(static_cast<float>(c));
                  fp16_t s_value = utils::cast<fp16_t>(static_cast<float>(s));
                  op::rot_(x, y, out_x, out_y, &c_value, &s_value);
                  return;
              }
              case DataType::BF16: {
                  bf16_t c_value = utils::cast<bf16_t>(static_cast<float>(c));
                  bf16_t s_value = utils::cast<bf16_t>(static_cast<float>(s));
                  op::rot_(x, y, out_x, out_y, &c_value, &s_value);
                  return;
              }
              case DataType::F32: {
                  float c_value = static_cast<float>(c);
                  float s_value = static_cast<float>(s);
                  op::rot_(x, y, out_x, out_y, &c_value, &s_value);
                  return;
              }
              case DataType::F64: {
                  double c_value = c;
                  double s_value = s;
                  op::rot_(x, y, out_x, out_y, &c_value, &s_value);
                  return;
              }
              default:
                  throw std::runtime_error("rot only supports floating dtypes.");
              }
          },
          py::arg("x"),
          py::arg("y"),
          py::arg("out_x"),
          py::arg("out_y"),
          py::arg("c"),
          py::arg("s"),
          R"doc(BLAS Level1 in-place rot function.)doc");
}

} // namespace infinicore::ops