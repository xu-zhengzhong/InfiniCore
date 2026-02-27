#pragma once
#include "../device.hpp"
#include "common/op.hpp"
#include <optional>
#include <string>
#include <vector>

namespace infinicore::op {

enum class InterpolationMode {
    LINEAR,
    LOWER,
    HIGHER,
    NEAREST,
    MIDPOINT
};

class Quantile {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, std::optional<int64_t>, bool, InterpolationMode);
    static void execute(Tensor input, Tensor q, Tensor output, 
                       std::optional<int64_t> dim, bool keepdim, 
                       InterpolationMode interpolation);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor quantile(Tensor input, Tensor q, std::optional<int64_t> dim = std::nullopt, 
                bool keepdim = false, InterpolationMode interpolation = InterpolationMode::LINEAR);
void quantile_(Tensor input, Tensor q, Tensor output, 
               std::optional<int64_t> dim = std::nullopt, bool keepdim = false, 
               InterpolationMode interpolation = InterpolationMode::LINEAR);

InterpolationMode parse_interpolation_mode(const std::string &mode);

} // namespace infinicore::op