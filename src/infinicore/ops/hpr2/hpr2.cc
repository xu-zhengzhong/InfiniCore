#include "infinicore/ops/hpr2.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Hpr2);

Hpr2::Hpr2(const Tensor &alpha, const Tensor &x, const Tensor &y, Tensor ap, int uplo) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(alpha, x, y, ap);
    INFINICORE_GRAPH_OP_DISPATCH(ap->device().getType(), alpha, x, y, ap, uplo);
}

void Hpr2::execute(const Tensor &alpha, const Tensor &x, const Tensor &y, Tensor ap, int uplo) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Hpr2, alpha, x, y, ap, uplo);
}

void hpr2_(const Tensor &alpha, const Tensor &x, const Tensor &y, Tensor ap, int uplo) {
    Hpr2::execute(alpha, x, y, ap, uplo);
}

} // namespace infinicore::op
