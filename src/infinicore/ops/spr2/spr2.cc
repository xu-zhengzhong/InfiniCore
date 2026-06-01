#include "infinicore/ops/spr2.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Spr2);

Spr2::Spr2(const Tensor &alpha, const Tensor &x, const Tensor &y, Tensor ap, int uplo) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(alpha, x, y, ap);
    INFINICORE_GRAPH_OP_DISPATCH(ap->device().getType(), alpha, x, y, ap, uplo);
}

void Spr2::execute(const Tensor &alpha, const Tensor &x, const Tensor &y, Tensor ap, int uplo) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Spr2, alpha, x, y, ap, uplo);
}

void spr2_(const Tensor &alpha, const Tensor &x, const Tensor &y, Tensor ap, int uplo) {
    Spr2::execute(alpha, x, y, ap, uplo);
}

} // namespace infinicore::op
