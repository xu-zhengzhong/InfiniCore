#include "infinicore/ops/spr.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Spr);

Spr::Spr(const Tensor &alpha, const Tensor &x, Tensor ap, int uplo) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(alpha, x, ap);
    INFINICORE_GRAPH_OP_DISPATCH(ap->device().getType(), alpha, x, ap, uplo);
}

void Spr::execute(const Tensor &alpha, const Tensor &x, Tensor ap, int uplo) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Spr, alpha, x, ap, uplo);
}

void spr_(const Tensor &alpha, const Tensor &x, Tensor ap, int uplo) {
    Spr::execute(alpha, x, ap, uplo);
}

} // namespace infinicore::op
