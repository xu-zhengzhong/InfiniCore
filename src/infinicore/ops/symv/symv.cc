#include "infinicore/ops/symv.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Symv);

Symv::Symv(const Tensor &alpha, const Tensor &a, const Tensor &x, const Tensor &beta, Tensor y, int uplo) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(alpha, a, x, beta, y);
    INFINICORE_GRAPH_OP_DISPATCH(y->device().getType(), alpha, a, x, beta, y, uplo);
}

void Symv::execute(const Tensor &alpha, const Tensor &a, const Tensor &x, const Tensor &beta, Tensor y, int uplo) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Symv, alpha, a, x, beta, y, uplo);
}

void symv_(const Tensor &alpha, const Tensor &a, const Tensor &x, const Tensor &beta, Tensor y, int uplo) {
    Symv::execute(alpha, a, x, beta, y, uplo);
}

} // namespace infinicore::op
