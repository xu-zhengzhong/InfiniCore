#include "infinicore/ops/gbmv.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Gbmv);

Gbmv::Gbmv(const Tensor &alpha, const Tensor &a, const Tensor &x, const Tensor &beta, Tensor y, int trans, size_t kl, size_t ku) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(alpha, a, x, beta, y);
    INFINICORE_GRAPH_OP_DISPATCH(y->device().getType(), alpha, a, x, beta, y, trans, kl, ku);
}

void Gbmv::execute(const Tensor &alpha, const Tensor &a, const Tensor &x, const Tensor &beta, Tensor y, int trans, size_t kl, size_t ku) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Gbmv, alpha, a, x, beta, y, trans, kl, ku);
}

void gbmv_(const Tensor &alpha, const Tensor &a, const Tensor &x, const Tensor &beta, Tensor y, int trans, size_t kl, size_t ku) {
    Gbmv::execute(alpha, a, x, beta, y, trans, kl, ku);
}

} // namespace infinicore::op
