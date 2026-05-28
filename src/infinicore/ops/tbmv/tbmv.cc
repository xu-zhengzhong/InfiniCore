#include "infinicore/ops/tbmv.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Tbmv);

Tbmv::Tbmv(const Tensor &a, Tensor x, int uplo, int trans, int diag, size_t k) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(a, x);
    INFINICORE_GRAPH_OP_DISPATCH(x->device().getType(), a, x, uplo, trans, diag, k);
}

void Tbmv::execute(const Tensor &a, Tensor x, int uplo, int trans, int diag, size_t k) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Tbmv, a, x, uplo, trans, diag, k);
}

void tbmv_(const Tensor &a, Tensor x, int uplo, int trans, int diag, size_t k) {
    Tbmv::execute(a, x, uplo, trans, diag, k);
}

} // namespace infinicore::op
