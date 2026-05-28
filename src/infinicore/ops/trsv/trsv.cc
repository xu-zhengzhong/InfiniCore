#include "infinicore/ops/trsv.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Trsv);

Trsv::Trsv(const Tensor &a, Tensor x, int uplo, int trans, int diag) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(a, x);
    INFINICORE_GRAPH_OP_DISPATCH(x->device().getType(), a, x, uplo, trans, diag);
}

void Trsv::execute(const Tensor &a, Tensor x, int uplo, int trans, int diag) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Trsv, a, x, uplo, trans, diag);
}

void trsv_(const Tensor &a, Tensor x, int uplo, int trans, int diag) {
    Trsv::execute(a, x, uplo, trans, diag);
}

} // namespace infinicore::op
