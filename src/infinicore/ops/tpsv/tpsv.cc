#include "infinicore/ops/tpsv.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Tpsv);

Tpsv::Tpsv(const Tensor &ap, Tensor x, int uplo, int trans, int diag) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(ap, x);
    INFINICORE_GRAPH_OP_DISPATCH(x->device().getType(), ap, x, uplo, trans, diag);
}

void Tpsv::execute(const Tensor &ap, Tensor x, int uplo, int trans, int diag) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Tpsv, ap, x, uplo, trans, diag);
}

void tpsv_(const Tensor &ap, Tensor x, int uplo, int trans, int diag) {
    Tpsv::execute(ap, x, uplo, trans, diag);
}

} // namespace infinicore::op
