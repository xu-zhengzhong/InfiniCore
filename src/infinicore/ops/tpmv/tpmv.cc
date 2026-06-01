#include "infinicore/ops/tpmv.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(Tpmv);

Tpmv::Tpmv(const Tensor &ap, Tensor x, int uplo, int trans, int diag) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(ap, x);
    INFINICORE_GRAPH_OP_DISPATCH(x->device().getType(), ap, x, uplo, trans, diag);
}

void Tpmv::execute(const Tensor &ap, Tensor x, int uplo, int trans, int diag) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(Tpmv, ap, x, uplo, trans, diag);
}

void tpmv_(const Tensor &ap, Tensor x, int uplo, int trans, int diag) {
    Tpmv::execute(ap, x, uplo, trans, diag);
}

} // namespace infinicore::op
