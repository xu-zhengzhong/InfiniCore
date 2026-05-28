#include "infinicore/ops/spmv.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::spmv_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Spmv, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, alpha, ap, x, beta, y;
};

void *plan(const Tensor &alpha, const Tensor &ap, const Tensor &x, const Tensor &beta, Tensor y, int uplo) {
    size_t seed = hash_combine(y, alpha, ap, x, beta, uplo);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Spmv,
        seed,
        static_cast<infiniopBlasFillMode_t>(uplo),
        alpha->desc(), ap->desc(), x->desc(), beta->desc(), y->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, Spmv, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(alpha),
        graph::GraphTensor(ap),
        graph::GraphTensor(x),
        graph::GraphTensor(beta),
        graph::GraphTensor(y)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopSpmv(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->alpha->data(),
        planned->ap->data(),
        planned->x->data(),
        planned->beta->data(),
        planned->y->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Spmv, &plan, &run, &cleanup);

} // namespace infinicore::op::spmv_impl::infiniop
