#include "infinicore/ops/gbmv.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::gbmv_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Gbmv, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, alpha, a, x, beta, y;
};

void *plan(const Tensor &alpha, const Tensor &a, const Tensor &x, const Tensor &beta, Tensor y, int trans, size_t kl, size_t ku) {
    size_t seed = hash_combine(y, alpha, a, x, beta, trans, kl, ku);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Gbmv,
        seed,
        static_cast<infiniopBlasOperation_t>(trans),
        kl,
        ku,
        alpha->desc(), a->desc(), x->desc(), beta->desc(), y->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, Gbmv, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(alpha),
        graph::GraphTensor(a),
        graph::GraphTensor(x),
        graph::GraphTensor(beta),
        graph::GraphTensor(y)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopGbmv(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->alpha->data(),
        planned->a->data(),
        planned->x->data(),
        planned->beta->data(),
        planned->y->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Gbmv, &plan, &run, &cleanup);

} // namespace infinicore::op::gbmv_impl::infiniop
