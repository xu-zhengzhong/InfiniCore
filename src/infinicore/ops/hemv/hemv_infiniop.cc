#include "infinicore/ops/hemv.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::hemv_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Hemv, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, alpha, a, x, beta, y;
};

void *plan(const Tensor &alpha, const Tensor &a, const Tensor &x, const Tensor &beta, Tensor y, int uplo) {
    size_t seed = hash_combine(y, alpha, a, x, beta, uplo);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Hemv,
        seed,
        static_cast<infiniopBlasFillMode_t>(uplo),
        alpha->desc(), a->desc(), x->desc(), beta->desc(), y->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, Hemv, descriptor);

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

    INFINICORE_CHECK_ERROR(infiniopHemv(
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

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Hemv, &plan, &run, &cleanup);

} // namespace infinicore::op::hemv_impl::infiniop
