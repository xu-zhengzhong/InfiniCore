#include "infinicore/ops/hemm.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::hemm_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Hemm, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, alpha, a, b, beta, c;
};

void *plan(const Tensor &a, const Tensor &b, const Tensor &alpha, const Tensor &beta, Tensor c, int side, int uplo) {
    size_t seed = hash_combine(c, a, b, alpha, beta, side, uplo);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Hemm,
        seed,
        static_cast<infiniopBlasSideMode_t>(side),
        static_cast<infiniopBlasFillMode_t>(uplo),
        alpha->desc(), a->desc(), b->desc(), beta->desc(), c->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, Hemm, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(alpha),
        graph::GraphTensor(a),
        graph::GraphTensor(b),
        graph::GraphTensor(beta),
        graph::GraphTensor(c)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopHemm(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->alpha->data(),
        planned->a->data(),
        planned->b->data(),
        planned->beta->data(),
        planned->c->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Hemm, &plan, &run, &cleanup);

} // namespace infinicore::op::hemm_impl::infiniop
