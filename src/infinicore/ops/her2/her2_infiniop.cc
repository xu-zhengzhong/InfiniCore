#include "infinicore/ops/her2.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::her2_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Her2, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, alpha, x, y, a;
};

void *plan(const Tensor &alpha, const Tensor &x, const Tensor &y, Tensor a, int uplo) {
    size_t seed = hash_combine(a, alpha, x, y, uplo);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Her2,
        seed,
        static_cast<infiniopBlasFillMode_t>(uplo),
        alpha->desc(), x->desc(), y->desc(), a->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, Her2, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(alpha),
        graph::GraphTensor(x),
        graph::GraphTensor(y),
        graph::GraphTensor(a)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopHer2(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->alpha->data(),
        planned->x->data(),
        planned->y->data(),
        planned->a->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Her2, &plan, &run, &cleanup);

} // namespace infinicore::op::her2_impl::infiniop
