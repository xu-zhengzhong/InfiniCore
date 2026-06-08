#include "infinicore/ops/axpby.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::axpby_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Axpby, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, x, y;
    float alpha, beta;
};

void *plan(const Tensor &x, Tensor y, float alpha, float beta) {
    size_t seed = hash_combine(y, x);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Axpby,
        seed,
        x->desc(), y->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, Axpby, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(x),
        graph::GraphTensor(y),
        alpha,
        beta};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopAxpby(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->x->data(),
        planned->y->data(),
        planned->alpha,
        planned->beta,
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Axpby, &plan, &run, &cleanup);

} // namespace infinicore::op::axpby_impl::infiniop
