#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/copy.hpp"

#include <infiniop.h>

namespace infinicore::op::copy_impl::infiniop {

thread_local common::OpCache<size_t, infiniopCopyDescriptor_t> caches(
    100, // capacity
    [](infiniopCopyDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyCopyDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor x, Tensor y) {
    size_t seed = hash_combine(x, y);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopCopyDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateCopyDescriptor(
            context::getInfiniopHandle(x->device()), &desc,
            x->desc(), y->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetCopyWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopCopy(
        desc, workspace->data(), workspace_size,
        x->data(), y->data(), context::getStream()));
}

static bool registered = []() {
    Copy::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::copy_impl::infiniop