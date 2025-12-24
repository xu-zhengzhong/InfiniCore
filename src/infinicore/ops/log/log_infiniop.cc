#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/log.hpp"
#include "infinicore/ops/common/cache.hpp"
#include <infiniop.h>
#include <unordered_map>

namespace infinicore::op::log_impl::infiniop {

thread_local common::OpCache<size_t, infiniopLogDescriptor_t> caches(
    100,
    [](infiniopLogDescriptor_t &desc) {
        if (desc) { INFINICORE_CHECK_ERROR(infiniopDestroyLogDescriptor(desc)); desc=nullptr; }
    }
);

struct WorkspaceEntry { size_t size=0; std::shared_ptr<Memory> buf=nullptr; };

void calculate(Tensor output, Tensor input) {
    size_t seed = hash_combine(output, input);
    auto device_type  = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();
    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopLogDescriptor_t desc = nullptr;
    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateLogDescriptor(
            context::getInfiniopHandle(output->device()), &desc,
            output->desc(), input->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    static thread_local std::unordered_map<infiniopLogDescriptor_t, WorkspaceEntry> s_workspace;
    auto it = s_workspace.find(desc);
    if (it == s_workspace.end()) {
        size_t ws = 0; INFINICORE_CHECK_ERROR(infiniopGetLogWorkspaceSize(desc, &ws));
        WorkspaceEntry e;
        if (ws) { e.buf = context::allocateMemory(ws); e.size = ws; }
        it = s_workspace.emplace(desc, std::move(e)).first;
    } else {
        size_t need = 0; INFINICORE_CHECK_ERROR(infiniopGetLogWorkspaceSize(desc, &need));
        if (need > it->second.size) {
            it->second.buf = context::allocateMemory(need);
            it->second.size = need;
        }
    }

    void* ws_ptr = (it!=s_workspace.end() && it->second.buf) ? it->second.buf->data() : nullptr;
    size_t ws_size = (it!=s_workspace.end()) ? it->second.size : 0;

    INFINICORE_CHECK_ERROR(infiniopLog(
        desc, ws_ptr, ws_size, output->data(), input->data(), context::getStream()));
}

static bool registered = [](){ Log::dispatcher().registerAll(&calculate, false); return true; }();

} // namespace infinicore::op::log_impl::infiniop