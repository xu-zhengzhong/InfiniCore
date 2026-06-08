#ifndef __AXPBY_METAX_H__
#define __AXPBY_METAX_H__

#include "../axpby.h"

AXPBY_DESCRIPTOR(metax);

namespace op::axpby::metax {
infiniStatus_t launchAxpby(const AxpbyInfo &info, const void *x, void *y, float alpha, float beta, void *stream);
}

#endif // __AXPBY_METAX_H__
