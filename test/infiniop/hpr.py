import ctypes
from ctypes import c_uint64

import torch
from libinfiniop import (
    LIBINFINIOP,
    InfiniDeviceNames,
    InfiniDtype,
    InfiniDtypeNames,
    TestTensor,
    TestWorkspace,
    check_error,
    debug,
    get_args,
    get_test_devices,
    get_tolerance,
    infiniopOperatorDescriptor_t,
    profile_operation,
    test_operator,
)

# ==============================================================================
#  Configuration
# ==============================================================================

_TEST_CASES = [
    # uplo, n, x_stride
    (0, 1, None),
    (0, 5, None),
    (0, 17, (2,)),
    (0, 33, None),
    (0, 128, (3,)),
    (0, 1024, None),
    (1, 1, None),
    (1, 5, None),
    (1, 17, None),
    (1, 33, (2,)),
    (1, 128, (3,)),
    (1, 1024, None),
]

_TENSOR_DTYPES = [
    InfiniDtype.C64,
    # InfiniDtype.C128,
]

_TOLERANCE_MAP = {
    InfiniDtype.C64: {"atol": 1e-5, "rtol": 1e-5},
    InfiniDtype.C128: {"atol": 1e-9, "rtol": 1e-9},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def _real_dtype(dtype):
    if dtype == InfiniDtype.C64:
        return InfiniDtype.F32
    if dtype == InfiniDtype.C128:
        return InfiniDtype.F64
    raise ValueError(f"Unsupported Hpr dtype: {dtype}")


def _packed_indices(uplo, n, device):
    if uplo == 0:
        return torch.tril_indices(n, n, device=device)
    return torch.triu_indices(n, n, device=device)


def _packed_rank1_update(AP, x, alpha, uplo, n):
    rows, cols = _packed_indices(uplo, n, AP.device)
    update = alpha * torch.outer(x, x.conj())
    out = AP + update[cols, rows]

    diag = rows == cols
    out[diag] = out[diag].real.to(out.dtype)
    return out


def _packed_rank1_update_mlu(AP, x, alpha, uplo, n):
    rows, cols = _packed_indices(uplo, n, AP.device)
    x_real = x.real
    x_imag = x.imag
    update_real = alpha * (torch.outer(x_real, x_real) + torch.outer(x_imag, x_imag))
    update_imag = alpha * (torch.outer(x_imag, x_real) - torch.outer(x_real, x_imag))

    out = torch.empty_like(AP)
    out.real.copy_(AP.real + update_real[cols, rows])
    out.imag.copy_(AP.imag + update_imag[cols, rows])
    out.imag[rows == cols] = 0
    return out


def hpr(alpha, AP, x, uplo, n):
    if AP.device.type == "mlu":
        return _packed_rank1_update_mlu(AP, x, alpha, uplo, n)
    return _packed_rank1_update(AP, x, alpha, uplo, n)


def test(
    handle,
    device,
    uplo,
    n,
    x_stride=None,
    dtype=InfiniDtype.C64,
    sync=None,
):
    print(
        f"Testing Hpr on {InfiniDeviceNames[device]} with uplo:{uplo} n:{n} "
        f"x_stride:{x_stride} dtype:{InfiniDtypeNames[dtype]}"
    )

    torch.manual_seed(0)
    if device != 0:
        torch.cuda.manual_seed_all(0)

    packed_len = n * (n + 1) // 2
    alpha = TestTensor(tuple(), None, _real_dtype(dtype), device)
    x = TestTensor((n,), x_stride, dtype, device)
    AP = TestTensor((packed_len,), None, dtype, device)

    AP_ref = hpr(
        alpha.torch_tensor(),
        AP.torch_tensor(),
        x.torch_tensor(),
        uplo,
        n,
    )
    AP.update_torch_tensor(AP_ref)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateHprDescriptor(
            handle,
            ctypes.byref(descriptor),
            uplo,
            alpha.descriptor,
            x.descriptor,
            AP.descriptor,
        )
    )

    for tensor in [alpha, x, AP]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetHprWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    def lib_hpr():
        check_error(
            LIBINFINIOP.infiniopHpr(
                descriptor,
                workspace.data(),
                workspace_size.value,
                alpha.data(),
                x.data(),
                AP.data(),
                None,
            )
        )

    lib_hpr()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(AP.actual_tensor().real, AP.torch_tensor().real, atol=atol, rtol=rtol)
        debug(AP.actual_tensor().imag, AP.torch_tensor().imag, atol=atol, rtol=rtol)
    assert torch.allclose(
        AP.actual_tensor().real, AP.torch_tensor().real, atol=atol, rtol=rtol
    ) and torch.allclose(
        AP.actual_tensor().imag, AP.torch_tensor().imag, atol=atol, rtol=rtol
    )

    if PROFILE:
        profile_operation(
            "PyTorch",
            lambda: hpr(
                alpha.torch_tensor(),
                AP.torch_tensor(),
                x.torch_tensor(),
                uplo,
                n,
            ),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
        profile_operation(
            "    lib", lambda: lib_hpr(), device, NUM_PRERUN, NUM_ITERATIONS
        )

    check_error(LIBINFINIOP.infiniopDestroyHprDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
