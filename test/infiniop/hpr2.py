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
    # uplo, n, x_stride, y_stride
    (0, 1, None, None),
    (0, 5, None, None),
    (0, 17, (2,), None),
    (0, 33, None, (2,)),
    (0, 128, (2,), (3,)),
    (0, 1024, None, None),
    (1, 1, None, None),
    (1, 5, None, None),
    (1, 17, None, (2,)),
    (1, 33, (2,), None),
    (1, 128, (3,), (2,)),
    (1, 1024, None, None),
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


def _packed_indices(uplo, n, device):
    if uplo == 0:
        return torch.tril_indices(n, n, device=device)
    return torch.triu_indices(n, n, device=device)


def _packed_rank2_update(AP, x, y, alpha, uplo, n):
    rows, cols = _packed_indices(uplo, n, AP.device)
    update = alpha * torch.outer(x, y.conj()) + alpha.conj() * torch.outer(y, x.conj())
    out = AP + update[cols, rows]

    diag = rows == cols
    out[diag] = out[diag].real.to(out.dtype)
    return out


def _packed_rank2_update_mlu(AP, x, y, alpha, uplo, n):
    rows, cols = _packed_indices(uplo, n, AP.device)
    x_real = x.real
    x_imag = x.imag
    y_real = y.real
    y_imag = y.imag
    alpha_real = alpha.real
    alpha_imag = alpha.imag

    xyh_real = torch.outer(x_real, y_real) + torch.outer(x_imag, y_imag)
    xyh_imag = torch.outer(x_imag, y_real) - torch.outer(x_real, y_imag)
    yxh_real = torch.outer(y_real, x_real) + torch.outer(y_imag, x_imag)
    yxh_imag = torch.outer(y_imag, x_real) - torch.outer(y_real, x_imag)

    update_real = alpha_real * xyh_real - alpha_imag * xyh_imag
    update_real = update_real + alpha_real * yxh_real + alpha_imag * yxh_imag
    update_imag = alpha_real * xyh_imag + alpha_imag * xyh_real
    update_imag = update_imag + alpha_real * yxh_imag - alpha_imag * yxh_real

    out = torch.empty_like(AP)
    out.real.copy_(AP.real + update_real[cols, rows])
    out.imag.copy_(AP.imag + update_imag[cols, rows])
    out.imag[rows == cols] = 0
    return out


def hpr2(alpha, AP, x, y, uplo, n):
    if AP.device.type == "mlu":
        return _packed_rank2_update_mlu(AP, x, y, alpha, uplo, n)
    return _packed_rank2_update(AP, x, y, alpha, uplo, n)


def test(
    handle,
    device,
    uplo,
    n,
    x_stride=None,
    y_stride=None,
    dtype=InfiniDtype.C64,
    sync=None,
):
    print(
        f"Testing Hpr2 on {InfiniDeviceNames[device]} with uplo:{uplo} n:{n} "
        f"x_stride:{x_stride} y_stride:{y_stride} dtype:{InfiniDtypeNames[dtype]}"
    )

    torch.manual_seed(0)
    if device != 0:
        torch.cuda.manual_seed_all(0)

    packed_len = n * (n + 1) // 2
    alpha = TestTensor(tuple(), None, dtype, device)
    x = TestTensor((n,), x_stride, dtype, device)
    y = TestTensor((n,), y_stride, dtype, device)
    AP = TestTensor((packed_len,), None, dtype, device)

    AP_ref = hpr2(
        alpha.torch_tensor(),
        AP.torch_tensor(),
        x.torch_tensor(),
        y.torch_tensor(),
        uplo,
        n,
    )
    AP.update_torch_tensor(AP_ref)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateHpr2Descriptor(
            handle,
            ctypes.byref(descriptor),
            uplo,
            alpha.descriptor,
            x.descriptor,
            y.descriptor,
            AP.descriptor,
        )
    )

    for tensor in [alpha, x, y, AP]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetHpr2WorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    def lib_hpr2():
        check_error(
            LIBINFINIOP.infiniopHpr2(
                descriptor,
                workspace.data(),
                workspace_size.value,
                alpha.data(),
                x.data(),
                y.data(),
                AP.data(),
                None,
            )
        )

    lib_hpr2()

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
            lambda: hpr2(
                alpha.torch_tensor(),
                AP.torch_tensor(),
                x.torch_tensor(),
                y.torch_tensor(),
                uplo,
                n,
            ),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
        profile_operation(
            "    lib", lambda: lib_hpr2(), device, NUM_PRERUN, NUM_ITERATIONS
        )

    check_error(LIBINFINIOP.infiniopDestroyHpr2Descriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
