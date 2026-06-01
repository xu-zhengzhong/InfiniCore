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
    InfiniDtype.C64: {"atol": 5e-4, "rtol": 5e-4},
    InfiniDtype.C128: {"atol": 1e-9, "rtol": 1e-9},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def _packed_to_full(AP, uplo, n):
    matrix = torch.empty((n, n), dtype=AP.dtype, device=AP.device)
    if uplo == 0:
        rows, cols = torch.tril_indices(n, n, device=AP.device)
    else:
        rows, cols = torch.triu_indices(n, n, device=AP.device)
    matrix[cols, rows] = AP
    matrix[rows, cols] = AP.conj()

    idx = torch.arange(n, device=AP.device)
    matrix[idx, idx] = matrix[idx, idx].real.to(matrix.dtype)
    return matrix


def _packed_to_full_mlu(AP, uplo, n):
    matrix_real = torch.empty((n, n), dtype=AP.real.dtype, device=AP.device)
    matrix_imag = torch.empty((n, n), dtype=AP.real.dtype, device=AP.device)
    if uplo == 0:
        rows, cols = torch.tril_indices(n, n, device=AP.device)
    else:
        rows, cols = torch.triu_indices(n, n, device=AP.device)

    matrix_real[cols, rows] = AP.real
    matrix_real[rows, cols] = AP.real
    matrix_imag[cols, rows] = AP.imag
    matrix_imag[rows, cols] = -AP.imag

    idx = torch.arange(n, device=AP.device)
    matrix_imag[idx, idx] = 0
    return matrix_real, matrix_imag


def _hpmv_mlu(alpha, AP, x, beta, y, uplo, n):
    matrix_real, matrix_imag = _packed_to_full_mlu(AP, uplo, n)
    x_real = x.real
    x_imag = x.imag

    mv_real = torch.mv(matrix_real, x_real) - torch.mv(matrix_imag, x_imag)
    mv_imag = torch.mv(matrix_real, x_imag) + torch.mv(matrix_imag, x_real)

    alpha_real = alpha.real
    alpha_imag = alpha.imag
    beta_real = beta.real
    beta_imag = beta.imag
    y_real = y.real
    y_imag = y.imag

    out_real = alpha_real * mv_real - alpha_imag * mv_imag
    out_real = out_real + beta_real * y_real - beta_imag * y_imag
    out_imag = alpha_real * mv_imag + alpha_imag * mv_real
    out_imag = out_imag + beta_real * y_imag + beta_imag * y_real

    out = torch.empty_like(y)
    out.real.copy_(out_real)
    out.imag.copy_(out_imag)
    return out


def hpmv(alpha, AP, x, beta, y, uplo, n):
    if AP.device.type == "mlu":
        return _hpmv_mlu(alpha, AP, x, beta, y, uplo, n)

    matrix = _packed_to_full(AP, uplo, n)
    return alpha * torch.mv(matrix, x) + beta * y


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
        f"Testing Hpmv on {InfiniDeviceNames[device]} with uplo:{uplo} n:{n} "
        f"x_stride:{x_stride} y_stride:{y_stride} dtype:{InfiniDtypeNames[dtype]}"
    )

    torch.manual_seed(0)
    if device != 0:
        torch.cuda.manual_seed_all(0)

    packed_len = n * (n + 1) // 2
    alpha = TestTensor(tuple(), None, dtype, device)
    beta = TestTensor(tuple(), None, dtype, device)
    AP = TestTensor((packed_len,), None, dtype, device)
    x = TestTensor((n,), x_stride, dtype, device)
    y = TestTensor((n,), y_stride, dtype, device)

    y_ref = hpmv(
        alpha.torch_tensor(),
        AP.torch_tensor(),
        x.torch_tensor(),
        beta.torch_tensor(),
        y.torch_tensor(),
        uplo,
        n,
    )
    y.update_torch_tensor(y_ref)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateHpmvDescriptor(
            handle,
            ctypes.byref(descriptor),
            uplo,
            alpha.descriptor,
            AP.descriptor,
            x.descriptor,
            beta.descriptor,
            y.descriptor,
        )
    )

    for tensor in [alpha, beta, AP, x, y]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetHpmvWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    def lib_hpmv():
        check_error(
            LIBINFINIOP.infiniopHpmv(
                descriptor,
                workspace.data(),
                workspace_size.value,
                alpha.data(),
                AP.data(),
                x.data(),
                beta.data(),
                y.data(),
                None,
            )
        )

    lib_hpmv()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(y.actual_tensor().real, y.torch_tensor().real, atol=atol, rtol=rtol)
        debug(y.actual_tensor().imag, y.torch_tensor().imag, atol=atol, rtol=rtol)
    assert torch.allclose(
        y.actual_tensor().real, y.torch_tensor().real, atol=atol, rtol=rtol
    ) and torch.allclose(
        y.actual_tensor().imag, y.torch_tensor().imag, atol=atol, rtol=rtol
    )

    if PROFILE:
        profile_operation(
            "PyTorch",
            lambda: hpmv(
                alpha.torch_tensor(),
                AP.torch_tensor(),
                x.torch_tensor(),
                beta.torch_tensor(),
                y.torch_tensor(),
                uplo,
                n,
            ),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
        profile_operation(
            "    lib", lambda: lib_hpmv(), device, NUM_PRERUN, NUM_ITERATIONS
        )

    check_error(LIBINFINIOP.infiniopDestroyHpmvDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
