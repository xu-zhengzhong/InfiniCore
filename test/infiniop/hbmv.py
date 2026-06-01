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
    # uplo, n, k, a_stride, x_stride, y_stride
    (0, 1, 0, None, None, None),
    (0, 5, 0, None, None, None),
    (0, 5, 1, None, None, None),
    (0, 17, 3, None, (2,), None),
    (0, 33, 4, (1, 8), None, (2,)),
    (0, 33, 32, None, (2,), None),
    (0, 128, 7, None, (2,), (3,)),
    (0, 1024, 2, None, None, None),
    (1, 1, 0, None, None, None),
    (1, 5, 0, None, None, None),
    (1, 5, 1, None, None, None),
    (1, 17, 3, None, None, (2,)),
    (1, 33, 4, (1, 8), (2,), None),
    (1, 33, 32, None, None, None),
    (1, 128, 7, None, (3,), (2,)),
    (1, 1024, 2, None, None, None),
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


def _full_from_hermitian_band(a, n, k, uplo):
    full = torch.zeros((n, n), dtype=a.dtype, device=a.device)
    if uplo == 0:
        for j in range(n):
            i_begin = max(0, j - k)
            for i in range(i_begin, j + 1):
                value = a[k + i - j, j]
                if i == j:
                    value = value.real.to(a.dtype)
                full[i, j] = value
                full[j, i] = value.conj()
    else:
        for j in range(n):
            i_end = min(n, j + k + 1)
            for i in range(j, i_end):
                value = a[i - j, j]
                if i == j:
                    value = value.real.to(a.dtype)
                full[i, j] = value
                full[j, i] = value.conj()
    return full


def _full_from_hermitian_band_mlu(a, n, k, uplo):
    full_real = torch.zeros((n, n), dtype=a.real.dtype, device=a.device)
    full_imag = torch.zeros((n, n), dtype=a.real.dtype, device=a.device)
    a_real = a.real
    a_imag = a.imag

    if uplo == 0:
        for j in range(n):
            i_begin = max(0, j - k)
            for i in range(i_begin, j + 1):
                value_real = a_real[k + i - j, j]
                value_imag = a_imag[k + i - j, j]
                if i == j:
                    value_imag = value_imag * 0
                full_real[i, j] = value_real
                full_real[j, i] = value_real
                full_imag[i, j] = value_imag
                full_imag[j, i] = -value_imag
    else:
        for j in range(n):
            i_end = min(n, j + k + 1)
            for i in range(j, i_end):
                value_real = a_real[i - j, j]
                value_imag = a_imag[i - j, j]
                if i == j:
                    value_imag = value_imag * 0
                full_real[i, j] = value_real
                full_real[j, i] = value_real
                full_imag[i, j] = value_imag
                full_imag[j, i] = -value_imag

    return full_real, full_imag


def _hbmv_mlu(alpha, a, x, beta, y, uplo, k):
    matrix_real, matrix_imag = _full_from_hermitian_band_mlu(a, x.shape[0], k, uplo)
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


def hbmv(alpha, a, x, beta, y, uplo, k):
    if a.device.type == "mlu":
        return _hbmv_mlu(alpha, a, x, beta, y, uplo, k)

    matrix = _full_from_hermitian_band(a, x.shape[0], k, uplo)
    return alpha * torch.mv(matrix, x) + beta * y


def test(
    handle,
    device,
    uplo,
    n,
    k,
    a_stride=None,
    x_stride=None,
    y_stride=None,
    dtype=InfiniDtype.C64,
    sync=None,
):
    print(
        f"Testing Hbmv on {InfiniDeviceNames[device]} with uplo:{uplo} n:{n} k:{k} "
        f"a_stride:{a_stride} x_stride:{x_stride} y_stride:{y_stride} dtype:{InfiniDtypeNames[dtype]}"
    )

    if a_stride is None:
        a_stride = (1, k + 1)

    torch.manual_seed(0)
    if device != 0:
        torch.cuda.manual_seed_all(0)

    alpha = TestTensor(tuple(), None, dtype, device)
    beta = TestTensor(tuple(), None, dtype, device)
    A = TestTensor((k + 1, n), a_stride, dtype, device)
    x = TestTensor((n,), x_stride, dtype, device)
    y = TestTensor((n,), y_stride, dtype, device)

    y_ref = hbmv(
        alpha.torch_tensor(),
        A.torch_tensor(),
        x.torch_tensor(),
        beta.torch_tensor(),
        y.torch_tensor(),
        uplo,
        k,
    )
    y.update_torch_tensor(y_ref)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateHbmvDescriptor(
            handle,
            ctypes.byref(descriptor),
            uplo,
            k,
            alpha.descriptor,
            A.descriptor,
            x.descriptor,
            beta.descriptor,
            y.descriptor,
        )
    )

    for tensor in [alpha, beta, A, x, y]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetHbmvWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    def lib_hbmv():
        check_error(
            LIBINFINIOP.infiniopHbmv(
                descriptor,
                workspace.data(),
                workspace_size.value,
                alpha.data(),
                A.data(),
                x.data(),
                beta.data(),
                y.data(),
                None,
            )
        )

    lib_hbmv()

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
            lambda: hbmv(
                alpha.torch_tensor(),
                A.torch_tensor(),
                x.torch_tensor(),
                beta.torch_tensor(),
                y.torch_tensor(),
                uplo,
                k,
            ),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
        profile_operation(
            "    lib", lambda: lib_hbmv(), device, NUM_PRERUN, NUM_ITERATIONS
        )

    check_error(LIBINFINIOP.infiniopDestroyHbmvDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
