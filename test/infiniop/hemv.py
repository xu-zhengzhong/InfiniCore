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
    # uplo, n, a_stride, x_stride, y_stride
    (0, 1, None, None, None),
    (0, 5, None, None, None),
    (0, 17, None, (2,), None),
    (0, 33, (1, 40), None, (2,)),
    (0, 128, None, (2,), (3,)),
    (0, 1024, None, None, None),
    (0, 2048, None, (2,), None),
    (0, 4096, (1, 8192), None, None),
    (0, 5632, None, None, (3,)),
    (1, 1, None, None, None),
    (1, 5, None, None, None),
    (1, 17, None, None, (2,)),
    (1, 33, (1, 40), (2,), None),
    (1, 128, None, (3,), (2,)),
    (1, 1024, None, None, None),
    (1, 2048, None, (2,), None),
    (1, 4096, (1, 8192), None, None),
    (1, 5632, None, None, (3,)),
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


def _full_from_triangle(A, uplo):
    if uplo == 0:
        matrix = torch.triu(A) + torch.triu(A, diagonal=1).mH
    else:
        matrix = torch.tril(A) + torch.tril(A, diagonal=-1).mH
    idx = torch.arange(A.shape[0], device=A.device)
    matrix[idx, idx] = matrix[idx, idx].real.to(matrix.dtype)
    return matrix


def _full_from_triangle_mlu(A, uplo):
    A_real = A.real
    A_imag = A.imag

    if uplo == 0:
        matrix_real = torch.triu(A_real) + torch.triu(A_real, diagonal=1).t()
        matrix_imag = torch.triu(A_imag) - torch.triu(A_imag, diagonal=1).t()
    else:
        matrix_real = torch.tril(A_real) + torch.tril(A_real, diagonal=-1).t()
        matrix_imag = torch.tril(A_imag) - torch.tril(A_imag, diagonal=-1).t()

    idx = torch.arange(A.shape[0], device=A.device)
    matrix_imag[idx, idx] = 0
    return matrix_real, matrix_imag


def _hemv_mlu(alpha, matrix, x, beta, y, uplo):
    matrix_real, matrix_imag = _full_from_triangle_mlu(matrix, uplo)
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


def hemv(alpha, A, x, beta, y, uplo):
    if A.device.type == "mlu":
        return _hemv_mlu(alpha, A, x, beta, y, uplo)

    matrix = _full_from_triangle(A, uplo)
    return alpha * torch.mv(matrix, x) + beta * y


def test(
    handle,
    device,
    uplo,
    n,
    a_stride=None,
    x_stride=None,
    y_stride=None,
    dtype=InfiniDtype.C64,
    sync=None,
):
    print(
        f"Testing Hemv on {InfiniDeviceNames[device]} with uplo:{uplo} n:{n} "
        f"a_stride:{a_stride} x_stride:{x_stride} y_stride:{y_stride} dtype:{InfiniDtypeNames[dtype]}"
    )

    if a_stride is None:
        a_stride = (1, n)

    torch.manual_seed(0)
    if device != 0:
        torch.cuda.manual_seed_all(0)

    alpha = TestTensor(tuple(), None, dtype, device)
    beta = TestTensor(tuple(), None, dtype, device)
    A = TestTensor((n, n), a_stride, dtype, device)
    x = TestTensor((n,), x_stride, dtype, device)
    y = TestTensor((n,), y_stride, dtype, device)

    y_ref = hemv(
        alpha.torch_tensor(),
        A.torch_tensor(),
        x.torch_tensor(),
        beta.torch_tensor(),
        y.torch_tensor(),
        uplo,
    )
    y.update_torch_tensor(y_ref)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateHemvDescriptor(
            handle,
            ctypes.byref(descriptor),
            uplo,
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
        LIBINFINIOP.infiniopGetHemvWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    def lib_hemv():
        check_error(
            LIBINFINIOP.infiniopHemv(
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

    lib_hemv()

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
            lambda: hemv(
                alpha.torch_tensor(),
                A.torch_tensor(),
                x.torch_tensor(),
                beta.torch_tensor(),
                y.torch_tensor(),
                uplo,
            ),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
        profile_operation(
            "    lib", lambda: lib_hemv(), device, NUM_PRERUN, NUM_ITERATIONS
        )

    check_error(LIBINFINIOP.infiniopDestroyHemvDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
