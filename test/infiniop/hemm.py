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

_SIDE_UPLO_CASES = [(side, uplo) for side in (0, 1) for uplo in (0, 1)]

_DENSE_SHAPES = [
    # m, n
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2),
    (3, 5),
    (5, 3),
    (8, 8),
    (9, 17),
    (17, 9),
    (31, 32),
    (32, 31),
    (65, 65),
    (127, 128),
    (128, 127),
    (256, 256),
    (512, 512),
    (1024, 1024),
    (4096, 1),
    (1, 4096),
    (4096, 2),
    (2, 4096),
    (4097, 1),
    (1, 4097),
]

_STRIDED_SHAPES = [
    # m, n, a_stride_left, a_stride_right, b_stride, c_stride
    (17, 9, (1, 24), (1, 16), None, None),
    (31, 32, (1, 40), (1, 48), (1, 36), (1, 40)),
    (4096, 2, (1, 4104), (1, 4), (1, 4100), (1, 4104)),
    (2, 4096, (1, 4), (1, 4104), (1, 4), (1, 4)),
    (4097, 1, (1, 4104), (1, 8), (1, 4100), (1, 4104)),
    (1, 4097, (1, 8), (1, 4104), (1, 8), (1, 8)),
]

_TEST_CASES = [
    # side, uplo, m, n, a_stride, b_stride, c_stride
    (side, uplo, m, n, None, None, None)
    for m, n in _DENSE_SHAPES
    for side, uplo in _SIDE_UPLO_CASES
] + [
    (
        side,
        uplo,
        m,
        n,
        a_stride_left if side == 0 else a_stride_right,
        b_stride,
        c_stride,
    )
    for m, n, a_stride_left, a_stride_right, b_stride, c_stride in _STRIDED_SHAPES
    for side, uplo in _SIDE_UPLO_CASES
]

_TENSOR_DTYPES = [
    InfiniDtype.C64,
    # InfiniDtype.C128,
]

_TOLERANCE_MAP = {
    InfiniDtype.C64: {"atol": 3e-3, "rtol": 5e-4},
    InfiniDtype.C128: {"atol": 1e-9, "rtol": 1e-9},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def _full_from_triangle(A, uplo):
    if uplo == 0:
        tri = torch.triu(A)
        matrix = tri + torch.triu(A, diagonal=1).mH
    else:
        tri = torch.tril(A)
        matrix = tri + torch.tril(A, diagonal=-1).mH
    diag = torch.diagonal(matrix)
    diag.copy_(diag.real.clone())
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


def _hemm_mlu(alpha, A, B, beta, C, side, uplo):
    A_real, A_imag = _full_from_triangle_mlu(A, uplo)
    B_real = B.real
    B_imag = B.imag

    if side == 0:
        product_real = torch.mm(A_real, B_real) - torch.mm(A_imag, B_imag)
        product_imag = torch.mm(A_real, B_imag) + torch.mm(A_imag, B_real)
    else:
        product_real = torch.mm(B_real, A_real) - torch.mm(B_imag, A_imag)
        product_imag = torch.mm(B_real, A_imag) + torch.mm(B_imag, A_real)

    alpha_real = alpha.real
    alpha_imag = alpha.imag
    beta_real = beta.real
    beta_imag = beta.imag
    C_real = C.real
    C_imag = C.imag

    out_real = alpha_real * product_real - alpha_imag * product_imag
    out_real = out_real + beta_real * C_real - beta_imag * C_imag
    out_imag = alpha_real * product_imag + alpha_imag * product_real
    out_imag = out_imag + beta_real * C_imag + beta_imag * C_real

    out = torch.empty_like(C)
    out.real.copy_(out_real)
    out.imag.copy_(out_imag)
    return out


def hemm(alpha, A, B, beta, C, side, uplo):
    if A.device.type == "mlu":
        return _hemm_mlu(alpha, A, B, beta, C, side, uplo)

    matrix = _full_from_triangle(A, uplo)
    product = torch.mm(matrix, B) if side == 0 else torch.mm(B, matrix)
    return alpha * product + beta * C


def _default_col_major_stride(rows):
    return (1, rows)


def test(
    handle,
    device,
    side,
    uplo,
    m,
    n,
    a_stride=None,
    b_stride=None,
    c_stride=None,
    dtype=InfiniDtype.C64,
    sync=None,
):
    print(
        f"Testing Hemm on {InfiniDeviceNames[device]} with side:{side} uplo:{uplo} m:{m} n:{n} "
        f"a_stride:{a_stride} b_stride:{b_stride} c_stride:{c_stride} dtype:{InfiniDtypeNames[dtype]}"
    )

    dim_a = m if side == 0 else n
    if a_stride is None:
        a_stride = _default_col_major_stride(dim_a)
    if b_stride is None:
        b_stride = _default_col_major_stride(m)
    if c_stride is None:
        c_stride = _default_col_major_stride(m)

    torch.manual_seed(0)
    if device != 0:
        torch.cuda.manual_seed_all(0)

    alpha = TestTensor(tuple(), None, dtype, device)
    beta = TestTensor(tuple(), None, dtype, device)
    A = TestTensor((dim_a, dim_a), a_stride, dtype, device)
    B = TestTensor((m, n), b_stride, dtype, device, scale=0.5)
    C = TestTensor((m, n), c_stride, dtype, device)

    c_ref = hemm(
        alpha.torch_tensor(),
        A.torch_tensor(),
        B.torch_tensor(),
        beta.torch_tensor(),
        C.torch_tensor(),
        side,
        uplo,
    )
    C.update_torch_tensor(c_ref)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateHemmDescriptor(
            handle,
            ctypes.byref(descriptor),
            side,
            uplo,
            alpha.descriptor,
            A.descriptor,
            B.descriptor,
            beta.descriptor,
            C.descriptor,
        )
    )

    for tensor in [alpha, beta, A, B, C]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetHemmWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    def lib_hemm():
        check_error(
            LIBINFINIOP.infiniopHemm(
                descriptor,
                workspace.data(),
                workspace_size.value,
                alpha.data(),
                A.data(),
                B.data(),
                beta.data(),
                C.data(),
                None,
            )
        )

    lib_hemm()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(C.actual_tensor().real, C.torch_tensor().real, atol=atol, rtol=rtol)
        debug(C.actual_tensor().imag, C.torch_tensor().imag, atol=atol, rtol=rtol)
    assert torch.allclose(
        C.actual_tensor().real, C.torch_tensor().real, atol=atol, rtol=rtol
    ) and torch.allclose(
        C.actual_tensor().imag, C.torch_tensor().imag, atol=atol, rtol=rtol
    )

    if PROFILE:
        profile_operation(
            "PyTorch",
            lambda: hemm(
                alpha.torch_tensor(),
                A.torch_tensor(),
                B.torch_tensor(),
                beta.torch_tensor(),
                C.torch_tensor(),
                side,
                uplo,
            ),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
        profile_operation(
            "    lib", lambda: lib_hemm(), device, NUM_PRERUN, NUM_ITERATIONS
        )

    check_error(LIBINFINIOP.infiniopDestroyHemmDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
