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

_UPLO_TRANS_CASES = [(uplo, trans) for uplo in (0, 1) for trans in (0, 2)]

_DENSE_SHAPES = [
    # n, k
    (1, 1),
    (1, 7),
    (2, 1),
    (2, 2),
    (7, 3),
    (8, 8),
    (9, 17),
    (17, 9),
    (31, 32),
    (32, 31),
    (33, 64),
    (64, 33),
    (127, 128),
    (128, 127),
    (256, 256),
    (1024, 32),
    (32, 1024),
]

_STRIDED_SHAPES = [
    # n, k, a_stride_n, a_stride_c, b_stride_n, b_stride_c, c_stride
    (17, 9, (1, 24), (1, 16), (1, 28), (1, 18), None),
    (31, 32, (1, 40), (1, 48), (1, 44), (1, 52), (1, 40)),
]

_TEST_CASES = [
    # uplo, trans, n, k, a_stride, b_stride, c_stride
    (uplo, trans, n, k, None, None, None)
    for n, k in _DENSE_SHAPES
    for uplo, trans in _UPLO_TRANS_CASES
] + [
    (
        uplo,
        trans,
        n,
        k,
        a_stride_n if trans == 0 else a_stride_c,
        b_stride_n if trans == 0 else b_stride_c,
        c_stride,
    )
    for n, k, a_stride_n, a_stride_c, b_stride_n, b_stride_c, c_stride in _STRIDED_SHAPES
    for uplo, trans in _UPLO_TRANS_CASES
]

_TENSOR_DTYPES = [
    InfiniDtype.C64,
    # InfiniDtype.C128,
]

_TOLERANCE_MAP = {
    InfiniDtype.C64: {"atol": 2e-4, "rtol": 2e-4},
    InfiniDtype.C128: {"atol": 1e-9, "rtol": 1e-9},
}

_REAL_DTYPE = {
    InfiniDtype.C64: InfiniDtype.F32,
    InfiniDtype.C128: InfiniDtype.F64,
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def _default_col_major_stride(rows):
    return (1, rows)


def _triangle_update(C, update, uplo):
    update = update.clone()
    update.diagonal().imag.zero_()
    if uplo == 0:
        return torch.triu(update) + torch.tril(C, diagonal=-1)
    return torch.tril(update) + torch.triu(C, diagonal=1)


def _triangle_update_mlu(C, update_real, update_imag, uplo):
    idx = torch.arange(C.shape[0], device=C.device)
    update_imag[idx, idx] = 0

    if uplo == 0:
        out_real = torch.triu(update_real) + torch.tril(C.real, diagonal=-1)
        out_imag = torch.triu(update_imag) + torch.tril(C.imag, diagonal=-1)
    else:
        out_real = torch.tril(update_real) + torch.triu(C.real, diagonal=1)
        out_imag = torch.tril(update_imag) + torch.triu(C.imag, diagonal=1)

    out = torch.empty_like(C)
    out.real.copy_(out_real)
    out.imag.copy_(out_imag)
    return out


def _complex_mul(real_l, imag_l, real_r, imag_r):
    return real_l * real_r - imag_l * imag_r, real_l * imag_r + imag_l * real_r


def _her2k_update_mlu(alpha, A, B, beta, C, trans):
    A_real = A.real
    A_imag = A.imag
    B_real = B.real
    B_imag = B.imag
    alpha_real = alpha.real
    alpha_imag = alpha.imag

    if trans == 0:
        ab_real = torch.mm(A_real, B_real.t()) + torch.mm(A_imag, B_imag.t())
        ab_imag = torch.mm(A_imag, B_real.t()) - torch.mm(A_real, B_imag.t())
        ba_real = torch.mm(B_real, A_real.t()) + torch.mm(B_imag, A_imag.t())
        ba_imag = torch.mm(B_imag, A_real.t()) - torch.mm(B_real, A_imag.t())
    else:
        ab_real = torch.mm(A_real.t(), B_real) + torch.mm(A_imag.t(), B_imag)
        ab_imag = torch.mm(A_real.t(), B_imag) - torch.mm(A_imag.t(), B_real)
        ba_real = torch.mm(B_real.t(), A_real) + torch.mm(B_imag.t(), A_imag)
        ba_imag = torch.mm(B_real.t(), A_imag) - torch.mm(B_imag.t(), A_real)

    update_ab_real, update_ab_imag = _complex_mul(
        alpha_real, alpha_imag, ab_real, ab_imag
    )
    update_ba_real, update_ba_imag = _complex_mul(
        alpha_real, -alpha_imag, ba_real, ba_imag
    )
    update_real = update_ab_real + update_ba_real + beta * C.real
    update_imag = update_ab_imag + update_ba_imag + beta * C.imag
    return update_real, update_imag


def her2k(alpha, A, B, beta, C, trans, uplo):
    if A.device.type == "mlu":
        update_real, update_imag = _her2k_update_mlu(alpha, A, B, beta, C, trans)
        return _triangle_update_mlu(C, update_real, update_imag, uplo)

    product = (
        alpha * torch.mm(A, B.mH) + alpha.conj() * torch.mm(B, A.mH)
        if trans == 0
        else alpha * torch.mm(A.mH, B) + alpha.conj() * torch.mm(B.mH, A)
    )
    update = product + beta * C
    return _triangle_update(C, update, uplo)


def test(
    handle,
    device,
    uplo,
    trans,
    n,
    k,
    a_stride=None,
    b_stride=None,
    c_stride=None,
    dtype=InfiniDtype.C64,
    sync=None,
):
    print(
        f"Testing Her2k on {InfiniDeviceNames[device]} with uplo:{uplo} trans:{trans} n:{n} k:{k} "
        f"a_stride:{a_stride} b_stride:{b_stride} c_stride:{c_stride} dtype:{InfiniDtypeNames[dtype]}"
    )

    ab_shape = (n, k) if trans == 0 else (k, n)
    if a_stride is None:
        a_stride = _default_col_major_stride(ab_shape[0])
    if b_stride is None:
        b_stride = _default_col_major_stride(ab_shape[0])
    if c_stride is None:
        c_stride = _default_col_major_stride(n)

    torch.manual_seed(0)
    if device != 0:
        torch.cuda.manual_seed_all(0)

    real_dtype = _REAL_DTYPE[dtype]
    alpha = TestTensor(tuple(), None, dtype, device)
    beta = TestTensor(tuple(), None, real_dtype, device)
    A = TestTensor(ab_shape, a_stride, dtype, device, scale=0.5)
    B = TestTensor(ab_shape, b_stride, dtype, device, scale=0.5)
    C = TestTensor((n, n), c_stride, dtype, device)

    C_ref = her2k(
        alpha.torch_tensor(),
        A.torch_tensor(),
        B.torch_tensor(),
        beta.torch_tensor(),
        C.torch_tensor(),
        trans,
        uplo,
    )
    C.update_torch_tensor(C_ref)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateHer2kDescriptor(
            handle,
            ctypes.byref(descriptor),
            uplo,
            trans,
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
        LIBINFINIOP.infiniopGetHer2kWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    def lib_her2k():
        check_error(
            LIBINFINIOP.infiniopHer2k(
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

    lib_her2k()

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
            lambda: her2k(
                alpha.torch_tensor(),
                A.torch_tensor(),
                B.torch_tensor(),
                beta.torch_tensor(),
                C.torch_tensor(),
                trans,
                uplo,
            ),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
        profile_operation(
            "    lib", lambda: lib_her2k(), device, NUM_PRERUN, NUM_ITERATIONS
        )

    check_error(LIBINFINIOP.infiniopDestroyHer2kDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
