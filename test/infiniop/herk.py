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
    # n, k, a_stride_n, a_stride_c, c_stride
    (17, 9, (1, 24), (1, 16), None),
    (31, 32, (1, 40), (1, 48), (1, 40)),
]

_TEST_CASES = [
    # uplo, trans, n, k, a_stride, c_stride
    (uplo, trans, n, k, None, None)
    for n, k in _DENSE_SHAPES
    for uplo, trans in _UPLO_TRANS_CASES
] + [
    (
        uplo,
        trans,
        n,
        k,
        a_stride_n if trans == 0 else a_stride_c,
        c_stride,
    )
    for n, k, a_stride_n, a_stride_c, c_stride in _STRIDED_SHAPES
    for uplo, trans in _UPLO_TRANS_CASES
]

_TENSOR_DTYPES = [
    InfiniDtype.C64,
    # InfiniDtype.C128,
]

_TOLERANCE_MAP = {
    InfiniDtype.C64: {"atol": 1e-4, "rtol": 1e-4},
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


def _herk_update_mlu(alpha, A, beta, C, trans):
    A_real = A.real
    A_imag = A.imag

    if trans == 0:
        product_real = torch.mm(A_real, A_real.t()) + torch.mm(A_imag, A_imag.t())
        product_imag = torch.mm(A_imag, A_real.t()) - torch.mm(A_real, A_imag.t())
    else:
        product_real = torch.mm(A_real.t(), A_real) + torch.mm(A_imag.t(), A_imag)
        product_imag = torch.mm(A_real.t(), A_imag) - torch.mm(A_imag.t(), A_real)

    update_real = alpha * product_real + beta * C.real
    update_imag = alpha * product_imag + beta * C.imag
    return update_real, update_imag


def herk(alpha, A, beta, C, trans, uplo):
    if A.device.type == "mlu":
        update_real, update_imag = _herk_update_mlu(alpha, A, beta, C, trans)
        return _triangle_update_mlu(C, update_real, update_imag, uplo)

    product = torch.mm(A, A.mH) if trans == 0 else torch.mm(A.mH, A)
    update = alpha * product + beta * C
    return _triangle_update(C, update, uplo)


def test(
    handle,
    device,
    uplo,
    trans,
    n,
    k,
    a_stride=None,
    c_stride=None,
    dtype=InfiniDtype.C64,
    sync=None,
):
    print(
        f"Testing Herk on {InfiniDeviceNames[device]} with uplo:{uplo} trans:{trans} n:{n} k:{k} "
        f"a_stride:{a_stride} c_stride:{c_stride} dtype:{InfiniDtypeNames[dtype]}"
    )

    a_shape = (n, k) if trans == 0 else (k, n)
    if a_stride is None:
        a_stride = _default_col_major_stride(a_shape[0])
    if c_stride is None:
        c_stride = _default_col_major_stride(n)

    torch.manual_seed(0)
    if device != 0:
        torch.cuda.manual_seed_all(0)

    real_dtype = _REAL_DTYPE[dtype]
    alpha = TestTensor(tuple(), None, real_dtype, device)
    beta = TestTensor(tuple(), None, real_dtype, device)
    A = TestTensor(a_shape, a_stride, dtype, device, scale=0.5)
    C = TestTensor((n, n), c_stride, dtype, device)

    C_ref = herk(
        alpha.torch_tensor(),
        A.torch_tensor(),
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
        LIBINFINIOP.infiniopCreateHerkDescriptor(
            handle,
            ctypes.byref(descriptor),
            uplo,
            trans,
            alpha.descriptor,
            A.descriptor,
            beta.descriptor,
            C.descriptor,
        )
    )

    for tensor in [alpha, beta, A, C]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetHerkWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    def lib_herk():
        check_error(
            LIBINFINIOP.infiniopHerk(
                descriptor,
                workspace.data(),
                workspace_size.value,
                alpha.data(),
                A.data(),
                beta.data(),
                C.data(),
                None,
            )
        )

    lib_herk()

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
            lambda: herk(
                alpha.torch_tensor(),
                A.torch_tensor(),
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
            "    lib", lambda: lib_herk(), device, NUM_PRERUN, NUM_ITERATIONS
        )

    check_error(LIBINFINIOP.infiniopDestroyHerkDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
