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
    (1, 1, None, None, None),
    (1, 5, None, None, None),
    (1, 17, None, None, (2,)),
    (1, 33, (1, 40), (2,), None),
    (1, 128, None, (3,), (2,)),
    (1, 1024, None, None, None),
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


def _triangle_update(A, update, uplo):
    if uplo == 0:
        out = torch.triu(A + update) + torch.tril(A, diagonal=-1)
    else:
        out = torch.tril(A + update) + torch.triu(A, diagonal=1)

    idx = torch.arange(A.shape[0], device=A.device)
    out[idx, idx] = out[idx, idx].real.to(out.dtype)
    return out


def _triangle_update_mlu(A, update_real, update_imag, uplo):
    A_real = A.real
    A_imag = A.imag

    if uplo == 0:
        out_real = torch.triu(A_real + update_real) + torch.tril(A_real, diagonal=-1)
        out_imag = torch.triu(A_imag + update_imag) + torch.tril(A_imag, diagonal=-1)
    else:
        out_real = torch.tril(A_real + update_real) + torch.triu(A_real, diagonal=1)
        out_imag = torch.tril(A_imag + update_imag) + torch.triu(A_imag, diagonal=1)

    idx = torch.arange(A.shape[0], device=A.device)
    out_imag[idx, idx] = 0

    out = torch.empty_like(A)
    out.real.copy_(out_real)
    out.imag.copy_(out_imag)
    return out


def her2(alpha, A, x, y, uplo):
    if A.device.type == "mlu":
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
        return _triangle_update_mlu(A, update_real, update_imag, uplo)

    update = alpha * torch.outer(x, y.conj()) + alpha.conj() * torch.outer(y, x.conj())
    return _triangle_update(A, update, uplo)


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
        f"Testing Her2 on {InfiniDeviceNames[device]} with uplo:{uplo} n:{n} "
        f"a_stride:{a_stride} x_stride:{x_stride} y_stride:{y_stride} "
        f"dtype:{InfiniDtypeNames[dtype]}"
    )

    if a_stride is None:
        a_stride = (1, n)

    torch.manual_seed(0)
    if device != 0:
        torch.cuda.manual_seed_all(0)

    alpha = TestTensor(tuple(), None, dtype, device)
    x = TestTensor((n,), x_stride, dtype, device)
    y = TestTensor((n,), y_stride, dtype, device)
    A = TestTensor((n, n), a_stride, dtype, device)

    A_ref = her2(
        alpha.torch_tensor(),
        A.torch_tensor(),
        x.torch_tensor(),
        y.torch_tensor(),
        uplo,
    )
    A.update_torch_tensor(A_ref)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateHer2Descriptor(
            handle,
            ctypes.byref(descriptor),
            uplo,
            alpha.descriptor,
            x.descriptor,
            y.descriptor,
            A.descriptor,
        )
    )

    for tensor in [alpha, x, y, A]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetHer2WorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    def lib_her2():
        check_error(
            LIBINFINIOP.infiniopHer2(
                descriptor,
                workspace.data(),
                workspace_size.value,
                alpha.data(),
                x.data(),
                y.data(),
                A.data(),
                None,
            )
        )

    lib_her2()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(A.actual_tensor().real, A.torch_tensor().real, atol=atol, rtol=rtol)
        debug(A.actual_tensor().imag, A.torch_tensor().imag, atol=atol, rtol=rtol)
    assert torch.allclose(
        A.actual_tensor().real, A.torch_tensor().real, atol=atol, rtol=rtol
    ) and torch.allclose(
        A.actual_tensor().imag, A.torch_tensor().imag, atol=atol, rtol=rtol
    )

    if PROFILE:
        profile_operation(
            "PyTorch",
            lambda: her2(
                alpha.torch_tensor(),
                A.torch_tensor(),
                x.torch_tensor(),
                y.torch_tensor(),
                uplo,
            ),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
        profile_operation(
            "    lib", lambda: lib_her2(), device, NUM_PRERUN, NUM_ITERATIONS
        )

    check_error(LIBINFINIOP.infiniopDestroyHer2Descriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
