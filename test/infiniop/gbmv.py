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
    # trans, m, n, kl, ku, a_stride, x_stride, y_stride
    (0, 1, 1, 0, 0, None, None, None),
    (0, 1, 7, 0, 3, None, None, None),
    (0, 6, 1, 4, 0, None, None, None),
    (0, 5, 5, 0, 0, None, None, None),
    (0, 6, 6, 5, 5, None, None, None),
    (0, 9, 9, 8, 0, None, None, None),
    (0, 9, 9, 0, 8, None, None, None),
    (0, 4, 5, 1, 2, None, None, None),
    (0, 3, 11, 2, 4, None, (2,), None),
    (0, 11, 3, 4, 2, None, None, (2,)),
    (0, 7, 3, 3, 1, (1, 8), (2,), (3,)),
    (0, 8, 10, 2, 3, (1, 8), None, None),
    (0, 8, 10, 2, 3, (1, 16), (2,), (3,)),
    (0, 16, 17, 2, 2, None, (2,), (2,)),
    (0, 33, 65, 1, 4, None, None, None),
    (0, 65, 33, 4, 1, None, (2,), (2,)),
    (0, 16, 5632, 2, 2, None, (2,), (2,)),
    (0, 5632, 33, 4, 1, None, None, None),
    (0, 2048, 2560, 2, 3, (1, 8), None, None),
    (0, 256, 65535, 1, 1, (1, 8), None, None),
    (1, 1, 7, 0, 3, None, None, None),
    (1, 6, 1, 4, 0, None, None, None),
    (1, 5, 5, 0, 0, None, None, None),
    (1, 6, 6, 5, 5, None, None, None),
    (1, 4, 5, 1, 2, None, None, None),
    (1, 3, 11, 2, 4, None, None, (2,)),
    (1, 11, 3, 4, 2, None, (2,), None),
    (1, 7, 3, 3, 1, (1, 8), (2,), (3,)),
    (1, 8, 10, 2, 3, (1, 8), None, None),
    (1, 8, 10, 2, 3, (1, 16), (3,), (2,)),
    (1, 33, 65, 1, 4, None, (2,), (2,)),
    (1, 65, 33, 4, 1, None, None, None),
    (1, 16, 5632, 2, 2, None, (2,), (2,)),
    (1, 5632, 33, 4, 1, None, None, None),
    (1, 2048, 2560, 2, 3, (1, 8), None, None),
    (1, 256, 65535, 1, 1, (1, 8), None, None),
]

_TENSOR_DTYPES = [
    InfiniDtype.F32,
    # InfiniDtype.F64,
]

_TOLERANCE_MAP = {
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
    InfiniDtype.F64: {"atol": 1e-9, "rtol": 1e-9},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def _full_from_band(A, m, n, kl, ku):
    full = torch.zeros((m, n), dtype=A.dtype, device=A.device)
    for j in range(n):
        i_begin = max(0, j - ku)
        i_end = min(m, j + kl + 1)
        for i in range(i_begin, i_end):
            full[i, j] = A[ku + i - j, j]
    return full


def test(
    handle,
    device,
    trans,
    m,
    n,
    kl,
    ku,
    a_stride=None,
    x_stride=None,
    y_stride=None,
    dtype=torch.float32,
    sync=None,
):
    print(
        f"Testing Gbmv on {InfiniDeviceNames[device]} with trans:{trans} m:{m} n:{n} kl:{kl} ku:{ku} "
        f"a_stride:{a_stride} x_stride:{x_stride} y_stride:{y_stride} dtype:{InfiniDtypeNames[dtype]}"
    )

    if a_stride is None:
        a_stride = (1, kl + ku + 1)

    torch.manual_seed(0)
    if device != 0:
        torch.cuda.manual_seed_all(0)

    alpha = TestTensor(tuple(), None, dtype, device)
    beta = TestTensor(tuple(), None, dtype, device)
    A = TestTensor((kl + ku + 1, n), a_stride, dtype, device)
    x_len = n if trans == 0 else m
    y_len = m if trans == 0 else n
    x = TestTensor((x_len,), x_stride, dtype, device)
    y = TestTensor((y_len,), y_stride, dtype, device)

    full = _full_from_band(A.torch_tensor(), m, n, kl, ku)
    matrix = full if trans == 0 else full.t()
    y_ref = alpha.torch_tensor() * torch.mv(matrix, x.torch_tensor())
    y_ref = y_ref + beta.torch_tensor() * y.torch_tensor()
    y.update_torch_tensor(y_ref)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateGbmvDescriptor(
            handle,
            ctypes.byref(descriptor),
            trans,
            kl,
            ku,
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
        LIBINFINIOP.infiniopGetGbmvWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    def lib_gbmv():
        check_error(
            LIBINFINIOP.infiniopGbmv(
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

    lib_gbmv()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)

    if PROFILE:
        profile_operation(
            "PyTorch",
            lambda: torch.mv(matrix, x.torch_tensor())
            .mul_(alpha.torch_tensor())
            .add_(y.torch_tensor(), alpha=beta.torch_tensor().item()),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
        profile_operation(
            "    lib", lambda: lib_gbmv(), device, NUM_PRERUN, NUM_ITERATIONS
        )

    check_error(LIBINFINIOP.infiniopDestroyGbmvDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
