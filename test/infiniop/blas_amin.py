import torch
import ctypes
from ctypes import c_uint64, c_int32
from libinfiniop import (
    LIBINFINIOP,
    TestTensor,
    get_test_devices,
    check_error,
    test_operator,
    get_args,
    profile_operation,
    TestWorkspace,
    InfiniDtype,
    InfiniDtypeNames,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
)

_TEST_CASES = [
    ((13,), None),
    ((13,), (10,)),
    ((5632,), None),
    ((5632,), (5,)),
    ((16,), (4,)),
    ((5632,), (32,)),
]

_TENSOR_DTYPES = [
    InfiniDtype.F32,
]

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def torch_amin_index(x):
    return torch.argmin(x.abs())


def test(
    handle,
    device,
    shape,
    x_stride=None,
    dtype=torch.float16,
    sync=None,
):
    x = TestTensor(shape, x_stride, dtype, device)
    result = c_int32(0)

    print(
        f"Testing BlasAmin on {InfiniDeviceNames[device]} with shape:{shape} x_stride:{x_stride} "
        f"dtype:{InfiniDtypeNames[dtype]}"
    )

    expected_idx = torch_amin_index(x.torch_tensor()).item() + 1

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateBlasAminDescriptor(
            handle,
            ctypes.byref(descriptor),
            x.descriptor,
        )
    )

    x.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetBlasAminWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, x.device)

    def lib_amin():
        check_error(
            LIBINFINIOP.infiniopBlasAmin(
                descriptor,
                workspace.data(),
                workspace.size(),
                x.data(),
                ctypes.byref(result),
                None,
            )
        )

    lib_amin()

    actual_idx = result.value
    if DEBUG:
        print(f"Expected Index: {expected_idx}, Actual Index: {actual_idx}")

    assert actual_idx == expected_idx, f"Index mismatch: {actual_idx} != {expected_idx}"

    if PROFILE:
        profile_operation("PyTorch", lambda: torch_amin_index(x.torch_tensor()), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_amin(), device, NUM_PRERUN, NUM_ITERATIONS)

    check_error(LIBINFINIOP.infiniopDestroyBlasAminDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
