import torch
import ctypes
import math
from ctypes import c_uint64, c_uint32, c_int32, c_float, c_double
from libinfiniop import (
    LIBINFINIOP,
    TestTensor,
    get_test_devices,
    check_error,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
    TestWorkspace,
    InfiniDtype,
    InfiniDtypeNames,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
)

# ==============================================================================
# Configuration
# ==============================================================================
# Format: (shape, x_stride)
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
    # InfiniDtype.F64,
]

# asum returns a floating-point sum, so we need actual tolerances.
# PyTorch and the C library might accumulate floats in different orders,
# so slight deviations are expected.
_TOLERANCE_MAP = {
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-4},
    InfiniDtype.F64: {"atol": 1e-5, "rtol": 1e-4},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000

def torch_asum(x):
    # BLAS asum calculates the sum of the absolute values of a vector
    return torch.sum(x.abs())

def test(
    handle,
    device,
    shape,
    x_stride=None,
    dtype=torch.float16,
    sync=None,
):
    x = TestTensor(shape, x_stride, dtype, device)
    
    # asum returns a scalar float. We use c_float to capture the result.
    if dtype is InfiniDtype.F32:
        result = c_float(0.0)
    else:
        result = c_double(0.0)

    print(
        f"Testing Asum on {InfiniDeviceNames[device]} with shape:{shape} x_stride:{x_stride} "
        f"dtype:{InfiniDtypeNames[dtype]}"
    )

    # Compute PyTorch reference
    expected_val = torch_asum(x.torch_tensor()).item()

    if sync is not None:
        sync()

    # Create Descriptor
    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateAsumDescriptor(
            handle,
            ctypes.byref(descriptor),
            x.descriptor,
        )
    )

    # Invalidate descriptor to ensure kernel uses passed-in pointers/logic
    x.destroy_desc()

    # Allocate Workspace
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetAsumWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, x.device)
    
    # Execute C library op
    def lib_asum():
        check_error(
            LIBINFINIOP.infiniopAsum(
                descriptor,
                workspace.data(),
                workspace.size(),
                x.data(),
                ctypes.byref(result),
                None,
            )
        )

    lib_asum()

    # Compare results
    actual_val = result.value
    
    # Fetch tolerances dynamically based on dtype
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    
    if DEBUG:
        print(f"Expected Sum: {expected_val:.6f}, Actual Sum: {actual_val:.6f}")
    
    assert math.isclose(actual_val, expected_val, rel_tol=rtol, abs_tol=atol), \
        f"Value mismatch: actual={actual_val} != expected={expected_val} (atol={atol}, rtol={rtol})"

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: torch_asum(x.torch_tensor()), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_asum(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
        
    check_error(LIBINFINIOP.infiniopDestroyAsumDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")