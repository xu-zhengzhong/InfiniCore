import torch
import ctypes
from ctypes import c_uint64
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
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
# Format: (shape, x_stride)
_TEST_CASES = [
    ((13, 4), None),
    ((13, 4), (10, 1)),
    ((13, 4, 4), None),
    ((13, 4, 4), (20, 4, 1)),
    ((16, 5632), None),
    ((16, 5632), (13312, 1)),
    ((13, 16, 2), (128, 4, 1)),
    ((4, 4, 5632), None),
    ((4, 4, 5632), (45056, 5632, 1)),
]

# Data types used for testing
_TENSOR_DTYPES = [
    InfiniDtype.F16,
    InfiniDtype.F32,
    InfiniDtype.BF16,
]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    InfiniDtype.BF16: {"atol": 1e-3, "rtol": 1e-3},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def scal(x, alpha):
    x.mul_(alpha)


def test(
    handle,
    device,
    shape,
    x_stride=None,
    dtype=torch.float16,
    sync=None,
):
    x = TestTensor(shape, x_stride, dtype, device)

    # In-place broadcast target is structurally invalid for scal
    if x.is_broadcast():
        return

    print(
        f"Testing Scal on {InfiniDeviceNames[device]} with shape:{shape} x_stride:{x_stride} "
        f"dtype:{InfiniDtypeNames[dtype]}"
    )

    # Generate a scalar multiplier
    # Use integers for int dtypes, floats for float/half dtypes
    if dtype in [InfiniDtype.I32, InfiniDtype.I64, InfiniDtype.I16, InfiniDtype.I8]:
        alpha_val = 3
    else:
        alpha_val = 2.5

    # Create a 0-D tensor for alpha on the same device to extract a valid C data pointer
    alpha_tensor = torch.tensor(
        alpha_val, dtype=x.torch_tensor().dtype, device=x.torch_tensor().device
    )

    # Compute PyTorch reference
    scal(x.torch_tensor(), alpha_val)

    if sync is not None:
        sync()

    # Create Descriptor
    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateScalDescriptor(
            handle,
            ctypes.byref(descriptor),
            x.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    x.destroy_desc()

    # Allocate Workspace
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetScalWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, x.device)

    # Execute C library op
    def lib_scal():
        check_error(
            LIBINFINIOP.infiniopScal(
                descriptor,
                workspace.data(),
                workspace.size(),
                x.data(),
                alpha_tensor.data_ptr(),  # Pass the scalar pointer
                None,
            )
        )

    lib_scal()

    # Compare results
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(x.actual_tensor(), x.torch_tensor(), atol=atol, rtol=rtol)
    
    assert torch.allclose(x.actual_tensor(), x.torch_tensor(), atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: scal(x.torch_tensor(), alpha_val), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_scal(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
        
    check_error(LIBINFINIOP.infiniopDestroyScalDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")