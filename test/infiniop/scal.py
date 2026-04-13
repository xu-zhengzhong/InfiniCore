import torch
import ctypes
from ctypes import c_uint64, c_float, c_double, c_int16
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
# Format: (shape, x_stride, alpha)
_TEST_CASES = [
    ((13,), None, 2.5),
    ((13,), (10,), 2.5),
    ((5632,), None, 2.5),
    ((5632,), (5,), 2.5),
    ((16,), (4,), 2.5),
    ((5632,), (32,), 2.5),
]

# Data types used for testing
_TENSOR_DTYPES = [
    InfiniDtype.F16,
    InfiniDtype.F32,
    # InfiniDtype.F64,
    InfiniDtype.BF16,
]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    InfiniDtype.F64: {"atol": 1e-15, "rtol": 1e-15},
    InfiniDtype.BF16: {"atol": 5e-3, "rtol": 1e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def scal(x, alpha):
    x.mul_(alpha)


def _alpha_ptr(alpha, dtype):
    if dtype == InfiniDtype.F32:
        holder = c_float(alpha)
    elif dtype == InfiniDtype.F64:
        holder = c_double(alpha)
    elif dtype == InfiniDtype.F16:
        bits = torch.tensor(alpha, dtype=torch.float16).view(torch.int16).item()
        holder = c_int16(bits)
    elif dtype == InfiniDtype.BF16:
        bits = torch.tensor(alpha, dtype=torch.bfloat16).view(torch.int16).item()
        holder = c_int16(bits)
    else:
        raise ValueError(f"Unsupported dtype for alpha: {dtype}")

    return holder, ctypes.byref(holder)


def test(
    handle,
    device,
    shape,
    x_stride=None,
    alpha=2.5,
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

    # Compute PyTorch reference
    scal(x.torch_tensor(), alpha)

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
    alpha_holder, alpha_ptr = _alpha_ptr(alpha, dtype)

    def lib_scal():
        check_error(
            LIBINFINIOP.infiniopScal(
                descriptor,
                workspace.data(),
                workspace.size(),
                x.data(),
                alpha_ptr,
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
        profile_operation("PyTorch", lambda: scal(x.torch_tensor(), alpha), device, NUM_PRERUN, NUM_ITERATIONS)
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