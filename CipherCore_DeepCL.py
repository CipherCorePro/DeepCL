# -*- coding: utf-8 -*-
"""
Python Driver for C/OpenCL Brain-Inspired Network Model.

This script serves as the main Python interface for interacting with a compiled
C/OpenCL library (`CipherCore_OpenCl.dll` or `libsimulated_driver.so`).
It focuses on character-level text generation using a model architecture that
includes an Embedding layer, a custom Bio-Inspired Associative Layer, and a
standard Linear output layer.

Key functionalities include:
- GPU detection and selection using PyOpenCL.
- Initialization and management of GPU resources via ctypes calls to the C library.
- Data preprocessing for character-level tasks (vocabulary creation, sequence generation).
- Definition of model layers (Embedding, BioInspired, Linear) wrapping C/OpenCL kernels.
- Implementation of the training loop with Adam optimizer, gradient clipping,
  learning rate scheduling, and checkpointing.
- Calculation of loss (Cross-Entropy) and accuracy.
- Text generation (sampling) based on a trained model.

Dependencies:
- Python 3.x
- NumPy: For numerical operations and data handling.
- ctypes: For interfacing with the C library.
- PyOpenCL (optional but recommended): For GPU detection and information.
- pickle: For saving and loading model checkpoints and vocabulary.
- The compiled C/OpenCL library (`CipherCore_OpenCl.dll` or `libsimulated_driver.so`)
  must be present in the 'CL' subdirectory relative to this script.
- An input text file (e.g., `mini_input.txt`) in the 'mini_data' subdirectory.

Setup:
1. Compile the C/OpenCL code located in the 'CL' directory to produce the
   shared library (`.dll` for Windows, `.so` for Linux).
2. Ensure the compiled library is placed within the 'CL' subdirectory.
3. Place your training text data in `mini_data/mini_input.txt`.
4. Run this Python script.
"""

import ctypes
import numpy as np
import os
import platform
import time
import math
import pickle
from typing import Optional, Tuple, Generator, Dict, Any, List, Mapping, Union
import sys
from collections import Counter

# Attempt to import PyOpenCL for GPU detection, fail gracefully if not installed.
try:
    import pyopencl as cl
except ImportError:
    print("FEHLER: PyOpenCL nicht gefunden. GPU-Auswahl eingeschränkt.")
    print("        Bitte installieren für volle Funktionalität: pip install pyopencl")
    cl = None # Set cl to None to allow checks later

# --- Constants and Configuration ---

# Data types used throughout the model and for C interop.
FP_TYPE: type = np.float32
FP_TYPE_C: type = ctypes.c_float
INT_TYPE: type = np.int32
INT_TYPE_C: type = ctypes.c_int

# Default GPU index; may be overridden by user selection.
GPU_INDEX: int = 0
# Flag to enable/disable detailed print statements for debugging.
DEBUG_PRINTS: bool = False
# Small epsilon value for numerical stability in Adam optimizer denominator.
ADAM_EPS: float = 1e-8
# Special index used for padding target sequences in batches.
PAD_INDEX: int = -1

# --- Training Hyperparameters ---
INITIAL_LEARNING_RATE: float = 1e-4       # Starting learning rate for the optimizer.
WEIGHT_DECAY: float = 0.001             # L2 regularization factor for weights (excluding biases/embeddings).
NUM_EPOCHS: int = 50                    # Total number of training epochs.
BATCH_SIZE: int = 64                    # Number of sequences per training batch.
GRADIENT_CLIP_VALUE: Optional[float] = 1.0 # Max L2 norm for gradients (None to disable clipping).
LR_DECAY_STEP: int = 2                  # Number of epochs after which the learning rate is decayed.
LR_DECAY_GAMMA: float = 0.5             # Multiplicative factor for learning rate decay.

# --- Model Hyperparameters ---
SEQ_LEN: int = 64                       # Length of input/target character sequences.
EMBEDDING_DIM: int = 128                # Dimensionality of character embeddings (input to BioLayer).
HIDDEN_DIM: int = 384                   # Dimensionality of the hidden state in the BioLayer.
VOCAB_SIZE: int = -1                    # Size of the character vocabulary (determined from data).
NUM_TOKEN_PROTOTYPES: int = 72          # Number of dynamic token prototypes in the BioLayer.
# BioLayer specific parameters
HEBBIAN_LR: float = 0.01                # Learning rate for the Hebbian weight updates.
SPIKE_THRESHOLD: float = 0.45           # Activation threshold for generating 'spikes'.
PROTOTYPE_LR: float = 0.005             # Learning rate for updating token prototypes.
USE_GPU_PROTOTYPE_UPDATE: bool = True   # Whether to use the (faster) GPU kernel for prototype updates if available.

# --- Path Configuration ---
script_dir: str = os.path.dirname(os.path.abspath(__file__))
# Directory containing the compiled C/OpenCL library and kernels.
cl_dir: str = os.path.join(script_dir, "CL")
# Directory for storing input data and processed datasets.
data_dir: str = os.path.join(script_dir, "mini_data")
# Directory for saving model checkpoints.
checkpoint_dir: str = os.path.join(script_dir, "mini_checkpoints")

# Create directories if they don't exist.
os.makedirs(data_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# Input text file path (user should place their data here).
input_text_file: str = os.path.join(data_dir, "mini_input.txt")
# Path for the processed NumPy dataset file.
processed_data_file: str = os.path.join(data_dir, "mini_char_dataset.npz")
# Vocabulary file path (associated with the processed data).
vocab_file: str = processed_data_file + "_mini_vocab.pkl"

# Checkpoint filename pattern based on key hyperparameters.
checkpoint_filename: str = f"model_char_emb{EMBEDDING_DIM}_h{HIDDEN_DIM}_t{NUM_TOKEN_PROTOTYPES}.pkl"
# Path for the latest checkpoint.
checkpoint_path: str = os.path.join(checkpoint_dir, checkpoint_filename)
# Path for the checkpoint with the best validation loss.
best_checkpoint_path: str = os.path.join(checkpoint_dir, f"best_{checkpoint_filename}")

# --- Dynamic Library Loading Setup (Windows Specific) ---
# On Windows with Python 3.8+, explicitly add the CL directory to the DLL search path
# to ensure dependencies (like OpenCL.dll) alongside the main library are found.
dll_load_error: Optional[str] = None
if platform.system() == "Windows":
    if hasattr(os, 'add_dll_directory'):
        try:
            if DEBUG_PRINTS: print(f"[Python] Adding DLL search directory: {cl_dir}")
            # Add the CL directory to the DLL search path.
            _ = os.add_dll_directory(cl_dir)
        except OSError as e:
            dll_load_error = f"Warning: Could not add DLL directory {cl_dir}: {e}. Dependencies might not be found."
            print(f"[Python] {dll_load_error}")
    else:
        # Fallback message for older Python versions.
        dll_load_error = "Warning: os.add_dll_directory not available (requires Python 3.8+). Dependencies must be in PATH or script directory."
        print(f"[Python] {dll_load_error}")

# Determine the C library filename based on the operating system.
lib_name: str = "CipherCore_OpenCl.dll" if platform.system() == "Windows" else "libsimulated_driver.so"
lib_path: str = os.path.join(cl_dir, lib_name)

# Load the compiled C library using ctypes.
c_driver: Optional[ctypes.CDLL] = None # Initialize for potential use in finally block.
if not os.path.exists(lib_path):
    raise ImportError(f"Compiled C library not found at: {lib_path}\n"
                      f"Please compile the C code first (e.g., using instructions in the 'CL' directory) "
                      f"and ensure the resulting '{lib_name}' is in the '{cl_dir}' subdirectory.")
try:
    # Attempt to load the library.
    c_driver = ctypes.CDLL(lib_path)
    print(f"[Python] Successfully loaded C driver library from: {lib_path}")
except OSError as e:
     # Provide detailed error information if loading fails.
     print("\n--- Detailed Library Loading Error Information ---")
     print(f"Attempted Path: {lib_path}")
     print(f"Operating System: {platform.system()} {platform.architecture()}")
     print(f"Python Version: {platform.python_version()}")
     print(f"Current Directory: {os.getcwd()}")
     print(f"CL Directory (for library and dependencies): {cl_dir}")
     if dll_load_error: print(f"DLL Search Path Warning: {dll_load_error}")
     print(f"Error reported by ctypes: {e}")
     print("\nPossible Causes:")
     print("  1. Incorrect path to the primary library (.dll/.so).")
     print("  2. Missing dependency (e.g., OpenCL ICD loader like OpenCL.dll/libOpenCL.so) "
           "NOT found in the 'CL' directory or any system search path.")
     print("  3. Architecture mismatch (e.g., 32-bit Python trying to load 64-bit library or vice-versa).")
     print("  4. (Windows/MSVC) Missing VC++ Redistributables matching the compiler used for the DLL.")
     print("  5. (Linux/Rare) Missing standard C runtime libraries.")
     print("----------------------------------------------------\n")
     raise ImportError(f"Failed to load the C library '{lib_path}'. See details above. Error: {e}")


# --- GPU Detection / Selection Functions ---

def list_available_gpus() -> List[Tuple[int, Any]]:
    """
    Lists available OpenCL GPU devices using PyOpenCL.

    Iterates through OpenCL platforms and devices, printing information
    about detected GPUs (Name, Compute Units, Clock Speed, Memory).

    Returns:
        List[Tuple[int, cl.Device]]: A list where each tuple contains a
            global GPU index and the corresponding PyOpenCL Device object.
            Returns an empty list if PyOpenCL is not available or no GPUs
            are found.
    """
    if cl is None:
        print("[GPU Detect] PyOpenCL not installed. Cannot list OpenCL devices.")
        # Return a dummy entry if no PyOpenCL, assuming one device at index 0 for the C driver.
        # This allows the script to proceed if the user knows the C driver will find a device.
        print("[GPU Detect] Assuming C driver can find a device at index 0.")
        # Create a mock device object with a name attribute for compatibility
        class MockDevice:
            name = "Device (Index 0 assumed, PyOpenCL not installed)"
        return [(0, MockDevice())]

    platforms = cl.get_platforms()
    available_gpus: List[Tuple[int, cl.Device]] = []
    print("Available OpenCL GPUs:")
    print("-" * 30)
    global_gpu_index: int = 0
    for p_idx, platform_obj in enumerate(platforms):
        try:
            # Query for GPU devices on the current platform.
            devices = platform_obj.get_devices(device_type=cl.device_type.GPU)
            if not devices:
                continue # Skip if no GPUs found on this platform.

            print(f"Platform {p_idx}: {platform_obj.name}")
            for d_idx, device in enumerate(devices):
                 # Extract and display relevant device information.
                compute_units = device.max_compute_units
                clock_freq = device.max_clock_frequency
                global_mem_mb = device.global_mem_size // (1024 * 1024)
                print(f"  GPU Index {global_gpu_index}: {device.name}")
                print(f"    Compute Units: {compute_units}, Max Clock: {clock_freq} MHz, Global Memory: {global_mem_mb} MB")
                # Store the global index and the device object.
                available_gpus.append((global_gpu_index, device))
                global_gpu_index += 1
        except cl.LogicError as cle:
            # Handle cases where device info might be inaccessible.
            print(f"Warning: Could not query all devices/info for platform {p_idx} ({platform_obj.name}): {cle}")
        except Exception as e:
             # Catch other potential errors during platform/device enumeration.
             print(f"Error querying platform {p_idx} ({platform_obj.name}): {e}")
    print("-" * 30)
    if not available_gpus:
         print("No OpenCL GPUs found by PyOpenCL!")
         # Similar to the cl=None case, provide a default assumption if the C driver might still work.
         print("[GPU Detect] Assuming C driver might still find a device at index 0.")
         class MockDevice:
             name = "Device (Index 0 assumed, PyOpenCL found no GPUs)"
         return [(0, MockDevice())]

    return available_gpus

def select_gpu_index(available_gpus: List[Tuple[int, Any]]) -> int:
    """
    Prompts the user to select a GPU index from the provided list.

    Args:
        available_gpus: A list of tuples (global_index, device_object) as
                        returned by `list_available_gpus`.

    Returns:
        int: The user-selected GPU index.

    Raises:
        RuntimeError: If the `available_gpus` list is empty.
    """
    if not available_gpus:
        raise RuntimeError("No GPUs available for selection.")

    # If only one GPU is found, select it automatically.
    if len(available_gpus) == 1:
        print(f"Only one GPU (Index 0) found: {available_gpus[0][1].name}. Selecting automatically.")
        return 0

    # Prompt the user for input until a valid index is entered.
    while True:
        try:
            prompt = f"Please select the GPU index to use (0 to {len(available_gpus) - 1}) [Enter for 0]: "
            choice = input(prompt)
            # Default to GPU 0 if the user just presses Enter.
            if not choice:
                 print("No input provided, defaulting to GPU 0.")
                 return 0
            gpu_index = int(choice)
            # Validate the chosen index.
            if 0 <= gpu_index < len(available_gpus):
                print(f"Selected GPU {gpu_index}: {available_gpus[gpu_index][1].name}.")
                return gpu_index
            else:
                print(f"Invalid index. Please enter a number between 0 and {len(available_gpus) - 1}.")
        except ValueError:
            # Handle non-integer input.
            print("Invalid input. Please enter a number.")
        except EOFError:
            # Handle cases where input might be redirected or unavailable.
             print("Input stream closed, defaulting to GPU 0.")
             return 0

def initialize_selected_gpu(gpu_index: int):
    """
    Initializes the selected GPU device using the C driver function.

    Args:
        gpu_index: The index of the GPU to initialize.

    Raises:
        RuntimeError: If the C driver reports an initialization failure.
        AttributeError: If `c_driver` is None (library loading failed).
    """
    if c_driver is None:
        raise AttributeError("C driver library is not loaded. Cannot initialize GPU.")

    print(f"[Python] Attempting to initialize GPU with index {gpu_index} via C driver...")
    # Call the C function to initialize the GPU.
    # Assumes the C function returns 1 on success, 0 on failure.
    if c_driver.initialize_gpu(gpu_index) == 0:
        raise RuntimeError(f"C driver failed to initialize GPU at index {gpu_index}. Check C driver output for details.")
    print(f"[Python] GPU {gpu_index} successfully initialized via C driver.")


# --- C Function Signature Definitions using ctypes ---
# Define argument types (argtypes) and return types (restype) for functions
# exported by the C library. This allows Python to call them correctly.

# Define a type alias for the GPU buffer handle (likely a pointer).
GPU_BUFFER_HANDLE = ctypes.c_void_p

# Core GPU Management Functions
c_driver.initialize_gpu.argtypes = [ctypes.c_int]
c_driver.initialize_gpu.restype = ctypes.c_int # 1 for success, 0 for failure
c_driver.allocate_gpu_memory.argtypes = [ctypes.c_int, ctypes.c_size_t]
c_driver.allocate_gpu_memory.restype = GPU_BUFFER_HANDLE # Returns pointer or NULL
c_driver.free_gpu_memory.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE]
c_driver.free_gpu_memory.restype = None
c_driver.write_host_to_gpu_blocking.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p]
c_driver.write_host_to_gpu_blocking.restype = ctypes.c_int # 1 for success, 0 for failure
c_driver.read_gpu_to_host_blocking.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p]
c_driver.read_gpu_to_host_blocking.restype = ctypes.c_int # 1 for success, 0 for failure
c_driver.simulated_get_compute_unit_count.argtypes = [ctypes.c_int]
c_driver.simulated_get_compute_unit_count.restype = ctypes.c_uint
c_driver.shutdown_gpu.argtypes = [ctypes.c_int]
c_driver.shutdown_gpu.restype = None

# GPU Compute Kernel Execution Functions
c_driver.execute_clone_on_gpu.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, ctypes.c_size_t]
c_driver.execute_clone_on_gpu.restype = ctypes.c_int # 1 for success, 0 for failure
c_driver.execute_matmul_on_gpu.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
c_driver.execute_matmul_on_gpu.restype = ctypes.c_int # 1 for success, 0 for failure
c_driver.execute_add_on_gpu.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, ctypes.c_int]
c_driver.execute_add_on_gpu.restype = ctypes.c_int # 1 for success, 0 for failure
c_driver.execute_add_bias_on_gpu.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, ctypes.c_int, ctypes.c_int]
c_driver.execute_add_bias_on_gpu.restype = ctypes.c_int # 1 for success, 0 for failure
c_driver.execute_gelu_on_gpu.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, ctypes.c_int]
c_driver.execute_gelu_on_gpu.restype = ctypes.c_int # 1 for success, 0 for failure
c_driver.execute_layernorm_on_gpu.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, ctypes.c_int, ctypes.c_int, ctypes.c_float]
c_driver.execute_layernorm_on_gpu.restype = ctypes.c_int # 1 for success, 0 for failure
c_driver.execute_log_softmax_stable_gpu.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, ctypes.c_int, ctypes.c_int]
c_driver.execute_log_softmax_stable_gpu.restype = ctypes.c_int # 1 for success, 0 for failure
c_driver.execute_embedding_lookup_gpu.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
c_driver.execute_embedding_lookup_gpu.restype = ctypes.c_int # 1 for success, 0 for failure
c_driver.execute_gelu_backward_on_gpu.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, ctypes.c_int]
c_driver.execute_gelu_backward_on_gpu.restype = ctypes.c_int # 1 for success, 0 for failure
c_driver.execute_matmul_backward_on_gpu.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
c_driver.execute_matmul_backward_on_gpu.restype = ctypes.c_int # 1 for success, 0 for failure
c_driver.execute_reduce_sum_gpu.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, ctypes.c_int, ctypes.c_int, ctypes.c_int]
c_driver.execute_reduce_sum_gpu.restype = ctypes.c_int # 1 for success, 0 for failure
c_driver.execute_cross_entropy_loss_grad_gpu.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, ctypes.c_int, ctypes.c_int]
c_driver.execute_cross_entropy_loss_grad_gpu.restype = ctypes.c_int # 1 for success, 0 for failure
c_driver.execute_embedding_backward_gpu.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
c_driver.execute_embedding_backward_gpu.restype = ctypes.c_int # 1 for success, 0 for failure
print("[Python] CTypes definition for execute_embedding_backward_gpu loaded.")
c_driver.execute_adam_update_on_gpu.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
c_driver.execute_adam_update_on_gpu.restype = ctypes.c_int # 1 for success, 0 for failure

# Bio-Inspired Layer Specific Kernels
c_driver.execute_hebbian_update_on_gpu.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
c_driver.execute_hebbian_update_on_gpu.restype = ctypes.c_int # 1 for success, 0 for failure
c_driver.execute_threshold_spike_on_gpu.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, ctypes.c_float, ctypes.c_int]
c_driver.execute_threshold_spike_on_gpu.restype = ctypes.c_int # 1 for success, 0 for failure
c_driver.execute_dynamic_token_assignment_gpu.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
c_driver.execute_dynamic_token_assignment_gpu.restype = ctypes.c_int # 1 for success, 0 for failure
c_driver.execute_pairwise_similarity_gpu.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, ctypes.c_int, ctypes.c_int]
c_driver.execute_pairwise_similarity_gpu.restype = ctypes.c_int # 1 for success, 0 for failure

# Optional GPU Prototype Update Kernels (check if they exist in the loaded library)
CAN_USE_GPU_PROTO_UPDATE: bool = False
try:
    # Define signatures only if the functions are found.
    c_driver.execute_proto_segmented_sum_gpu.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    c_driver.execute_proto_segmented_sum_gpu.restype = ctypes.c_int # 1 for success, 0 for failure
    c_driver.execute_proto_update_step_gpu.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, ctypes.c_float, ctypes.c_int, ctypes.c_int]
    c_driver.execute_proto_update_step_gpu.restype = ctypes.c_int # 1 for success, 0 for failure
    print("[Python] CTypes definitions for optional GPU prototype update kernels loaded successfully.")
    CAN_USE_GPU_PROTO_UPDATE = True
except AttributeError:
    # Library was compiled without these functions.
    print("[Python] WARNING: Optional GPU prototype update functions not found in the C library. Host-based fallback will be used for prototype updates.")

# Sanity check: If GPU update was requested but functions aren't available, raise an error.
if USE_GPU_PROTOTYPE_UPDATE and not CAN_USE_GPU_PROTO_UPDATE:
     print("[Python] FATAL ERROR: Configuration requests GPU prototype update, but required functions are missing from the C library!")
     sys.exit(1)


# --- GPUTensor Class ---

class GPUTensor:
    """
    Manages a memory buffer allocated on a specific GPU device.

    This class acts as a wrapper around a GPU memory handle obtained from the
    C driver. It provides methods for allocating, freeing, writing data from
    NumPy arrays to the GPU, and reading data from the GPU back to NumPy arrays.

    Attributes:
        gpu_index (int): The index of the GPU where the buffer resides.
        size (int): The allocated size of the buffer in bytes.
        name (str): An optional name for debugging purposes.
        handle (GPU_BUFFER_HANDLE): The memory handle (pointer) returned by
            the C driver's allocation function. None if not allocated or freed.
    """
    def __init__(self, size_in_bytes: int, gpu_index: int = GPU_INDEX, name: str = "Tensor", zero_init: bool = False):
        """
        Initializes the GPUTensor and allocates memory on the specified GPU.

        Args:
            size_in_bytes: The desired size of the memory buffer in bytes.
            gpu_index: The index of the target GPU device. Defaults to `GPU_INDEX`.
            name: An optional name for the tensor, used in debug messages.
            zero_init: If True, the allocated GPU memory will be initialized to zeros.

        Raises:
            ValueError: If `size_in_bytes` is negative.
            MemoryError: If the C driver fails to allocate the requested memory.
            AttributeError: If `c_driver` is None (library loading failed).
        """
        if c_driver is None:
             raise AttributeError("C driver library is not loaded. Cannot create GPUTensor.")

        self.gpu_index: int = gpu_index
        self.size: int = int(size_in_bytes) # Ensure integer size
        self.name: str = name
        self._is_freed: bool = False # Internal flag to track memory state
        self.handle: Optional[GPU_BUFFER_HANDLE] = None # Initialize handle

        if self.size < 0:
             raise ValueError(f"Invalid negative size requested for GPUTensor '{self.name}': {self.size}")

        # Allocate memory only if size is positive.
        if self.size > 0:
            self.handle = c_driver.allocate_gpu_memory(self.gpu_index, self.size)
            # Check if allocation was successful (handle should not be NULL/None).
            if not self.handle:
                raise MemoryError(f"GPU memory allocation failed for {self.size} bytes "
                                  f"(Tensor: '{self.name}') on GPU {self.gpu_index}. "
                                  f"Check C driver output and available GPU memory.")

            # Optional debug print showing the allocated handle address.
            handle_int = ctypes.cast(self.handle, ctypes.c_void_p).value
            if DEBUG_PRINTS: print(f"[Python GPUTensor] Allocated '{self.name}' ({self.size} bytes), handle: {hex(handle_int) if handle_int else 'None'}) on GPU {self.gpu_index}")

            # Initialize the buffer with zeros if requested.
            if zero_init:
                self._zero_initialize()
        elif DEBUG_PRINTS:
             # Log allocation of zero-size tensors (handle remains None).
             print(f"[Python GPUTensor] Allocated '{self.name}' (0 bytes), handle: None")

    def _zero_initialize(self):
        """
        Fills the allocated GPU buffer with zeros.

        Uses a temporary NumPy array of zeros and writes it to the GPU.
        Attempts to infer the correct data type (int or float) based on
        common naming conventions (e.g., 'indices', 'counts', 'target')
        to create the zero buffer efficiently. Falls back to byte buffer
        if size is not a multiple of standard types.
        """
        if self.handle is None or self.size == 0:
            # Cannot zero-initialize if not allocated or size is zero.
            return

        item_size: int = 4 # Assume FP32 or INT32 elements (4 bytes)
        num_elements: int = self.size // item_size
        zeros_buffer: Optional[np.ndarray] = None # Initialize buffer variable

        # Check if size is compatible with standard element types.
        if num_elements * item_size != self.size:
            # If not a multiple of 4, create a byte buffer.
            print(f"[Python GPUTensor] Warning: Zero-initializing '{self.name}' byte-by-byte as size ({self.size}) is not multiple of {item_size}.")
            zeros_buffer = np.zeros(self.size, dtype=np.byte)
        else:
             # Attempt to guess the data type based on the tensor name.
             lname = self.name.lower()
             if any(term in lname for term in ['indices', 'counts', 'target', 'inputindices']):
                 zeros_buffer = np.zeros(num_elements, dtype=INT_TYPE)
             else:
                 # Default to float type for other tensors.
                 zeros_buffer = np.zeros(num_elements, dtype=FP_TYPE)

        try:
            # Write the zero buffer to the GPU.
            if zeros_buffer is not None:
                self.write(zeros_buffer)
                if DEBUG_PRINTS: print(f"[Python GPUTensor] Zero-initialized '{self.name}' ({self.size} bytes).")
        except Exception as e:
             # Catch potential errors during the write operation.
             print(f"[Python GPUTensor] ERROR during zero-initialization for '{self.name}': {e}")
        finally:
            # Ensure the temporary host buffer is deleted.
            if zeros_buffer is not None:
                del zeros_buffer

    def write(self, host_data_np: np.ndarray, offset_bytes: int = 0):
        """
        Writes data from a NumPy array (host) to the GPU buffer.

        Args:
            host_data_np: The NumPy array containing the data to write. Must be
                          C-contiguous.
            offset_bytes: The starting offset in bytes within the GPU buffer
                          where writing should begin. Defaults to 0.

        Raises:
            RuntimeError: If the tensor has been freed, the handle is invalid,
                          or the C driver reports a write error.
            ValueError: If the host array is not C-contiguous, or if the write
                        operation (offset + data size) exceeds the allocated
                        buffer size, or if offset/size are negative.
            AttributeError: If `c_driver` is None (library loading failed).
        """
        if c_driver is None:
             raise AttributeError("C driver library is not loaded. Cannot write to GPUTensor.")
        if self._is_freed:
            raise RuntimeError(f"Cannot write to freed GPUTensor '{self.name}'.")
        if self.handle is None:
             # Allow writing zero bytes to a zero-size tensor (effectively a no-op).
             if self.size == 0 and host_data_np.nbytes == 0:
                 return
             raise RuntimeError(f"Cannot write to GPUTensor '{self.name}' with invalid handle (possibly freed or zero-size).")

        # Ensure the NumPy array is C-contiguous for direct memory access.
        if not host_data_np.flags['C_CONTIGUOUS']:
            # Create a contiguous copy if necessary.
            host_data_np = np.ascontiguousarray(host_data_np)
            # Consider adding a warning here if performance is critical.

        # Get pointer to the NumPy array's data buffer.
        data_ptr = host_data_np.ctypes.data_as(ctypes.c_void_p)
        # Get the size of the data to write in bytes.
        size_bytes = host_data_np.nbytes

        # Basic validation of offset and size.
        if offset_bytes < 0 or size_bytes < 0:
            raise ValueError(f"Negative offset ({offset_bytes}) or size ({size_bytes}) specified for write operation on '{self.name}'.")
        # Boundary check: ensure write does not exceed allocated buffer size.
        if offset_bytes + size_bytes > self.size:
            raise ValueError(f"Write operation on '{self.name}' exceeds allocated buffer size. "
                             f"Offset: {offset_bytes}, Data Size: {size_bytes}, Buffer Size: {self.size}")

        # Call the C driver function to perform the blocking write.
        if c_driver.write_host_to_gpu_blocking(self.gpu_index, self.handle, offset_bytes, size_bytes, data_ptr) == 0:
            # C function returns 0 on failure.
            raise RuntimeError(f"C driver failed to write data to GPU for '{self.name}' (GPU {self.gpu_index}). Check C driver output.")

    def read(self, host_data_np: np.ndarray, offset_bytes: int = 0) -> np.ndarray:
        """
        Reads data from the GPU buffer into a NumPy array (host).

        Args:
            host_data_np: The pre-allocated, C-contiguous, writeable NumPy array
                          where the data from the GPU will be stored. Its size
                          determines how many bytes are read.
            offset_bytes: The starting offset in bytes within the GPU buffer
                          from where reading should begin. Defaults to 0.

        Returns:
            np.ndarray: The same `host_data_np` array, now filled with data
                        read from the GPU.

        Raises:
            RuntimeError: If the tensor has been freed, the handle is invalid,
                          or the C driver reports a read error.
            ValueError: If the target host array is not C-contiguous or not
                        writeable, or if the read operation (offset + data size)
                        exceeds the allocated buffer size, or if offset/size
                        are negative.
            AttributeError: If `c_driver` is None (library loading failed).
        """
        if c_driver is None:
             raise AttributeError("C driver library is not loaded. Cannot read from GPUTensor.")
        if self._is_freed:
            raise RuntimeError(f"Cannot read from freed GPUTensor '{self.name}'.")
        if self.handle is None:
             # Allow reading zero bytes from a zero-size tensor.
             if self.size == 0 and host_data_np.nbytes == 0:
                 return host_data_np
             raise RuntimeError(f"Cannot read from GPUTensor '{self.name}' with invalid handle (possibly freed or zero-size).")

        # Ensure the target NumPy array meets requirements for receiving data.
        if not host_data_np.flags['C_CONTIGUOUS']:
            raise ValueError(f"Target NumPy array for reading '{self.name}' must be C-contiguous.")
        if not host_data_np.flags['WRITEABLE']:
            raise ValueError(f"Target NumPy array for reading '{self.name}' must be writeable.")

        # Get pointer to the target NumPy array's data buffer.
        data_ptr = host_data_np.ctypes.data_as(ctypes.c_void_p)
        # Determine the number of bytes to read based on the target array's size.
        size_bytes = host_data_np.nbytes

        # Basic validation of offset and size.
        if offset_bytes < 0 or size_bytes < 0:
            raise ValueError(f"Negative offset ({offset_bytes}) or size ({size_bytes}) specified for read operation on '{self.name}'.")
        # Boundary check: ensure read does not exceed allocated buffer size.
        if offset_bytes + size_bytes > self.size:
             raise ValueError(f"Read operation on '{self.name}' exceeds allocated buffer size. "
                             f"Offset: {offset_bytes}, Read Size: {size_bytes}, Buffer Size: {self.size}")

        # Call the C driver function to perform the blocking read.
        if c_driver.read_gpu_to_host_blocking(self.gpu_index, self.handle, offset_bytes, size_bytes, data_ptr) == 0:
            # C function returns 0 on failure.
            raise RuntimeError(f"C driver failed to read data from GPU for '{self.name}' (GPU {self.gpu_index}). Check C driver output.")

        # Return the modified host array.
        return host_data_np

    def free(self):
        """
        Releases the allocated GPU memory associated with this tensor.

        Calls the C driver's `free_gpu_memory` function. Marks the tensor
        as freed to prevent further operations. Safe to call multiple times.
        """
        if not self._is_freed and self.handle:
            # Only attempt to free if not already freed and handle is valid.
            handle_to_free = self.handle
            handle_int = ctypes.cast(handle_to_free, ctypes.c_void_p).value # For logging
            if DEBUG_PRINTS: print(f"[Python GPUTensor] Freeing '{self.name}' (handle: {hex(handle_int)}) on GPU {self.gpu_index}")
            try:
                # Call the C driver function to free the memory.
                if c_driver: # Check if c_driver exists before calling
                    c_driver.free_gpu_memory(self.gpu_index, handle_to_free)
                else:
                    print(f"[Python GPUTensor] WARNING: Cannot free '{self.name}', C driver not loaded.")
            except Exception as e:
                # Log potential errors during freeing, but don't stop execution.
                print(f"[Python GPUTensor] WARNING: Exception during C free_gpu_memory for '{self.name}': {e}")
            finally:
                # Ensure handle and size are reset even if C call fails.
                self.handle = None
                self.size = 0
        # Mark as freed regardless of whether memory was actually released
        # (prevents repeated attempts or operations on invalid state).
        self._is_freed = True

    def __del__(self):
        """
        Destructor: Ensures GPU memory is freed when the GPUTensor object
        is garbage collected.

        This acts as a safeguard against memory leaks if `free()` is not
        explicitly called.
        """
        # `getattr` check prevents errors if `__init__` failed before `_is_freed` was set.
        if not getattr(self, '_is_freed', True):
             if DEBUG_PRINTS: print(f"[Python GPUTensor] Destructor called for '{self.name}' - ensuring memory is freed.")
             # Call the proper free method.
             self.free()

# --- Data Processing Functions for Character-Level Text ---

def preprocess_char_data(input_path: str, output_path: str, seq_len: int, val_split: float = 0.1):
    """
    Processes a raw text file for character-level modeling.

    1. Reads the text file.
    2. Builds a character vocabulary (maps characters to integer IDs).
    3. Converts the entire text into a sequence of integer IDs.
    4. Creates input/target pairs for next-character prediction:
       - Input: Sequence of `seq_len` character IDs.
       - Target: Sequence of `seq_len` character IDs, shifted by one position.
    5. Splits the sequences into training and validation sets.
    6. Saves the processed data (input/target arrays) into a NumPy `.npz` file.
    7. Saves the vocabulary mappings (`char_to_id`, `id_to_char`) into a
       separate pickle file (`_vocab.pkl`).

    Args:
        input_path: Path to the raw input text file (.txt).
        output_path: Base path for the output files. `.npz` will be appended for
                     data, and `_vocab.pkl` for the vocabulary.
        seq_len: The length of the character sequences to create.
        val_split: The fraction of data to use for the validation set (default: 0.1).

    Raises:
        FileNotFoundError: If the `input_path` does not exist.
        ValueError: If the text is too short for the specified `seq_len`.
    """
    vocab_output_path = output_path + "_mini_vocab.pkl"
    # Avoid reprocessing if output files already exist.
    if os.path.exists(output_path) and os.path.exists(vocab_output_path):
        print(f"[DataPrep] Processed data ('{output_path}') and vocab ('{vocab_output_path}') already exist. Skipping preprocessing.")
        return

    print(f"[DataPrep] Starting preprocessing for '{input_path}'...")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input text file not found: {input_path}")

    # Read the entire text content.
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"[DataPrep] Text loaded ({len(text)} characters). Building vocabulary...")

    # Create character vocabulary.
    chars = sorted(list(set(text))) # Unique characters, sorted for consistency.
    vocab_size = len(chars)
    char_to_id = {ch: i for i, ch in enumerate(chars)}
    id_to_char = {i: ch for i, ch in enumerate(chars)}
    print(f"[DataPrep] Vocabulary size: {vocab_size}")

    # Convert the text into a NumPy array of integer IDs.
    data_ids = np.array([char_to_id[ch] for ch in text], dtype=INT_TYPE)
    print(f"[DataPrep] Text converted to {len(data_ids)} integer IDs.")

    # Create input and target sequences for next-character prediction.
    num_sequences = len(data_ids) - seq_len
    if num_sequences <= 0:
        raise ValueError(f"Text is too short ({len(data_ids)} IDs) for sequence length {seq_len}. Need at least {seq_len + 1} characters.")

    # Pre-allocate NumPy arrays for efficiency.
    inputs = np.zeros((num_sequences, seq_len), dtype=INT_TYPE)
    targets = np.zeros((num_sequences, seq_len), dtype=INT_TYPE)

    print(f"[DataPrep] Creating {num_sequences} input/target sequence pairs (length {seq_len})...")
    # Populate the arrays using sliding windows.
    for i in range(num_sequences):
        inputs[i] = data_ids[i : i + seq_len]
        targets[i] = data_ids[i + 1 : i + 1 + seq_len] # Target is the next character.

    # Split into training and validation sets.
    split_idx = int(num_sequences * (1 - val_split))
    train_inputs = inputs[:split_idx]
    train_targets = targets[:split_idx]
    valid_inputs = inputs[split_idx:]
    valid_targets = targets[split_idx:]
    print(f"[DataPrep] Data split: {len(train_inputs)} training sequences, {len(valid_inputs)} validation sequences.")

    # Save the vocabulary using pickle.
    vocab_data = {'char_to_id': char_to_id, 'id_to_char': id_to_char}
    with open(vocab_output_path, 'wb') as f:
        pickle.dump(vocab_data, f)
    print(f"[DataPrep] Vocabulary saved to '{vocab_output_path}'.")

    # Save the training and validation data using NumPy's savez (compressed).
    np.savez(output_path,
             train_inputs=train_inputs,
             train_targets=train_targets,
             valid_inputs=valid_inputs,
             valid_targets=valid_targets)
    print(f"[DataPrep] Processed sequence data saved to '{output_path}'.")

def load_processed_data(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, dict, dict]:
    """
    Loads the preprocessed character data and vocabulary.

    Args:
        filepath: The base path to the processed data files (expects `.npz`
                  and `_vocab.pkl` suffixes).

    Returns:
        Tuple containing:
        - train_inputs (np.ndarray): Training input sequences (int IDs).
        - train_targets (np.ndarray): Training target sequences (int IDs).
        - valid_inputs (np.ndarray): Validation input sequences (int IDs).
        - valid_targets (np.ndarray): Validation target sequences (int IDs).
        - vocab_size (int): The number of unique characters in the vocabulary.
        - char_to_id (dict): Mapping from character to integer ID.
        - id_to_char (dict): Mapping from integer ID to character.

    Raises:
        FileNotFoundError: If the `.npz` data file or the `_vocab.pkl` file
                           is not found.
        ValueError: If the loaded data arrays do not have the expected 2D shape
                    or if input/target shapes within a split mismatch.
    """
    vocab_path = filepath + "_mini_vocab.pkl"
    if not os.path.exists(filepath) or not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Processed data file ('{filepath}') or vocabulary file ('{vocab_path}') not found. Run preprocessing first.")

    print(f"[DataLoader] Loading processed data from '{filepath}'...")
    # Load the NumPy archive.
    data = np.load(filepath)
    # Extract arrays and ensure correct integer type.
    train_inputs = data['train_inputs'].astype(INT_TYPE)
    train_targets = data['train_targets'].astype(INT_TYPE)
    valid_inputs = data['valid_inputs'].astype(INT_TYPE)
    valid_targets = data['valid_targets'].astype(INT_TYPE)

    print(f"[DataLoader] Loading vocabulary from '{vocab_path}'...")
    # Load the vocabulary dictionary from the pickle file.
    with open(vocab_path, 'rb') as f:
        vocab_data = pickle.load(f)
    char_to_id = vocab_data['char_to_id']
    id_to_char = vocab_data['id_to_char']
    vocab_size = len(char_to_id)

    # --- Data Validation ---
    # Check array dimensions (should be 2D: num_sequences x seq_len).
    if train_inputs.ndim != 2 or train_targets.ndim != 2 or \
       valid_inputs.ndim != 2 or valid_targets.ndim != 2:
        raise ValueError(f"Loaded data arrays do not have the expected 2 dimensions. Shapes: "
                         f"TrainIn={train_inputs.shape}, TrainTgt={train_targets.shape}, "
                         f"ValidIn={valid_inputs.shape}, ValidTgt={valid_targets.shape}")
    # Check if input and target shapes match within each split.
    if train_inputs.shape != train_targets.shape:
         raise ValueError(f"Training input shape {train_inputs.shape} does not match training target shape {train_targets.shape}.")
    if valid_inputs.shape != valid_targets.shape:
         raise ValueError(f"Validation input shape {valid_inputs.shape} does not match validation target shape {valid_targets.shape}.")

    print("[DataLoader] Processed data loaded successfully:")
    print(f"  Train Inputs Shape:  {train_inputs.shape}, Type: {train_inputs.dtype}")
    print(f"  Train Targets Shape: {train_targets.shape}, Type: {train_targets.dtype}")
    print(f"  Valid Inputs Shape:  {valid_inputs.shape}, Type: {valid_inputs.dtype}")
    print(f"  Valid Targets Shape: {valid_targets.shape}, Type: {valid_targets.dtype}")
    print(f"  Vocabulary Size: {vocab_size}")

    return train_inputs, train_targets, valid_inputs, valid_targets, vocab_size, char_to_id, id_to_char

def create_batches(inputs: np.ndarray, targets: np.ndarray, batch_size: int, seq_len: int, shuffle: bool = True) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Creates batches of input and target sequences from the provided data arrays.

    Handles shuffling and pads the last batch if it's smaller than `batch_size`.
    Padding uses the `PAD_INDEX` constant.

    Args:
        inputs: A 2D NumPy array of input sequences (num_samples x seq_len).
        targets: A 2D NumPy array of target sequences (num_samples x seq_len).
        batch_size: The desired number of sequences per batch.
        seq_len: The length of each sequence (should match the second dimension
                 of inputs/targets).
        shuffle: If True, shuffles the order of sequences before batching.

    Yields:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the input batch
            and the target batch, both as 2D NumPy arrays of shape
            (batch_size x seq_len). The last batch might be padded.
    """
    num_samples = inputs.shape[0]
    # Create an array of indices [0, 1, ..., num_samples-1].
    indices = np.arange(num_samples)
    # Shuffle indices randomly if requested.
    if shuffle:
        np.random.shuffle(indices)

    # Iterate through the data in steps of batch_size.
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        # Get the indices for the current batch.
        batch_indices = indices[start_idx:end_idx]
        current_batch_size = len(batch_indices) # Actual size of this batch.

        # Extract the corresponding input and target sequences using advanced indexing.
        batch_inputs = inputs[batch_indices]
        batch_targets = targets[batch_indices]

        # Pad the last batch if it's smaller than the requested batch_size.
        if current_batch_size < batch_size:
            num_padding = batch_size - current_batch_size
            # Create padding arrays filled with PAD_INDEX.
            # Input padding shape: (num_padding x seq_len).
            input_padding_shape = (num_padding, seq_len)
            input_padding = np.full(input_padding_shape, PAD_INDEX, dtype=inputs.dtype)
            # Concatenate original batch with padding.
            batch_inputs = np.concatenate([batch_inputs, input_padding], axis=0)

            # Target padding shape: (num_padding x seq_len).
            target_padding_shape = (num_padding, seq_len)
            target_padding = np.full(target_padding_shape, PAD_INDEX, dtype=targets.dtype)
            # Concatenate original batch with padding.
            batch_targets = np.concatenate([batch_targets, target_padding], axis=0)

        # Yield the completed (potentially padded) batch.
        yield batch_inputs, batch_targets


# --- Embedding Layer ---

class EmbeddingLayer:
    """
    Represents an Embedding layer that maps integer token IDs to dense vectors.

    Uses a trainable weight matrix (W_emb) stored on the GPU. The forward pass
    performs a lookup operation using the C/OpenCL kernel. The backward pass
    accumulates gradients for the used embeddings. Updates are done using Adam.

    Attributes:
        V (int): Vocabulary size.
        E (int): Embedding dimension.
        gpu_index (int): GPU device index.
        W_emb (GPUTensor): GPU tensor for the embedding weights (V x E).
        dW_emb (GPUTensor): GPU tensor for the gradients of W_emb (V x E).
        m_W_emb (GPUTensor): GPU tensor for the Adam first moment (momentum) of W_emb.
        v_W_emb (GPUTensor): GPU tensor for the Adam second moment (variance) of W_emb.
        output (Optional[GPUTensor]): Reference to the output buffer (managed externally).
        input_indices_handle_cache (Optional[GPU_BUFFER_HANDLE]): Cached handle of the
            input indices tensor from the last forward pass, needed for backward.
        current_B (int): Current batch size.
        current_S (int): Current sequence length.
    """
    def __init__(self, vocab_size: int, embedding_dim: int, gpu_index: int = GPU_INDEX):
        """
        Initializes the EmbeddingLayer.

        Args:
            vocab_size: The size of the vocabulary (number of unique tokens).
            embedding_dim: The desired dimension for the embedding vectors.
            gpu_index: The index of the GPU device to use.
        """
        self.V: int = vocab_size
        self.E: int = embedding_dim
        self.gpu_index: int = gpu_index
        self._is_freed: bool = False # Flag for resource management
        # Calculate item sizes for allocation.
        itemsize_fp = FP_TYPE().itemsize
        itemsize_adam = np.float32().itemsize # Adam states are often kept in float32

        # Allocate GPU tensors for weights, gradients, and Adam optimizer states.
        # Weights are initialized later; gradients and Adam states are zero-initialized.
        self.W_emb = GPUTensor(self.V * self.E * itemsize_fp, gpu_index, name="EmbeddingW")
        self.dW_emb = GPUTensor(self.V * self.E * itemsize_fp, gpu_index, name="dEmbeddingW", zero_init=True)
        self.m_W_emb = GPUTensor(self.V * self.E * itemsize_adam, gpu_index, name="m_EmbeddingW", zero_init=True)
        self.v_W_emb = GPUTensor(self.V * self.E * itemsize_adam, gpu_index, name="v_EmbeddingW", zero_init=True)

        # Output tensor is allocated and managed by the main Model class.
        self.output: Optional[GPUTensor] = None
        # Cache for the handle of the input tensor during forward pass.
        self.input_indices_handle_cache: Optional[GPU_BUFFER_HANDLE] = None
        # Current batch dimensions (set externally before forward/backward).
        self.current_B: int = 0
        self.current_S: int = 0

        if DEBUG_PRINTS: print(f"[EmbeddingLayer] Initialized: VocabSize={self.V}, EmbeddingDim={self.E} on GPU {self.gpu_index}")

    def _check_freed(self):
        """Raises RuntimeError if operations are attempted on a freed layer."""
        if self._is_freed:
            raise RuntimeError("Operation attempted on a freed EmbeddingLayer.")

    def set_current_batch_shape(self, b: int, s: int):
        """
        Sets the batch size (B) and sequence length (S) for the current operation.

        Args:
            b: Current batch size.
            s: Current sequence length.
        """
        self._check_freed()
        self.current_B = b
        self.current_S = s

    def forward(self, input_indices_gpu: GPUTensor, output_buffer_gpu: GPUTensor) -> GPUTensor:
        """
        Performs the embedding lookup operation on the GPU.

        Reads integer token IDs from `input_indices_gpu`, looks up the
        corresponding vectors in `W_emb`, and writes the results to
        `output_buffer_gpu`.

        Args:
            input_indices_gpu: A GPUTensor containing the batch of input token
                               IDs (shape B * S, expected int type).
            output_buffer_gpu: A pre-allocated GPUTensor where the resulting
                               embedding vectors will be stored (shape B * S x E).

        Returns:
            GPUTensor: The `output_buffer_gpu` containing the embeddings.

        Raises:
            RuntimeError: If C driver execution fails or required handles are None.
        """
        self._check_freed()
        if c_driver is None:
            raise AttributeError("C driver library is not loaded. Cannot perform Embedding forward pass.")

        # Cache the input handle for the backward pass.
        self.input_indices_handle_cache = input_indices_gpu.handle
        # Store reference to the externally provided output buffer.
        self.output = output_buffer_gpu

        # --- Handle Validation ---
        if self.output is None or self.output.handle is None:
            raise RuntimeError("Output buffer or its handle is None in EmbeddingLayer forward.")
        if self.input_indices_handle_cache is None:
            raise RuntimeError("Input indices handle cache is None in EmbeddingLayer forward.")
        if self.W_emb.handle is None:
            raise RuntimeError("Embedding weight handle (W_emb) is None in EmbeddingLayer forward.")
        # --- End Handle Validation ---

        # Call the C/OpenCL kernel for embedding lookup.
        success = c_driver.execute_embedding_lookup_gpu(
            self.gpu_index,                # GPU device index
            self.input_indices_handle_cache, # Input: Token indices (B*S)
            self.W_emb.handle,             # Input: Embedding matrix (V x E)
            self.output.handle,            # Output: Resulting embeddings (B*S x E)
            self.current_B,                # Batch size
            self.current_S,                # Sequence length
            self.E,                        # Embedding Dimension (d in C kernel)
            self.V                         # Vocabulary Size (v in C kernel)
        )

        if success == 0:
            raise RuntimeError("C driver execute_embedding_lookup_gpu failed. Check C driver output.")

        return self.output

    def backward(self, d_output: GPUTensor):
        """
        Computes the gradient of the loss with respect to the embedding weights (dW_emb).

        Uses the gradient coming from the next layer (`d_output`) and the cached
        input indices from the forward pass to accumulate gradients into `dW_emb`.
        This layer does not propagate gradients further back, as it's typically
        the first layer.

        Args:
            d_output: A GPUTensor containing the gradients flowing back from the
                      subsequent layer (shape B * S x E).

        Returns:
            None: Gradients are accumulated internally in `dW_emb`.

        Raises:
            RuntimeError: If the input indices were not cached from forward,
                          required handles are None, or C driver execution fails.
        """
        self._check_freed()
        if c_driver is None:
            raise AttributeError("C driver library is not loaded. Cannot perform Embedding backward pass.")

        # Backward pass requires the input indices used in the forward pass.
        if self.input_indices_handle_cache is None:
            raise RuntimeError("Cannot perform backward pass on EmbeddingLayer without cached input indices from forward pass.")

        # --- Handle Validation ---
        if d_output.handle is None:
             raise RuntimeError("Gradient input handle (d_output) is None in EmbeddingLayer backward.")
        if self.dW_emb.handle is None:
            raise RuntimeError("Gradient weight handle (dW_emb) is None in EmbeddingLayer backward.")
        # --- End Handle Validation ---

        # Call the C/OpenCL kernel for embedding backward pass (gradient accumulation).
        # Note: This likely uses an approach like segmented sum or multiple passes
        # if atomics are not reliably available/performant across all OpenCL devices.
        success = c_driver.execute_embedding_backward_gpu(
            self.gpu_index,                 # GPU device index
            d_output.handle,                # Input: Gradient from next layer (d_o, shape B*S x E)
            self.input_indices_handle_cache,  # Input: Original input token indices (idx, shape B*S)
            self.dW_emb.handle,             # Output: Accumulated gradients for weights (d_w, shape V x E)
            self.current_B,                 # Batch size
            self.current_S,                 # Sequence length
            self.E,                         # Embedding Dimension (d)
            self.V                          # Vocabulary Size (v)
        )

        if success == 0:
            raise RuntimeError("C driver execute_embedding_backward_gpu failed. Check C driver output.")

        # Embedding layer is usually the input layer, so it doesn't return gradients.
        return None

    def clip_gradients(self, clip_value: float):
        """
        Clips the L2 norm of the embedding gradients (dW_emb).

        Reads the gradient tensor from GPU, calculates its L2 norm on the host,
        scales it if the norm exceeds `clip_value`, and writes the scaled
        gradient back to the GPU.

        Args:
            clip_value: The maximum allowed L2 norm for the gradients. If the
                        norm is less than or equal to this value, gradients
                        remain unchanged. If `<= 0`, clipping is skipped.
        """
        self._check_freed()
        # Skip clipping if value is non-positive or tensor is empty.
        if clip_value <= 0 or self.dW_emb.size == 0:
            return

        grad_W_host: Optional[np.ndarray] = None # Initialize host buffer variable
        try:
            # Allocate host memory to read the gradients.
            grad_W_host = np.zeros((self.V, self.E), dtype=FP_TYPE)
            # Read gradients from GPU.
            self.dW_emb.read(grad_W_host)

            # Calculate L2 norm of the gradients.
            norm_W = np.linalg.norm(grad_W_host)

            # Calculate scaling factor if norm exceeds threshold.
            # Add small epsilon for numerical stability if norm is near zero.
            scale_W = clip_value / (norm_W + 1e-6) if norm_W > clip_value else 1.0

            # Apply scaling only if necessary (scale_W < 1.0).
            if scale_W < 1.0:
                if DEBUG_PRINTS: print(f"[GradClip] Clipping Embedding W gradient norm {norm_W:.4f} with scale {scale_W:.4f}")
                grad_W_host *= scale_W
                # Write the clipped gradients back to the GPU.
                self.dW_emb.write(grad_W_host)

        except Exception as e:
            print(f"[EmbeddingLayer] ERROR during gradient clipping: {e}")
        finally:
            # Ensure the host buffer is deleted.
            if grad_W_host is not None:
                del grad_W_host

    def update(self, t: int, lr: float, beta1: float = 0.9, beta2: float = 0.999, weight_decay: float = 0.0):
        """
        Updates the embedding weights (W_emb) using the Adam optimizer rule on the GPU.

        Args:
            t: The current timestep (iteration number, > 0) for bias correction.
            lr: The current learning rate.
            beta1: Adam hyperparameter beta1 (exponential decay rate for momentum).
            beta2: Adam hyperparameter beta2 (exponential decay rate for variance).
            weight_decay: L2 regularization factor. Applied by the C kernel if > 0.
                          *Note: Often set to 0 for embedding layers.*

        Raises:
            RuntimeError: If the C driver Adam update execution fails.
        """
        self._check_freed()
        if c_driver is None:
            raise AttributeError("C driver library is not loaded. Cannot perform Embedding update.")

        num_elements_W = self.V * self.E
        if num_elements_W > 0:
            # Call the C/OpenCL kernel for the Adam update.
            success = c_driver.execute_adam_update_on_gpu(
                self.gpu_index,     # GPU index
                self.W_emb.handle,  # Parameter tensor (to be updated)
                self.dW_emb.handle, # Gradient tensor
                self.m_W_emb.handle,# Adam m tensor (1st moment)
                self.v_W_emb.handle,# Adam v tensor (2nd moment)
                num_elements_W,     # Total number of elements in W_emb
                t,                  # Timestep (for bias correction)
                lr,                 # Learning rate
                beta1,              # Adam beta1
                beta2,              # Adam beta2
                ADAM_EPS,           # Adam epsilon (for numerical stability)
                weight_decay        # Weight decay factor (L2 regularization)
            )
            if success == 0:
                raise RuntimeError(f"C driver execute_adam_update_on_gpu failed for Embedding W (t={t}). Check C driver output.")

            # After update, it's good practice to zero the gradient buffer
            # if the optimizer kernel doesn't do it implicitly.
            # Assuming the Adam kernel uses the gradient but doesn't zero it:
            self.dW_emb._zero_initialize() # Prepare for next gradient accumulation

    def get_state(self) -> Dict[str, np.ndarray]:
        """
        Retrieves the layer's state (weights and optimizer moments) as NumPy arrays.

        Reads the relevant GPUTensors into host memory.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing NumPy arrays for
            'W_emb', 'm_W_emb', and 'v_W_emb'.
        """
        self._check_freed()
        state: Dict[str, np.ndarray] = {}

        # Read Embedding Weights
        state['W_emb'] = np.zeros((self.V, self.E), dtype=FP_TYPE)
        if self.W_emb.size > 0:
            self.W_emb.read(state['W_emb'])

        # Read Adam Optimizer States (ensure correct shape and type)
        adam_shape = (self.V, self.E)
        state['m_W_emb'] = np.zeros(adam_shape, dtype=np.float32)
        state['v_W_emb'] = np.zeros(adam_shape, dtype=np.float32)
        if self.m_W_emb.size > 0:
            self.m_W_emb.read(state['m_W_emb'])
        if self.v_W_emb.size > 0:
            self.v_W_emb.read(state['v_W_emb'])

        return state

    def set_state(self, state: Dict[str, np.ndarray]):
        """
        Loads the layer's state (weights and optimizer moments) from NumPy arrays.

        Writes the provided NumPy arrays to the corresponding GPUTensors.

        Args:
            state: A dictionary containing NumPy arrays for 'W_emb', 'm_W_emb',
                   and 'v_W_emb'.

        Raises:
            KeyError: If a required key is missing in the `state` dictionary.
            ValueError: If the shape or type of the loaded arrays mismatches
                        the expected dimensions. (Caught by GPUTensor.write)
        """
        self._check_freed()
        if DEBUG_PRINTS: print(f"[EmbeddingLayer] Setting state from checkpoint...")
        try:
            # Write Embedding Weights (ensure correct type)
            if 'W_emb' in state and self.W_emb.size > 0:
                self.W_emb.write(state['W_emb'].astype(FP_TYPE))

            # Write Adam Optimizer States (ensure correct type np.float32)
            if 'm_W_emb' in state and self.m_W_emb.size > 0:
                self.m_W_emb.write(state['m_W_emb'].astype(np.float32))
            if 'v_W_emb' in state and self.v_W_emb.size > 0:
                self.v_W_emb.write(state['v_W_emb'].astype(np.float32))

            if DEBUG_PRINTS: print(f"[EmbeddingLayer] State loaded successfully for {self.V} x {self.E} embeddings.")

        except KeyError as e:
            print(f"[EmbeddingLayer] WARNING: Missing key '{e}' in checkpoint state dictionary while loading EmbeddingLayer.")
        except Exception as e:
            # Catch potential errors during GPU write (e.g., shape mismatch)
            print(f"[EmbeddingLayer] ERROR setting state: {e}")
            # Consider re-raising or handling more gracefully depending on desired behavior.

    def initialize_weights(self):
        """
        Initializes the embedding weights (W_emb) using a uniform distribution.

        Initializes weights in the range [-limit, limit], where limit is
        `sqrt(1 / embedding_dim)`, a common heuristic.
        """
        self._check_freed()
        # Calculate initialization range limit.
        limit = np.sqrt(1.0 / self.E)
        # Generate weights using uniform distribution.
        weights = np.random.uniform(-limit, limit, (self.V, self.E)).astype(FP_TYPE)
        # Write initialized weights to the GPU tensor.
        self.W_emb.write(weights)
        print(f"[EmbeddingLayer] Embedding weights (W_emb) initialized ({self.V}x{self.E}) with uniform U[-{limit:.4f}, {limit:.4f}].")
        del weights # Free host memory

    def free(self):
        """Releases all GPU resources allocated by this layer."""
        if not self._is_freed:
            if DEBUG_PRINTS: print(f"[EmbeddingLayer] Freeing GPU resources...")
            # List all GPUTensor attributes of this layer.
            tensors_to_free = [getattr(self, attr) for attr in vars(self) if isinstance(getattr(self, attr, None), GPUTensor)]
            for tensor in tensors_to_free:
                if tensor:
                    try:
                        tensor.free()
                    except Exception as e:
                        print(f"[EmbeddingLayer] Warning: Error freeing tensor {getattr(tensor,'name','Unknown')} during layer cleanup: {e}")
            # Mark the layer as freed.
            self._is_freed = True
            if DEBUG_PRINTS: print(f"[EmbeddingLayer] Resources freed.")


    def __del__(self):
        """Ensures resources are freed when the object is garbage collected."""
        if not getattr(self, '_is_freed', True):
            self.free()

    def train(self):
        """Sets the layer to training mode (currently no effect)."""
        # Placeholder for potential future use (e.g., dropout).
        pass

    def eval(self):
        """Sets the layer to evaluation mode (currently no effect)."""
        # Placeholder for potential future use.
        pass


# --- Linear Layer ---

class LinearLayer:
    """
    Represents a standard fully connected (dense) linear layer.

    Performs the operation: `output = input @ W + b`, where `@` is matrix
    multiplication, `W` is the weight matrix, and `b` is the bias vector.
    Operations (forward, backward, update) are executed using C/OpenCL kernels.

    Assumes input is flattened along batch and sequence dimensions (M_flat = B * S).

    Attributes:
        M_flat (int): Maximum flattened dimension (Batch Size * Sequence Length).
        E_in (int): Input dimension (number of features per element in M_flat).
        E_out (int): Output dimension (number of features per element in M_flat).
        gpu_index (int): GPU device index.
        W (GPUTensor): GPU tensor for the weights (E_in x E_out).
        b (GPUTensor): GPU tensor for the biases (E_out).
        dW (GPUTensor): GPU tensor for the gradients of W (E_in x E_out).
        db (GPUTensor): GPU tensor for the gradients of b (E_out).
        m_W, v_W (GPUTensor): Adam optimizer states for W.
        m_b, v_b (GPUTensor): Adam optimizer states for b.
        output (GPUTensor): GPU tensor for the layer's output (M_flat x E_out).
        d_input (GPUTensor): GPU tensor for the gradient w.r.t. the layer's input (M_flat x E_in).
        input_handle_cache (Optional[GPU_BUFFER_HANDLE]): Cached handle of the input tensor.
        current_M_flat (int): Actual flattened dimension for the current batch.
    """
    def __init__(self, batch_size_seq_len_flat: int, input_dim: int, output_dim: int, gpu_index: int = GPU_INDEX):
        """
        Initializes the LinearLayer.

        Args:
            batch_size_seq_len_flat: The maximum expected flattened dimension
                                     (max_batch_size * seq_len). Buffers are
                                     allocated based on this size.
            input_dim: The number of input features (size of the last dimension
                       of the input tensor).
            output_dim: The desired number of output features.
            gpu_index: The index of the GPU device to use.
        """
        self.M_flat: int = batch_size_seq_len_flat # Max size for buffer allocation
        self.E_in: int = input_dim
        self.E_out: int = output_dim
        self.gpu_index: int = gpu_index
        self._is_freed: bool = False
        # Calculate item sizes.
        itemsize_fp = FP_TYPE().itemsize
        itemsize_adam = np.float32().itemsize

        # Allocate GPU tensors for weights, biases, gradients, Adam states,
        # output buffer, and input gradient buffer.
        # Biases and gradients/Adam states are zero-initialized. Weights initialized later.
        self.W = GPUTensor(self.E_in * self.E_out * itemsize_fp, gpu_index, name="LinearW")
        self.b = GPUTensor(self.E_out * itemsize_fp, gpu_index, name="LinearB", zero_init=True)
        self.dW = GPUTensor(self.E_in * self.E_out * itemsize_fp, gpu_index, name="dLinearW", zero_init=True)
        self.db = GPUTensor(self.E_out * itemsize_fp, gpu_index, name="dLinearB", zero_init=True)
        self.m_W = GPUTensor(self.E_in * self.E_out * itemsize_adam, gpu_index, name="m_LinearW", zero_init=True)
        self.v_W = GPUTensor(self.E_in * self.E_out * itemsize_adam, gpu_index, name="v_LinearW", zero_init=True)
        self.m_b = GPUTensor(self.E_out * itemsize_adam, gpu_index, name="m_LinearB", zero_init=True)
        self.v_b = GPUTensor(self.E_out * itemsize_adam, gpu_index, name="v_LinearB", zero_init=True)

        # Allocate output and input gradient buffers based on max M_flat size.
        self.output = GPUTensor(self.M_flat * self.E_out * itemsize_fp, gpu_index, name="LinearOutput")
        self.d_input = GPUTensor(self.M_flat * self.E_in * itemsize_fp, gpu_index, name="dLinearInput")

        # Cache for input tensor handle during forward pass.
        self.input_handle_cache: Optional[GPU_BUFFER_HANDLE] = None
        # Actual M_flat for the current batch (can be <= self.M_flat).
        self.current_M_flat: int = self.M_flat

        if DEBUG_PRINTS: print(f"[LinearLayer] Initialized: In={self.E_in}, Out={self.E_out}, Max M_flat={self.M_flat} on GPU {self.gpu_index}")

    def _check_freed(self):
        """Raises RuntimeError if operations are attempted on a freed layer."""
        if self._is_freed:
            raise RuntimeError("Operation attempted on a freed LinearLayer.")

    def set_current_batch_m_flat(self, m_flat: int):
        """
        Sets the actual flattened dimension (M_flat) for the current batch.

        This is needed because the last batch might be smaller than the maximum
        size used for buffer allocation.

        Args:
            m_flat: The actual current batch_size * seq_len.

        Raises:
            ValueError: If `m_flat` exceeds the maximum allocated size `self.M_flat`.
        """
        self._check_freed()
        if m_flat > self.M_flat:
            raise ValueError(f"Current M_flat ({m_flat}) cannot exceed allocated max M_flat ({self.M_flat}) for LinearLayer.")
        if m_flat < 0:
             raise ValueError(f"Current M_flat ({m_flat}) cannot be negative.")
        self.current_M_flat = m_flat

    def forward(self, input_tensor: GPUTensor) -> GPUTensor:
        """
        Performs the forward pass: `output = input @ W + b`.

        Args:
            input_tensor: A GPUTensor containing the input data, expected to be
                          flattened to shape (current_M_flat x E_in).

        Returns:
            GPUTensor: The `output` tensor containing the result
                       (shape current_M_flat x E_out).

        Raises:
            RuntimeError: If C driver execution fails for MatMul or AddBias.
            AttributeError: If `c_driver` is None.
        """
        self._check_freed()
        if c_driver is None:
            raise AttributeError("C driver library is not loaded. Cannot perform Linear forward pass.")

        # Cache input handle for backward pass.
        self.input_handle_cache = input_tensor.handle
        if self.input_handle_cache is None:
             raise RuntimeError("Input tensor handle is None in LinearLayer forward.")
        if self.W.handle is None or self.b.handle is None or self.output.handle is None:
             raise RuntimeError("Weight, Bias, or Output handle is None in LinearLayer forward.")


        # 1. Matrix Multiplication: output = input @ W
        # C kernel expects (M, N, K) where Result(MxN) = A(MxK) @ B(KxN)
        # Here: M=current_M_flat, N=E_out, K=E_in. Input A is input_tensor, Input B is W.
        success_matmul = c_driver.execute_matmul_on_gpu(
            self.gpu_index,           # GPU index
            self.input_handle_cache,  # Input A (M x K)
            self.W.handle,            # Input B (K x N) -> Needs to be transposed in C or handled! Assume C handles W as (E_in x E_out) directly.
            self.output.handle,       # Output C (M x N)
            1,                        # Transpose A (0=No, 1=Yes)? Assuming input is (M x K) - C code expects (K x M)? Check C kernel docs! Let's assume C expects A(MxK), W(KxN) -> Out(MxN)
                                      # Let's re-read common convention: Output(MxN) = Input(MxK) * Weight(KxN).
                                      # Our shapes: Output(M_flat x E_out) = Input(M_flat x E_in) * Weight(E_in x E_out)
                                      # So M=M_flat, N=E_out, K=E_in. C func: (gpu, A, B, C, transA, M, N, K)
                                      # Call should be: (gpu, input, W, output, 0, M_flat, E_out, E_in) <-- Check this matches C kernel expectation!
                                      # *** Assuming C kernel expects A(M x K), B(K x N) format ***
            0,                        # Transpose A (No)
            self.current_M_flat,      # M dimension
            self.E_out,               # N dimension
            self.E_in                 # K dimension
         )
        if success_matmul == 0:
            raise RuntimeError("C driver execute_matmul_on_gpu failed for Linear forward. Check C driver output.")

        # 2. Add Bias: output = output + b
        # C kernel expects (gpu, io_tensor, bias_tensor, M, N)
        success_addbias = c_driver.execute_add_bias_on_gpu(
            self.gpu_index,        # GPU index
            self.output.handle,    # Input/Output tensor (M x N)
            self.b.handle,         # Bias tensor (N)
            self.current_M_flat,   # M dimension
            self.E_out             # N dimension (bias dimension)
        )
        if success_addbias == 0:
            raise RuntimeError("C driver execute_add_bias_on_gpu failed for Linear forward. Check C driver output.")

        return self.output

    def backward(self, d_output: GPUTensor) -> GPUTensor:
        """
        Performs the backward pass for the linear layer.

        Computes:
        - Gradient w.r.t. weights: `dW = input^T @ d_output`
        - Gradient w.r.t. bias: `db = sum(d_output, axis=0)`
        - Gradient w.r.t. input: `d_input = d_output @ W^T`

        Args:
            d_output: A GPUTensor containing the gradients flowing back from the
                      subsequent layer (shape current_M_flat x E_out).

        Returns:
            GPUTensor: The `d_input` tensor containing the gradients w.r.t. the
                       layer's input (shape current_M_flat x E_in).

        Raises:
            RuntimeError: If input handle wasn't cached, handles are None, or C driver execution fails.
            AssertionError: If input handle cache is None.
            AttributeError: If `c_driver` is None.
        """
        self._check_freed()
        if c_driver is None:
            raise AttributeError("C driver library is not loaded. Cannot perform Linear backward pass.")
        # Ensure input handle was cached from forward pass.
        assert self.input_handle_cache is not None, "Input handle cache is None in LinearLayer backward. Forward pass must precede backward."
        if d_output.handle is None or self.db.handle is None or self.W.handle is None or self.d_input.handle is None or self.dW.handle is None:
             raise RuntimeError("Required handles (d_output, db, W, d_input, dW) are None in LinearLayer backward.")


        # 1. Compute Gradient w.r.t. Bias (db): Sum d_output along the batch dimension.
        # C kernel execute_reduce_sum_gpu(gpu, input, output, axis, M, N)
        # Here: Input=d_output(M_flat x E_out), Output=db(E_out), axis=0 (reduce rows)
        # M=current_M_flat, N=E_out
        success_reducesum = c_driver.execute_reduce_sum_gpu(
            self.gpu_index,      # GPU index
            d_output.handle,     # Input tensor (M x N)
            self.db.handle,      # Output tensor (N)
            0,                   # Axis to reduce (0 for summing rows)
            self.current_M_flat, # M dimension
            self.E_out           # N dimension
        )
        if success_reducesum == 0:
            raise RuntimeError("C driver execute_reduce_sum_gpu failed for Linear backward bias gradient. Check C driver output.")

        # 2. Compute Gradients w.r.t. Weights (dW) and Input (d_input) using MatMul Backward kernel.
        # C kernel: execute_matmul_backward_on_gpu(gpu, A, B, dC, dA, dB, transA, M, N, K)
        # Original forward: C(MxN) = A(MxK) @ B(KxN) --> Output = Input @ W
        #   M=current_M_flat, N=E_out, K=E_in
        #   A = input_tensor, B = W, C = output, dC = d_output
        # We need:
        #   dA (d_input) = dC @ B^T = d_output @ W^T  (Shape: M x K)
        #   dB (dW)      = A^T @ dC = input^T @ d_output (Shape: K x N)
        # Call: (gpu, input, W, d_output, d_input, dW, 0, M_flat, E_out, E_in) <-- Check if C kernel matches this exactly!
        success_matmul_bwd = c_driver.execute_matmul_backward_on_gpu(
            self.gpu_index,           # GPU index
            self.input_handle_cache,  # Original Input A (M x K)
            self.W.handle,            # Original Input B (K x N)
            d_output.handle,          # Gradient dC (M x N)
            self.d_input.handle,      # Output dA (gradient w.r.t. input A, shape M x K)
            self.dW.handle,           # Output dB (gradient w.r.t. input B, shape K x N)
            0,                        # Transpose A in original forward? (No)
            self.current_M_flat,      # M dimension
            self.E_out,               # N dimension
            self.E_in                 # K dimension
        )
        if success_matmul_bwd == 0:
            raise RuntimeError("C driver execute_matmul_backward_on_gpu failed for Linear backward. Check C driver output.")

        # Return the gradient computed for the layer's input.
        return self.d_input

    def clip_gradients(self, clip_value: float):
        """
        Clips the L2 norm of the weight (dW) and bias (db) gradients.

        Reads gradient tensors from GPU, calculates norms on host, scales
        if necessary, and writes back to GPU.

        Args:
            clip_value: The maximum allowed L2 norm. If `<= 0`, clipping is skipped.
        """
        self._check_freed()
        # Skip if clipping is disabled or gradients are zero-size.
        if clip_value <= 0 or (self.dW.size == 0 and self.db.size == 0):
            return

        grad_W_host: Optional[np.ndarray] = None
        grad_b_host: Optional[np.ndarray] = None
        try:
            # Clip Weight Gradients (dW)
            if self.dW.size > 0:
                grad_W_host = np.zeros((self.E_in, self.E_out), dtype=FP_TYPE) # Host buffer
                self.dW.read(grad_W_host) # Read from GPU
                norm_W = np.linalg.norm(grad_W_host) # Calculate L2 norm
                scale_W = clip_value / (norm_W + 1e-6) if norm_W > clip_value else 1.0 # Calculate scale factor
                if scale_W < 1.0: # Apply scaling only if norm exceeds threshold
                    if DEBUG_PRINTS: print(f"[GradClip] Clipping Linear W gradient norm {norm_W:.4f} with scale {scale_W:.4f}")
                    grad_W_host *= scale_W
                    self.dW.write(grad_W_host) # Write back to GPU

            # Clip Bias Gradients (db)
            if self.db.size > 0:
                grad_b_host = np.zeros(self.E_out, dtype=FP_TYPE) # Host buffer
                self.db.read(grad_b_host) # Read from GPU
                norm_b = np.linalg.norm(grad_b_host) # Calculate L2 norm
                scale_b = clip_value / (norm_b + 1e-6) if norm_b > clip_value else 1.0 # Calculate scale factor
                if scale_b < 1.0: # Apply scaling only if norm exceeds threshold
                    if DEBUG_PRINTS: print(f"[GradClip] Clipping Linear b gradient norm {norm_b:.4f} with scale {scale_b:.4f}")
                    grad_b_host *= scale_b
                    self.db.write(grad_b_host) # Write back to GPU

        except Exception as e:
            print(f"[LinearLayer] ERROR during gradient clipping: {e}")
        finally:
             # Ensure host buffers are deleted
             if grad_W_host is not None: del grad_W_host
             if grad_b_host is not None: del grad_b_host

    def update(self, t: int, lr: float, beta1: float = 0.9, beta2: float = 0.999, weight_decay: float = 0.0):
        """
        Updates the weights (W) and biases (b) using the Adam optimizer rule on the GPU.

        Args:
            t: Current timestep (> 0) for bias correction.
            lr: Current learning rate.
            beta1: Adam beta1 parameter.
            beta2: Adam beta2 parameter.
            weight_decay: L2 regularization factor (applied to weights W, not biases b).

        Raises:
            RuntimeError: If the C driver Adam update execution fails.
            AttributeError: If `c_driver` is None.
        """
        self._check_freed()
        if c_driver is None:
            raise AttributeError("C driver library is not loaded. Cannot perform Linear update.")

        num_elements_W = self.E_in * self.E_out
        num_elements_b = self.E_out

        # Update Weights (W) with Adam and Weight Decay
        if num_elements_W > 0:
            success_W = c_driver.execute_adam_update_on_gpu(
                self.gpu_index, self.W.handle, self.dW.handle, self.m_W.handle, self.v_W.handle,
                num_elements_W, t, lr, beta1, beta2, ADAM_EPS, weight_decay
            )
            if success_W == 0:
                raise RuntimeError(f"C driver execute_adam_update_on_gpu failed for Linear W (t={t}). Check C driver output.")
            # Zero gradient after update
            self.dW._zero_initialize()


        # Update Biases (b) with Adam (NO Weight Decay typically)
        if num_elements_b > 0:
            success_b = c_driver.execute_adam_update_on_gpu(
                self.gpu_index, self.b.handle, self.db.handle, self.m_b.handle, self.v_b.handle,
                num_elements_b, t, lr, beta1, beta2, ADAM_EPS, 0.0 # weight_decay = 0.0 for bias
            )
            if success_b == 0:
                raise RuntimeError(f"C driver execute_adam_update_on_gpu failed for Linear b (t={t}). Check C driver output.")
            # Zero gradient after update
            self.db._zero_initialize()

    def get_state(self) -> Dict[str, np.ndarray]:
        """
        Retrieves the layer's state (W, b, and Adam moments) as NumPy arrays.

        Returns:
            Dict[str, np.ndarray]: Dictionary with state arrays.
        """
        self._check_freed()
        state: Dict[str, np.ndarray] = {}
        # Read Weights and Biases
        state['W'] = np.zeros((self.E_in, self.E_out), dtype=FP_TYPE)
        state['b'] = np.zeros(self.E_out, dtype=FP_TYPE)
        if self.W.size > 0: self.W.read(state['W'])
        if self.b.size > 0: self.b.read(state['b'])

        # Read Adam Optimizer States for W
        adam_shape_W = (self.E_in, self.E_out)
        state['m_W'] = np.zeros(adam_shape_W, dtype=np.float32)
        state['v_W'] = np.zeros(adam_shape_W, dtype=np.float32)
        if self.m_W.size > 0: self.m_W.read(state['m_W'])
        if self.v_W.size > 0: self.v_W.read(state['v_W'])

        # Read Adam Optimizer States for b
        adam_shape_b = (self.E_out,)
        state['m_b'] = np.zeros(adam_shape_b, dtype=np.float32)
        state['v_b'] = np.zeros(adam_shape_b, dtype=np.float32)
        if self.m_b.size > 0: self.m_b.read(state['m_b'])
        if self.v_b.size > 0: self.v_b.read(state['v_b'])

        return state

    def set_state(self, state: Dict[str, np.ndarray]):
        """
        Loads the layer's state (W, b, and Adam moments) from NumPy arrays.

        Args:
            state: Dictionary containing the state arrays.

        Raises:
            KeyError: If a required key is missing.
            Exception: Catches potential errors during GPU writes.
        """
        self._check_freed()
        if DEBUG_PRINTS: print(f"[LinearLayer] Setting state from checkpoint...")
        try:
            # Write Weights and Biases
            if 'W' in state and self.W.size > 0: self.W.write(state['W'].astype(FP_TYPE))
            if 'b' in state and self.b.size > 0: self.b.write(state['b'].astype(FP_TYPE))
            # Write Adam States for W
            if 'm_W' in state and self.m_W.size > 0: self.m_W.write(state['m_W'].astype(np.float32))
            if 'v_W' in state and self.v_W.size > 0: self.v_W.write(state['v_W'].astype(np.float32))
            # Write Adam States for b
            if 'm_b' in state and self.m_b.size > 0: self.m_b.write(state['m_b'].astype(np.float32))
            if 'v_b' in state and self.v_b.size > 0: self.v_b.write(state['v_b'].astype(np.float32))
            if DEBUG_PRINTS: print(f"[LinearLayer] State loaded successfully.")
        except KeyError as e:
            print(f"[LinearLayer] WARNING: Missing key '{e}' in checkpoint state dictionary while loading LinearLayer.")
        except Exception as e:
            print(f"[LinearLayer] ERROR setting state: {e}")

    def free(self):
        """Releases all GPU resources allocated by this layer."""
        if not self._is_freed:
            if DEBUG_PRINTS: print(f"[LinearLayer] Freeing GPU resources...")
            tensors_to_free = [getattr(self, attr) for attr in vars(self) if isinstance(getattr(self, attr, None), GPUTensor)]
            for tensor in tensors_to_free:
                if tensor:
                    try:
                        tensor.free()
                    except Exception as e:
                        print(f"[LinearLayer] Warning: Error freeing tensor {getattr(tensor,'name','Unknown')} during layer cleanup: {e}")
            self._is_freed = True
            if DEBUG_PRINTS: print(f"[LinearLayer] Resources freed.")

    def __del__(self):
        """Ensures resources are freed when the object is garbage collected."""
        if not getattr(self, '_is_freed', True):
            self.free()

    def train(self):
        """Sets the layer to training mode (no specific effect currently)."""
        pass

    def eval(self):
        """Sets the layer to evaluation mode (no specific effect currently)."""
        pass


# --- Bio-Inspired Associative Layer ---

class BioInspiredAssociativeLayer:
    """
    Implements a custom bio-inspired layer with associative features.

    Combines a standard linear transformation with GELU activation with
    more experimental features:
    - Optional dynamic token assignment based on prototype vectors.
    - Threshold-based spiking mechanism on activations.
    - Optional Hebbian learning updates on a recurrent weight matrix (W_hebb).
    - Optional prototype vector updates based on assigned activations.

    Attributes:
        B (int): Maximum batch size.
        S (int): Sequence length.
        E_in (int): Input embedding dimension.
        E_hidden (int): Hidden layer dimension.
        T (int): Number of token prototypes (0 to disable).
        hebbian_lr (float): Learning rate for Hebbian updates.
        spike_threshold (float): Threshold for spike generation.
        prototype_lr (float): Learning rate for prototype updates.
        gpu_index (int): GPU device index.
        W1, b1 (GPUTensor): Weights and biases for the initial linear transformation.
        dW1, db1 (GPUTensor): Gradients for W1 and b1.
        m_W1, v_W1, m_b1, v_b1 (GPUTensor): Adam states for W1 and b1.
        W_hebb (GPUTensor): Hebbian associative weight matrix (E_hidden x E_hidden).
        prototypes (Optional[GPUTensor]): Token prototype vectors (T x E_hidden).
        pre_gelu_activations (GPUTensor): Buffer for activations before GELU.
        hidden_activations (GPUTensor): Buffer for activations after GELU (layer output).
        d_pre_gelu (GPUTensor): Buffer for gradients w.r.t. pre-GELU activations.
        d_input (GPUTensor): Buffer for gradients w.r.t. the layer's input.
        spikes (GPUTensor): Buffer for thresholded spike outputs.
        token_indices (Optional[GPUTensor]): Buffer for assigned prototype indices per token.
        proto_sums_gpu, proto_counts_gpu (Optional[GPUTensor]): GPU buffers for accelerated prototype updates.
        input_handle_cache (Optional[GPU_BUFFER_HANDLE]): Cached input tensor handle.
        current_B (int): Current batch size.
        current_M_flat (int): Current flattened dimension (B * S).
    """
    def __init__(self, batch_size: int, seq_len: int, embedding_dim: int, hidden_dim: int, num_prototypes: int,
                 hebbian_lr: float = 0.001, spike_threshold: float = 0.5, prototype_lr: float = 0.01, gpu_index: int = GPU_INDEX):
        """
        Initializes the BioInspiredAssociativeLayer.

        Args:
            batch_size: Maximum expected batch size.
            seq_len: Sequence length.
            embedding_dim: Dimension of the input vectors (from previous layer).
            hidden_dim: Dimension of the hidden state and output of this layer.
            num_prototypes: Number of dynamic token prototypes to use (set to 0 to disable).
            hebbian_lr: Learning rate for the Hebbian weight updates.
            spike_threshold: Activation threshold for generating 'spikes'.
            prototype_lr: Learning rate for updating token prototypes.
            gpu_index: The index of the GPU device to use.
        """
        self.B: int = batch_size
        self.S: int = seq_len
        self.E_in: int = embedding_dim
        self.E_hidden: int = hidden_dim
        self.T: int = num_prototypes # Number of prototypes
        self.hebbian_lr: float = hebbian_lr
        self.spike_threshold: float = spike_threshold
        self.prototype_lr: float = prototype_lr
        self.gpu_index: int = gpu_index
        self._is_freed: bool = False
        # Calculate item sizes.
        itemsize_fp = FP_TYPE().itemsize
        itemsize_adam = np.float32().itemsize
        itemsize_int = INT_TYPE().itemsize

        # Max flattened dimension for buffer allocation.
        self.max_M_flat: int = self.B * self.S
        # Current batch/flattened size (updated by set_current_batch_shape).
        self.current_B: int = self.B
        self.current_M_flat: int = self.max_M_flat

        # --- Allocate GPUTensors ---
        # Linear transformation part (W1, b1) + Adam states + Gradients
        self.W1 = GPUTensor(self.E_in * self.E_hidden * itemsize_fp, gpu_index, name="BioW1")
        self.b1 = GPUTensor(self.E_hidden * itemsize_fp, gpu_index, name="BioB1", zero_init=True)
        self.dW1 = GPUTensor(self.E_in * self.E_hidden * itemsize_fp, gpu_index, name="dBioW1", zero_init=True)
        self.db1 = GPUTensor(self.E_hidden * itemsize_fp, gpu_index, name="dBioB1", zero_init=True)
        self.m_W1 = GPUTensor(self.E_in * self.E_hidden * itemsize_adam, gpu_index, name="m_BioW1", zero_init=True)
        self.v_W1 = GPUTensor(self.E_in * self.E_hidden * itemsize_adam, gpu_index, name="v_BioW1", zero_init=True)
        self.m_b1 = GPUTensor(self.E_hidden * itemsize_adam, gpu_index, name="m_BioB1", zero_init=True)
        self.v_b1 = GPUTensor(self.E_hidden * itemsize_adam, gpu_index, name="v_BioB1", zero_init=True)

        # Hebbian weights (initialized to zero, updated during training)
        self.W_hebb = GPUTensor(self.E_hidden * self.E_hidden * itemsize_fp, gpu_index, name="BioW_hebb", zero_init=True)

        # Prototypes (only if T > 0, initialized later)
        self.prototypes: Optional[GPUTensor] = None
        if self.T > 0:
            self.prototypes = GPUTensor(self.T * self.E_hidden * itemsize_fp, gpu_index, name="BioPrototypes")

        # Calculate max buffer sizes based on max batch size.
        self.num_elements_hidden_flat_max: int = self.max_M_flat * self.E_hidden
        self.num_elements_tokens_max: int = self.max_M_flat # For token indices
        self.num_elements_input_flat_max: int = self.max_M_flat * self.E_in

        # Intermediate activation and gradient buffers.
        self.input_handle_cache: Optional[GPU_BUFFER_HANDLE] = None
        self.pre_gelu_activations = GPUTensor(self.num_elements_hidden_flat_max * itemsize_fp, gpu_index, name="BioPreGELUActs")
        self.hidden_activations = GPUTensor(self.num_elements_hidden_flat_max * itemsize_fp, gpu_index, name="BioHiddenActs") # Layer output
        self.d_pre_gelu = GPUTensor(self.num_elements_hidden_flat_max * itemsize_fp, gpu_index, name="dBioPreGELU") # Grad w.r.t pre-activation
        self.d_input = GPUTensor(self.num_elements_input_flat_max * itemsize_fp, gpu_index, name="dBioInput") # Grad w.r.t layer input

        # Buffers for spiking and token assignment.
        self.spikes = GPUTensor(self.num_elements_hidden_flat_max * itemsize_fp, gpu_index, name="BioSpikes") # Output of thresholding
        self.token_indices: Optional[GPUTensor] = None # Assigned prototype index per activation
        if self.T > 0:
             self.token_indices = GPUTensor(self.num_elements_tokens_max * itemsize_int, gpu_index, name="BioTokenIndices")

        # Optional GPU buffers for accelerated prototype update (segmented sum).
        self.proto_sums_gpu: Optional[GPUTensor] = None
        self.proto_counts_gpu: Optional[GPUTensor] = None
        # Allocate only if using prototypes AND GPU update is enabled AND C functions are available.
        if self.T > 0 and USE_GPU_PROTOTYPE_UPDATE and CAN_USE_GPU_PROTO_UPDATE:
             self.proto_sums_gpu = GPUTensor(self.T * self.E_hidden * itemsize_fp, gpu_index, name="ProtoSumsGPU", zero_init=True)
             self.proto_counts_gpu = GPUTensor(self.T * itemsize_int, gpu_index, name="ProtoCountsGPU", zero_init=True)

        # Log initialization details.
        if DEBUG_PRINTS:
            print(f"[BioLayer] Initialized: Max B={self.B}, S={self.S}, E_in={self.E_in}, E_hidden={self.E_hidden}, T={self.T} on GPU {self.gpu_index}")
            print(f"[BioLayer] Max Flattened M dimension: {self.max_M_flat}")
            gpu_proto_status = 'Disabled (T=0)'
            if self.T > 0:
                 gpu_proto_status = 'Enabled' if USE_GPU_PROTOTYPE_UPDATE and CAN_USE_GPU_PROTO_UPDATE else 'Host Fallback'
            print(f"[BioLayer] Prototype Update Method: {gpu_proto_status}")

    def _check_freed(self):
        """Raises RuntimeError if operations are attempted on a freed layer."""
        if self._is_freed:
            raise RuntimeError("Operation attempted on a freed BioInspiredAssociativeLayer.")

    def set_current_batch_shape(self, b: int, s: int):
        """
        Sets the actual batch size (B) and sequence length (S) for the current operation.

        Updates `current_B` and `current_M_flat`.

        Args:
            b: Current batch size.
            s: Current sequence length (must match configured `self.S`).

        Raises:
            ValueError: If `b` > `self.B`, `s` != `self.S`, or sizes are negative.
        """
        self._check_freed()
        if b > self.B or s != self.S:
            raise ValueError(f"Current batch shape ({b}, {s}) is incompatible with layer configuration "
                             f"(Max B={self.B}, Fixed S={self.S}).")
        if b < 0 or s < 0:
             raise ValueError(f"Current batch shape ({b}, {s}) cannot have negative dimensions.")
        self.current_B = b
        self.current_M_flat = b * s # Recalculate flattened dimension

    def forward(self, input_tensor: GPUTensor) -> GPUTensor:
        """
        Performs the forward pass of the Bio-Inspired Layer.

        1. Linear transformation: `pre_gelu = input @ W1 + b1`
        2. GELU activation: `hidden_activations = GELU(pre_gelu)`
        3. (Optional) Dynamic Token Assignment: Assigns each hidden activation
           vector to the nearest prototype (updates `token_indices`).
        4. Thresholded Spiking: Generates binary spikes based on `hidden_activations`
           and `spike_threshold` (updates `spikes`).

        Args:
            input_tensor: GPUTensor containing input data (shape current_M_flat x E_in).

        Returns:
            GPUTensor: The `hidden_activations` tensor (output of GELU,
                       shape current_M_flat x E_hidden). Note: Subsequent layers
                       might use `hidden_activations` or potentially `spikes`.

        Raises:
            RuntimeError: If C driver execution fails for any kernel.
            AttributeError: If `c_driver` is None.
        """
        self._check_freed()
        if c_driver is None:
            raise AttributeError("C driver library is not loaded. Cannot perform BioLayer forward pass.")

        self.input_handle_cache = input_tensor.handle # Cache for backward pass
        if self.input_handle_cache is None: raise RuntimeError("Input tensor handle is None in BioLayer forward.")
        if self.W1.handle is None or self.b1.handle is None or self.pre_gelu_activations.handle is None or self.hidden_activations.handle is None or self.spikes.handle is None:
             raise RuntimeError("Required handles (W1, b1, pre_gelu, hidden, spikes) are None in BioLayer forward.")
        if self.T > 0 and (self.prototypes is None or self.prototypes.handle is None or self.token_indices is None or self.token_indices.handle is None):
            raise RuntimeError("Prototype or token_indices handle is None when T > 0 in BioLayer forward.")


        # Calculate number of elements for current batch size.
        current_num_elements_hidden = self.current_M_flat * self.E_hidden

        # 1. Linear transformation: pre_gelu = input @ W1
        # Shapes: Input(M_flat x E_in), W1(E_in x E_hidden), Output(M_flat x E_hidden)
        # C func: (gpu, A, B, C, transA, M, N, K) -> (gpu, input, W1, pre_gelu, 0, M_flat, E_hidden, E_in)
        success_matmul = c_driver.execute_matmul_on_gpu(
            self.gpu_index, self.input_handle_cache, self.W1.handle, self.pre_gelu_activations.handle,
            0, self.current_M_flat, self.E_hidden, self.E_in
        )
        if success_matmul == 0:
            raise RuntimeError("C driver execute_matmul_on_gpu failed for BioLayer W1 forward.")

        # 2. Add bias: pre_gelu = pre_gelu + b1
        # C func: (gpu, io_tensor, bias_tensor, M, N) -> (gpu, pre_gelu, b1, M_flat, E_hidden)
        success_addbias = c_driver.execute_add_bias_on_gpu(
            self.gpu_index, self.pre_gelu_activations.handle, self.b1.handle,
            self.current_M_flat, self.E_hidden
        )
        if success_addbias == 0:
            raise RuntimeError("C driver execute_add_bias_on_gpu failed for BioLayer b1 forward.")

        # 3. GELU Activation: hidden_activations = GELU(pre_gelu)
        # C func: (gpu, input, output, num_elements)
        success_gelu = c_driver.execute_gelu_on_gpu(
            self.gpu_index, self.pre_gelu_activations.handle, self.hidden_activations.handle,
            current_num_elements_hidden
        )
        if success_gelu == 0:
            raise RuntimeError("C driver execute_gelu_on_gpu failed for BioLayer forward.")

        # 4. (Optional) Dynamic Token Assignment
        if self.T > 0:
            # Assigns each hidden activation vector to the closest prototype.
            # C func: (gpu, hidden_activations, prototypes, token_indices_out, B, S, E_hidden, T)
            success_token_assign = c_driver.execute_dynamic_token_assignment_gpu(
                self.gpu_index,
                self.hidden_activations.handle, # Input activations (B*S x E_hidden)
                self.prototypes.handle,         # Prototypes (T x E_hidden)
                self.token_indices.handle,      # Output indices (B*S)
                self.current_B,                 # Batch size
                self.S,                         # Sequence length
                self.E_hidden,                  # Hidden dimension
                self.T                          # Number of prototypes
            )
            if success_token_assign == 0:
                raise RuntimeError("C driver execute_dynamic_token_assignment_gpu failed for BioLayer forward.")

        # 5. Thresholded Spiking: spikes = (hidden_activations > threshold) ? 1.0 : 0.0
        # C func: (gpu, input_activations, output_spikes, threshold, num_elements)
        success_spike = c_driver.execute_threshold_spike_on_gpu(
            self.gpu_index, self.hidden_activations.handle, self.spikes.handle,
            self.spike_threshold, current_num_elements_hidden
        )
        if success_spike == 0:
            raise RuntimeError("C driver execute_threshold_spike_on_gpu failed for BioLayer forward.")

        # Return the post-GELU activations (main output of this layer).
        return self.hidden_activations

    def backward(self, d_output: GPUTensor) -> GPUTensor:
        """
        Performs the backward pass for the Bio-Inspired Layer.

        Propagates gradients from `d_output` (gradient w.r.t. `hidden_activations`)
        back through the GELU activation and the linear transformation (W1, b1).
        Computes `dW1`, `db1`, and `d_input`.

        Note: This standard backward pass ignores gradients related to the
        spiking, Hebbian learning, or prototype assignment parts, as those
        are typically handled by separate update rules.

        Args:
            d_output: GPUTensor containing gradients w.r.t. the `hidden_activations`
                      (shape current_M_flat x E_hidden).

        Returns:
            GPUTensor: The `d_input` tensor containing gradients w.r.t. the
                       layer's input (shape current_M_flat x E_in).

        Raises:
            RuntimeError: If handles are None or C driver execution fails.
            AssertionError: If input handle cache is None.
            AttributeError: If `c_driver` is None.
        """
        self._check_freed()
        if c_driver is None:
            raise AttributeError("C driver library is not loaded. Cannot perform BioLayer backward pass.")
        assert self.input_handle_cache is not None, "Input handle cache is None in BioLayer backward."
        if d_output.handle is None or self.pre_gelu_activations.handle is None or self.d_pre_gelu.handle is None \
           or self.db1.handle is None or self.W1.handle is None or self.d_input.handle is None or self.dW1.handle is None:
            raise RuntimeError("Required handles are None in BioLayer backward.")

        current_num_elements_hidden = self.current_M_flat * self.E_hidden

        # 1. Backward through GELU: Compute d_pre_gelu = d_output * GELU_derivative(pre_gelu)
        # C func: (gpu, pre_gelu_acts, d_output, d_pre_gelu_out, num_elements)
        success_gelu_bwd = c_driver.execute_gelu_backward_on_gpu(
            self.gpu_index, self.pre_gelu_activations.handle, d_output.handle,
            self.d_pre_gelu.handle, current_num_elements_hidden
        )
        if success_gelu_bwd == 0:
            raise RuntimeError("C driver execute_gelu_backward_on_gpu failed for BioLayer backward.")

        # 2. Backward bias (db1): Sum d_pre_gelu along the batch dimension.
        # C func: (gpu, input, output, axis, M, N) -> (gpu, d_pre_gelu, db1, 0, M_flat, E_hidden)
        success_reducesum = c_driver.execute_reduce_sum_gpu(
            self.gpu_index, self.d_pre_gelu.handle, self.db1.handle,
            0, self.current_M_flat, self.E_hidden
        )
        if success_reducesum == 0:
            raise RuntimeError("C driver execute_reduce_sum_gpu failed for BioLayer backward bias gradient.")

        # 3. Backward MatMul (dW1 and d_input):
        # Uses d_pre_gelu as the incoming gradient (dC).
        # Forward was: pre_gelu(MxN) = input(MxK) @ W1(KxN)
        #   M=M_flat, N=E_hidden, K=E_in
        #   A = input, B = W1, C = pre_gelu, dC = d_pre_gelu
        # We need:
        #   dA (d_input) = dC @ B^T = d_pre_gelu @ W1^T (Shape: M x K)
        #   dB (dW1) = A^T @ dC = input^T @ d_pre_gelu (Shape: K x N)
        # C func: (gpu, A, B, dC, dA, dB, transA, M, N, K)
        # Call: (gpu, input, W1, d_pre_gelu, d_input, dW1, 0, M_flat, E_hidden, E_in)
        success_matmul_bwd = c_driver.execute_matmul_backward_on_gpu(
            self.gpu_index, self.input_handle_cache, self.W1.handle, self.d_pre_gelu.handle,
            self.d_input.handle, self.dW1.handle,
            0, self.current_M_flat, self.E_hidden, self.E_in
        )
        if success_matmul_bwd == 0:
            raise RuntimeError("C driver execute_matmul_backward_on_gpu failed for BioLayer backward MatMul.")

        return self.d_input

    def clip_gradients(self, clip_value: float):
        """
        Clips the L2 norm of the W1 and b1 gradients.

        Args:
            clip_value: The maximum allowed L2 norm. If `<= 0`, clipping is skipped.
        """
        self._check_freed()
        # Skip if clipping is disabled or gradients are zero-size.
        if clip_value <= 0 or (self.dW1.size == 0 and self.db1.size == 0) :
            return

        grad_W1_host: Optional[np.ndarray] = None
        grad_b1_host: Optional[np.ndarray] = None
        try:
            # Clip W1 gradients
            if self.dW1.size > 0:
                grad_W1_host = np.zeros((self.E_in, self.E_hidden), dtype=FP_TYPE)
                self.dW1.read(grad_W1_host)
                norm_W1 = np.linalg.norm(grad_W1_host)
                scale_W1 = clip_value / (norm_W1 + 1e-6) if norm_W1 > clip_value else 1.0
                if scale_W1 < 1.0:
                    if DEBUG_PRINTS: print(f"[GradClip] Clipping BioLayer W1 gradient norm {norm_W1:.4f} with scale {scale_W1:.4f}")
                    grad_W1_host *= scale_W1
                    self.dW1.write(grad_W1_host)

            # Clip b1 gradients
            if self.db1.size > 0:
                grad_b1_host = np.zeros(self.E_hidden, dtype=FP_TYPE)
                self.db1.read(grad_b1_host)
                norm_b1 = np.linalg.norm(grad_b1_host)
                scale_b1 = clip_value / (norm_b1 + 1e-6) if norm_b1 > clip_value else 1.0
                if scale_b1 < 1.0:
                    if DEBUG_PRINTS: print(f"[GradClip] Clipping BioLayer b1 gradient norm {norm_b1:.4f} with scale {scale_b1:.4f}")
                    grad_b1_host *= scale_b1
                    self.db1.write(grad_b1_host)
        except Exception as e:
            print(f"[BioLayer] ERROR during gradient clipping: {e}")
        finally:
             # Ensure host buffers are deleted
             if grad_W1_host is not None: del grad_W1_host
             if grad_b1_host is not None: del grad_b1_host

    def update(self, t: int, lr: float, beta1: float = 0.9, beta2: float = 0.999, weight_decay: float = 0.0):
        """
        Updates the W1 weights and b1 biases using the Adam optimizer rule on the GPU.

        Args:
            t: Current timestep (> 0) for bias correction.
            lr: Current learning rate.
            beta1: Adam beta1 parameter.
            beta2: Adam beta2 parameter.
            weight_decay: L2 regularization factor (applied to W1, not b1).

        Raises:
            RuntimeError: If the C driver Adam update execution fails.
            AssertionError: If t <= 0.
            AttributeError: If `c_driver` is None.
        """
        self._check_freed()
        if c_driver is None:
            raise AttributeError("C driver library is not loaded. Cannot perform BioLayer update.")
        assert t > 0, "Timestep t must be > 0 for Adam update."

        num_elements_W1 = self.E_in * self.E_hidden
        num_elements_b1 = self.E_hidden

        # Update W1 (with weight decay)
        if num_elements_W1 > 0:
            success_W1 = c_driver.execute_adam_update_on_gpu(
                self.gpu_index, self.W1.handle, self.dW1.handle, self.m_W1.handle, self.v_W1.handle,
                num_elements_W1, t, lr, beta1, beta2, ADAM_EPS, weight_decay
            )
            if success_W1 == 0:
                raise RuntimeError(f"C driver execute_adam_update_on_gpu failed for BioLayer W1 (t={t}).")
            self.dW1._zero_initialize() # Zero grad after update


        # Update b1 (without weight decay)
        if num_elements_b1 > 0:
            success_b1 = c_driver.execute_adam_update_on_gpu(
                self.gpu_index, self.b1.handle, self.db1.handle, self.m_b1.handle, self.v_b1.handle,
                num_elements_b1, t, lr, beta1, beta2, ADAM_EPS, 0.0 # No weight decay for bias
            )
            if success_b1 == 0:
                raise RuntimeError(f"C driver execute_adam_update_on_gpu failed for BioLayer b1 (t={t}).")
            self.db1._zero_initialize() # Zero grad after update

    # --- Special Bio-Inspired Update Rules ---

    def hebbian_learn(self, pre_synaptic_activations: GPUTensor, post_synaptic_activations: GPUTensor):
        """
        Performs a Hebbian-like update on the W_hebb weights.

        Typically, this might involve correlating pre-synaptic (e.g., `hidden_activations`)
        and post-synaptic signals (e.g., `spikes`) to update the associative
        weights `W_hebb`. The exact rule depends on the C kernel implementation.

        Args:
            pre_synaptic_activations: GPUTensor representing pre-synaptic activity
                                     (e.g., `hidden_activations`, shape M_flat x E_hidden).
            post_synaptic_activations: GPUTensor representing post-synaptic activity
                                      (e.g., `spikes`, shape M_flat x E_hidden).

        Raises:
            RuntimeError: If C driver execution fails.
            AssertionError: If required handles are None.
            AttributeError: If `c_driver` is None.
        """
        self._check_freed()
        if c_driver is None:
            raise AttributeError("C driver library is not loaded. Cannot perform Hebbian learning.")
        # Check if Hebbian learning is meaningful (W_hebb exists and has size)
        if self.W_hebb is None or self.W_hebb.size == 0:
            if DEBUG_PRINTS: print("[BioLayer] Skipping Hebbian learn: W_hebb is None or size 0.")
            return
        # Ensure input handles are valid.
        assert pre_synaptic_activations.handle is not None, "Pre-synaptic handle is None for Hebbian learn."
        assert post_synaptic_activations.handle is not None, "Post-synaptic handle is None for Hebbian learn."
        if self.W_hebb.handle is None: raise RuntimeError("W_hebb handle is None for Hebbian learn.")

        # Call the C/OpenCL kernel for the Hebbian update.
        # C func: (gpu, pre_acts, post_acts, W_hebb_io, lr, B, S, E_pre, E_post)
        # Assuming E_pre = E_post = E_hidden here.
        success = c_driver.execute_hebbian_update_on_gpu(
            self.gpu_index,
            pre_synaptic_activations.handle,
            post_synaptic_activations.handle,
            self.W_hebb.handle, # W_hebb is updated in-place
            self.hebbian_lr,
            self.current_B,
            self.S,
            self.E_hidden, # E_pre
            self.E_hidden  # E_post
        )
        if success == 0:
            raise RuntimeError("C driver execute_hebbian_update_on_gpu failed.")

    def update_prototypes(self):
        """
        Updates the token prototype vectors based on assigned hidden activations.

        Moves each prototype towards the mean of the hidden activation vectors
        that were assigned to it during the last forward pass's dynamic token
        assignment step. Uses an exponential moving average approach controlled
        by `prototype_lr`.

        Chooses between a GPU-accelerated kernel (if available and enabled)
        or a host-based NumPy implementation.
        """
        self._check_freed()
        # Skip if no prototypes are configured.
        if self.T == 0:
            if DEBUG_PRINTS: print("[BioLayer] Skipping prototype update: T=0.")
            return
        if self.prototypes is None:
             print("[BioLayer] WARNING: Skipping prototype update: prototypes tensor is None despite T > 0.")
             return
        if self.token_indices is None:
             print("[BioLayer] WARNING: Skipping prototype update: token_indices tensor is None despite T > 0.")
             return

        # Choose implementation path: GPU or Host.
        use_gpu = USE_GPU_PROTOTYPE_UPDATE and CAN_USE_GPU_PROTO_UPDATE
        if use_gpu:
            try:
                self._update_prototypes_gpu()
                if DEBUG_PRINTS: print("[BioLayer] Prototypes updated via GPU kernel.")
            except Exception as e:
                print(f"[BioLayer] ERROR during GPU prototype update: {e}. Falling back to host update.")
                self._update_prototypes_host() # Attempt host update as fallback
                if DEBUG_PRINTS: print("[BioLayer] Prototypes updated via Host fallback after GPU error.")
        else:
            # Use host implementation if GPU is disabled or unavailable.
            if USE_GPU_PROTOTYPE_UPDATE and not CAN_USE_GPU_PROTO_UPDATE and DEBUG_PRINTS:
                # Remind user if GPU was requested but unavailable.
                print("[BioLayer] INFO: Using Host prototype update (GPU update requested but C functions not found).")
            self._update_prototypes_host()
            if DEBUG_PRINTS: print("[BioLayer] Prototypes updated via Host.")

    def _update_prototypes_gpu(self):
        """GPU-accelerated prototype update using segmented sum and update kernels."""
        if c_driver is None:
             raise AttributeError("C driver library is not loaded. Cannot perform GPU prototype update.")
        # Assert that required GPU buffers exist (should be checked by caller too, but good practice).
        assert self.proto_sums_gpu is not None and self.proto_sums_gpu.handle is not None, "proto_sums_gpu handle is None."
        assert self.proto_counts_gpu is not None and self.proto_counts_gpu.handle is not None, "proto_counts_gpu handle is None."
        assert self.prototypes is not None and self.prototypes.handle is not None, "prototypes handle is None."
        assert self.token_indices is not None and self.token_indices.handle is not None, "token_indices handle is None."
        assert self.hidden_activations.handle is not None, "hidden_activations handle is None."


        # 1. Zero out the sum and count buffers for the current batch.
        self.proto_sums_gpu._zero_initialize()
        self.proto_counts_gpu._zero_initialize()

        # 2. Perform segmented sum: Calculate sum of activations and counts for each prototype index.
        # C func: (gpu, hidden_acts, token_idxs, sums_out, counts_out, M_flat, E_hidden, T)
        success_sum = c_driver.execute_proto_segmented_sum_gpu(
            self.gpu_index,
            self.hidden_activations.handle, # Input activations (M_flat x E_hidden)
            self.token_indices.handle,      # Input assignment indices (M_flat)
            self.proto_sums_gpu.handle,     # Output sums per prototype (T x E_hidden)
            self.proto_counts_gpu.handle,   # Output counts per prototype (T)
            self.current_M_flat,            # Number of input activations/indices
            self.E_hidden,                  # Dimension of activations/prototypes
            self.T                          # Number of prototypes
        )
        if success_sum == 0:
            raise RuntimeError("C driver execute_proto_segmented_sum_gpu failed.")

        # 3. Perform the update step: proto = (1-lr)*proto + lr*(sum/count)
        # C func: (gpu, prototypes_io, proto_sums, proto_counts, lr, E_hidden, T)
        success_update = c_driver.execute_proto_update_step_gpu(
            self.gpu_index,
            self.prototypes.handle,         # Prototypes (updated in-place)
            self.proto_sums_gpu.handle,     # Input sums per prototype
            self.proto_counts_gpu.handle,   # Input counts per prototype
            self.prototype_lr,              # Learning rate
            self.E_hidden,                  # Dimension of prototypes
            self.T                          # Number of prototypes
        )
        if success_update == 0:
            raise RuntimeError("C driver execute_proto_update_step_gpu failed.")

    def _update_prototypes_host(self):
        """Host-based prototype update using NumPy."""
        # Assert required tensors exist (should be checked by caller).
        assert self.token_indices is not None, "token_indices is None for host prototype update."
        assert self.prototypes is not None, "prototypes is None for host prototype update."

        # --- Read data from GPU to Host ---
        # Read assigned indices for the current batch.
        host_indices = np.zeros(self.current_M_flat, dtype=INT_TYPE)
        self.token_indices.read(host_indices, offset_bytes=0) # Only read the relevant part

        # Read hidden activations for the current batch.
        host_activations_flat = np.zeros(self.current_M_flat * self.E_hidden, dtype=FP_TYPE)
        self.hidden_activations.read(host_activations_flat, offset_bytes=0) # Only read the relevant part

        # Read current prototypes.
        current_prototypes_host = np.zeros((self.T, self.E_hidden), dtype=FP_TYPE)
        self.prototypes.read(current_prototypes_host)
        # --- End Read ---

        # Reshape activations for easier indexing (optional but can be clearer).
        # host_activations_rs = host_activations_flat.reshape(self.current_B, self.S, self.E_hidden)
        # host_indices_rs = host_indices.reshape(self.current_B, self.S)
        # Or work with flattened shapes directly:
        host_activations_rs = host_activations_flat.reshape(self.current_M_flat, self.E_hidden)
        host_indices_rs = host_indices # Already flat


        # --- Calculate Updates ---
        updated_prototypes_host = current_prototypes_host.copy() # Start with current values
        update_counts = np.zeros(self.T, dtype=int) # Track how many activations assigned to each proto

        # Iterate through each prototype index.
        for p_idx in range(self.T):
            # Find which activations were assigned to this prototype.
            assigned_mask = (host_indices_rs == p_idx)
            num_assigned = np.sum(assigned_mask)
            update_counts[p_idx] = num_assigned

            if num_assigned > 0:
                # Get the activation vectors assigned to this prototype.
                assigned_activations = host_activations_rs[assigned_mask]
                # Calculate the mean of these activations.
                mean_activation = np.mean(assigned_activations, axis=0)

                # Apply exponential moving average update rule.
                updated_prototypes_host[p_idx] = (
                    (1.0 - self.prototype_lr) * current_prototypes_host[p_idx] +
                    self.prototype_lr * mean_activation
                )
            # else: prototype remains unchanged if no activations were assigned.
        # --- End Calculation ---

        # --- Write Updated Prototypes back to GPU ---
        self.prototypes.write(updated_prototypes_host)

        if DEBUG_PRINTS > 1: # More verbose debug
            print(f"[BioLayer Host Proto Update] Counts per prototype: {update_counts}")

        # --- Cleanup Host Buffers ---
        del host_indices, host_activations_flat, host_indices_rs, host_activations_rs
        del current_prototypes_host, updated_prototypes_host, update_counts

    # --- State Management and Information Retrieval ---

    def get_state(self) -> Dict[str, np.ndarray]:
        """
        Retrieves the layer's state (W1, b1, Adam moments, W_hebb, prototypes) as NumPy arrays.

        Returns:
            Dict[str, np.ndarray]: Dictionary containing the state arrays.
        """
        self._check_freed()
        state: Dict[str, np.ndarray] = {}

        # Get state from the linear part (W1, b1, Adam states)
        # Read W1 and b1
        state['W1'] = np.zeros((self.E_in, self.E_hidden), dtype=FP_TYPE)
        state['b1'] = np.zeros(self.E_hidden, dtype=FP_TYPE)
        if self.W1.size > 0: self.W1.read(state['W1'])
        if self.b1.size > 0: self.b1.read(state['b1'])
        # Read Adam states for W1
        adam_shape_W1 = (self.E_in, self.E_hidden)
        state['m_W1'] = np.zeros(adam_shape_W1, dtype=np.float32)
        state['v_W1'] = np.zeros(adam_shape_W1, dtype=np.float32)
        if self.m_W1.size > 0: self.m_W1.read(state['m_W1'])
        if self.v_W1.size > 0: self.v_W1.read(state['v_W1'])
        # Read Adam states for b1
        adam_shape_b1 = (self.E_hidden,)
        state['m_b1'] = np.zeros(adam_shape_b1, dtype=np.float32)
        state['v_b1'] = np.zeros(adam_shape_b1, dtype=np.float32)
        if self.m_b1.size > 0: self.m_b1.read(state['m_b1'])
        if self.v_b1.size > 0: self.v_b1.read(state['v_b1'])

        # Get Hebbian weights
        state['W_hebb'] = self.get_hebbian_weights()

        # Get Prototypes (if they exist)
        if self.T > 0:
            state['prototypes'] = self.get_prototypes()

        return state

    def set_state(self, state: Dict[str, np.ndarray]):
        """
        Loads the layer's state from NumPy arrays.

        Args:
            state: Dictionary containing state arrays.

        Raises:
            KeyError: If a required key is missing.
            Exception: Catches potential errors during GPU writes.
        """
        self._check_freed()
        if DEBUG_PRINTS: print(f"[BioLayer] Setting state from checkpoint...")
        try:
            # Set state for the linear part (W1, b1, Adam states)
            if 'W1' in state and self.W1.size > 0: self.W1.write(state['W1'].astype(FP_TYPE))
            if 'b1' in state and self.b1.size > 0: self.b1.write(state['b1'].astype(FP_TYPE))
            if 'm_W1' in state and self.m_W1.size > 0: self.m_W1.write(state['m_W1'].astype(np.float32))
            if 'v_W1' in state and self.v_W1.size > 0: self.v_W1.write(state['v_W1'].astype(np.float32))
            if 'm_b1' in state and self.m_b1.size > 0: self.m_b1.write(state['m_b1'].astype(np.float32))
            if 'v_b1' in state and self.v_b1.size > 0: self.v_b1.write(state['v_b1'].astype(np.float32))

            # Set Hebbian weights
            if 'W_hebb' in state and self.W_hebb is not None and self.W_hebb.size > 0:
                self.W_hebb.write(state['W_hebb'].astype(FP_TYPE))
            elif 'W_hebb' in state and (self.W_hebb is None or self.W_hebb.size == 0):
                 print("[BioLayer] WARNING: 'W_hebb' found in checkpoint but layer has no W_hebb allocated. Skipping.")

            # Set Prototypes (if they exist in the layer and checkpoint)
            if 'prototypes' in state and self.T > 0 and self.prototypes is not None:
                # Check shape consistency before writing
                expected_shape = (self.T, self.E_hidden)
                loaded_shape = state['prototypes'].shape
                if loaded_shape == expected_shape:
                    self.prototypes.write(state['prototypes'].astype(FP_TYPE))
                else:
                    print(f"[BioLayer] WARNING: Checkpoint 'prototypes' shape {loaded_shape} mismatch with expected {expected_shape}. Skipping loading prototypes.")
            elif 'prototypes' in state and (self.T == 0 or self.prototypes is None):
                print("[BioLayer] WARNING: 'prototypes' found in checkpoint but layer has T=0 or prototypes=None. Skipping.")

            if DEBUG_PRINTS: print(f"[BioLayer] State loaded successfully.")
        except KeyError as e:
            print(f"[BioLayer] WARNING: Missing key '{e}' in checkpoint state dictionary while loading BioLayer.")
        except Exception as e:
            print(f"[BioLayer] ERROR setting state: {e}")

    def get_dynamic_tokens(self) -> Optional[np.ndarray]:
        """
        Retrieves the assigned token prototype indices for the last processed batch.

        Returns:
            Optional[np.ndarray]: A NumPy array of shape (current_B, S) containing
            the integer index of the assigned prototype for each position, or
            None if prototypes are disabled (T=0) or indices haven't been computed.
            Returns raw flattened array if B or S is unavailable.
        """
        self._check_freed()
        if self.token_indices is None or self.T == 0:
            return None

        host_indices = np.zeros(self.current_M_flat, dtype=INT_TYPE)
        try:
            # Read only the indices relevant to the current batch size.
            self.token_indices.read(host_indices, offset_bytes=0)
            # Try to reshape, fall back to flat array if dimensions unknown/inconsistent.
            try:
                return host_indices.reshape(self.current_B, self.S)
            except ValueError:
                 print(f"[BioLayer] Warning: Could not reshape token indices ({self.current_M_flat}) to ({self.current_B}, {self.S}). Returning flat array.")
                 return host_indices
        except Exception as e:
            print(f"[BioLayer] Error reading token indices from GPU: {e}")
            return None # Return None or empty array on error?


    def get_spikes(self) -> Optional[np.ndarray]:
        """
        Retrieves the computed spike values for the last processed batch.

        Returns:
            Optional[np.ndarray]: A NumPy array of shape (current_B, S, E_hidden)
            containing the spike values (typically 0.0 or 1.0), or None on error.
            Returns raw flattened array if B or S is unavailable.
        """
        self._check_freed()
        if self.spikes is None: return None

        host_spikes_flat = np.zeros(self.current_M_flat * self.E_hidden, dtype=FP_TYPE)
        try:
            self.spikes.read(host_spikes_flat, offset_bytes=0) # Read relevant part
            # Try to reshape.
            try:
                 return host_spikes_flat.reshape(self.current_B, self.S, self.E_hidden)
            except ValueError:
                 print(f"[BioLayer] Warning: Could not reshape spikes ({len(host_spikes_flat)}) to ({self.current_B}, {self.S}, {self.E_hidden}). Returning flat array.")
                 return host_spikes_flat
        except Exception as e:
            print(f"[BioLayer] Error reading spikes from GPU: {e}")
            return None

    def get_prototypes(self) -> Optional[np.ndarray]:
        """
        Retrieves the current token prototype vectors.

        Returns:
            Optional[np.ndarray]: A NumPy array of shape (T, E_hidden) containing
            the prototype vectors, or None if prototypes are disabled (T=0) or
            error occurs.
        """
        self._check_freed()
        if self.prototypes is None or self.T == 0:
            return None

        host_prototypes = np.zeros((self.T, self.E_hidden), dtype=FP_TYPE)
        try:
            self.prototypes.read(host_prototypes)
            return host_prototypes
        except Exception as e:
            print(f"[BioLayer] Error reading prototypes from GPU: {e}")
            return None

    def get_hebbian_weights(self) -> Optional[np.ndarray]:
        """
        Retrieves the current Hebbian weight matrix (W_hebb).

        Returns:
            Optional[np.ndarray]: A NumPy array of shape (E_hidden, E_hidden) containing
            the Hebbian weights, or None if W_hebb is None or error occurs.
        """
        self._check_freed()
        if self.W_hebb is None or self.W_hebb.size == 0:
            return np.zeros((self.E_hidden, self.E_hidden), dtype=FP_TYPE) # Return zeros if not used

        host_W_hebb = np.zeros((self.E_hidden, self.E_hidden), dtype=FP_TYPE)
        try:
            if self.W_hebb.size > 0: # Check size again just in case
                self.W_hebb.read(host_W_hebb)
            return host_W_hebb
        except Exception as e:
            print(f"[BioLayer] Error reading Hebbian weights from GPU: {e}")
            return None

    def get_W1_grad(self) -> Optional[np.ndarray]:
        """Retrieves the current gradients for W1."""
        self._check_freed();
        if self.dW1 is None or self.dW1.size == 0: return None
        grad = np.zeros((self.E_in, self.E_hidden), dtype=FP_TYPE);
        try:
            self.dW1.read(grad); return grad
        except Exception as e: print(f"[BioLayer] Error reading W1 gradient: {e}"); return None

    def get_b1_grad(self) -> Optional[np.ndarray]:
        """Retrieves the current gradients for b1."""
        self._check_freed();
        if self.db1 is None or self.db1.size == 0: return None
        grad = np.zeros(self.E_hidden, dtype=FP_TYPE);
        try:
            self.db1.read(grad); return grad
        except Exception as e: print(f"[BioLayer] Error reading b1 gradient: {e}"); return None


    # --- Resource Management ---

    def free(self):
        """Releases all GPU resources allocated by this layer."""
        if not self._is_freed:
            if DEBUG_PRINTS: print(f"[BioLayer] Freeing GPU resources...")
            tensors_to_free = [getattr(self, attr) for attr in vars(self) if isinstance(getattr(self, attr, None), GPUTensor)]
            for tensor in tensors_to_free:
                if tensor:
                    try:
                        tensor.free()
                    except Exception as e:
                        print(f"[BioLayer] Warning: Error freeing tensor {getattr(tensor,'name','Unknown')} during layer cleanup: {e}")
            self._is_freed = True
            if DEBUG_PRINTS: print(f"[BioLayer] Resources freed.")

    def __del__(self):
        """Ensures resources are freed when the object is garbage collected."""
        if not getattr(self, '_is_freed', True):
            self.free()

    def train(self):
        """Sets the layer to training mode (no specific effect currently)."""
        pass

    def eval(self):
        """Sets the layer to evaluation mode (no specific effect currently)."""
        pass


# --- Cross Entropy Loss ---

class CrossEntropyLoss:
    """
    Computes the Cross-Entropy loss between logits and target labels.

    Uses a C/OpenCL kernel that combines LogSoftmax and Negative Log Likelihood (NLL)
    loss calculation for numerical stability and efficiency. The same kernel
    also computes the gradient of the loss with respect to the input logits.

    Handles padding by ignoring target labels equal to `PAD_INDEX` when
    calculating the average loss on the host side.

    Attributes:
        max_M_flat (int): Maximum flattened dimension (Batch Size * Sequence Length)
                          used for buffer allocation.
        V (int): Vocabulary size.
        gpu_index (int): GPU device index.
        log_probs (GPUTensor): GPU buffer to store intermediate log probabilities
                               (max_M_flat x V).
        d_logits (GPUTensor): GPU buffer where the gradient w.r.t. input logits
                              is stored by the C kernel (max_M_flat x V).
        loss_per_sample (GPUTensor): GPU buffer to store the loss for each
                                     individual sample in the batch (max_M_flat).
    """
    def __init__(self, max_batch_size_seq_len_flat: int, vocab_size: int, gpu_index: int = GPU_INDEX):
        """
        Initializes the CrossEntropyLoss component.

        Args:
            max_batch_size_seq_len_flat: The maximum expected flattened dimension
                                         (max_batch_size * seq_len).
            vocab_size: The size of the vocabulary (number of output classes).
            gpu_index: The index of the GPU device to use.
        """
        self.max_M_flat: int = max_batch_size_seq_len_flat
        self.V: int = vocab_size
        self.gpu_index: int = gpu_index
        self._is_freed: bool = False
        itemsize_fp = FP_TYPE().itemsize

        # Allocate GPU buffers based on max size.
        self.log_probs = GPUTensor(self.max_M_flat * self.V * itemsize_fp, gpu_index, name="LogProbs")
        # Gradient buffer MUST be zero-initialized before each backward pass if the kernel accumulates.
        # However, the combined kernel calculates it directly, so zero-init here is for safety.
        self.d_logits = GPUTensor(self.max_M_flat * self.V * itemsize_fp, gpu_index, name="dLogits", zero_init=True)
        self.loss_per_sample = GPUTensor(self.max_M_flat * itemsize_fp, gpu_index, name="LossPerSample")

        if DEBUG_PRINTS: print(f"[CrossEntropyLoss] Initialized: VocabSize={self.V}, Max M_flat={self.max_M_flat} on GPU {self.gpu_index}")

    def _check_freed(self):
        """Raises RuntimeError if operations are attempted on a freed component."""
        if self._is_freed:
            raise RuntimeError("Operation attempted on a freed CrossEntropyLoss.")

    def forward(self, logits: GPUTensor, targets_gpu: GPUTensor, current_m_flat: int, targets_np: np.ndarray) -> float:
        """
        Computes the cross-entropy loss and the gradient w.r.t. logits.

        1. Calls a C/OpenCL kernel (`execute_cross_entropy_loss_grad_gpu`) that
           internally computes LogSoftmax(logits) and then calculates the NLLLoss
           based on `targets_gpu`.
        2. The same kernel directly computes the gradient `d_logits = Softmax(logits) - one_hot(targets)`
           and stores it in `self.d_logits`.
        3. It also stores the per-sample loss in `self.loss_per_sample`.
        4. Reads `loss_per_sample` back to the host.
        5. Calculates the average loss on the host, ignoring padded samples
           identified using `targets_np`.

        Args:
            logits: GPUTensor containing the raw output scores from the model
                    (shape current_m_flat x V).
            targets_gpu: GPUTensor containing the ground truth target labels (integer IDs)
                         (shape current_m_flat).
            current_m_flat: The actual flattened dimension (B * S) for the current batch.
            targets_np: The NumPy array containing the host-side target labels, used
                       to identify padding (`PAD_INDEX`) for correct averaging.
                       Expected shape (B, S) or flattened (B*S).

        Returns:
            float: The average cross-entropy loss for the batch, excluding padded elements.

        Raises:
            RuntimeError: If C driver execution fails.
            ValueError: If input handles are None or `current_m_flat` exceeds allocation.
            AttributeError: If `c_driver` is None.
        """
        self._check_freed()
        if c_driver is None:
            raise AttributeError("C driver library is not loaded. Cannot compute CrossEntropyLoss.")

        # --- Validation ---
        if logits.handle is None or targets_gpu.handle is None or self.d_logits.handle is None or self.loss_per_sample.handle is None:
            raise ValueError("Input/Output handles for CrossEntropyLoss forward/backward are None.")
        if current_m_flat > self.max_M_flat:
            raise ValueError(f"Current M_flat ({current_m_flat}) exceeds allocated max M_flat ({self.max_M_flat}) for CrossEntropyLoss.")
        if current_m_flat < 0:
             raise ValueError(f"Current M_flat ({current_m_flat}) cannot be negative.")
        # It might be beneficial to check target_np shape consistency here too.
        # --- End Validation ---

        # The combined kernel handles LogSoftmax internally, but if separated:
        # success_log_softmax = c_driver.execute_log_softmax_stable_gpu(
        #     self.gpu_index, logits.handle, self.log_probs.handle, current_m_flat, self.V
        # )
        # if success_log_softmax == 0:
        #      print("[FATAL] C driver execute_log_softmax_stable_gpu returned 0. Check C error messages.")
        #      raise RuntimeError("CrossEntropyLoss forward LogSoftmax step failed.")

        # Call the combined C/OpenCL kernel to calculate loss and gradient.
        # Assumes kernel takes logits, targets, outputs d_logits, outputs loss_per_sample.
        # C func: (gpu, logits_in, targets_in, d_logits_out, loss_per_sample_out, M_flat, V)
        success_ce_grad = c_driver.execute_cross_entropy_loss_grad_gpu(
            self.gpu_index,
            logits.handle,              # Input logits (M x V)
            targets_gpu.handle,         # Input target indices (M)
            self.d_logits.handle,       # Output gradient w.r.t logits (M x V)
            self.loss_per_sample.handle,# Output per-sample loss (M)
            current_m_flat,             # M dimension
            self.V                      # V dimension (Vocab size)
        )
        if success_ce_grad == 0:
            raise RuntimeError("C driver execute_cross_entropy_loss_grad_gpu failed. Check C driver output.")

        # --- Calculate Average Loss on Host (Ignoring Padding) ---
        # Allocate host buffer for per-sample losses.
        host_loss_per_sample = np.zeros(current_m_flat, dtype=FP_TYPE)
        # Read the per-sample losses from GPU.
        self.loss_per_sample.read(host_loss_per_sample, offset_bytes=0)

        mean_loss: float = 0.0
        # Flatten the host targets array to match the flat loss array.
        # Ensure we only take the elements corresponding to the current batch size.
        targets_np_flat = targets_np.flatten()[:current_m_flat]
        # Create a boolean mask for valid (non-padded) target entries.
        valid_mask = (targets_np_flat != PAD_INDEX)

        # Check if there are any valid samples in the batch.
        if np.any(valid_mask):
            # Select the losses corresponding to valid targets.
            valid_losses = host_loss_per_sample[valid_mask]
            # Calculate the mean of valid losses. Check for empty valid_losses (shouldn't happen if np.any(valid_mask) is true).
            if len(valid_losses) > 0:
                 mean_loss = np.mean(valid_losses)
                 # Add check for NaN/Inf loss read from GPU
                 if not math.isfinite(mean_loss):
                     print(f"[CrossEntropyLoss] WARNING: Calculated mean loss is NaN or Inf ({mean_loss}). Individual losses: {valid_losses}")
                     # Optionally return a high value or re-raise an error
                     # mean_loss = float('inf') # Or handle as needed
            # else: mean_loss remains 0.0

        # --- Cleanup Host Buffers ---
        del host_loss_per_sample, targets_np_flat, valid_mask
        # `valid_losses` is a view, no need to delete explicitly unless large

        # Return the computed average loss. The gradient (d_logits) is now stored
        # in self.d_logits GPUTensor, ready for the backward pass of the preceding layer.
        return float(mean_loss)

    def free(self):
        """Releases all GPU resources allocated by this component."""
        if not self._is_freed:
            if DEBUG_PRINTS: print(f"[CrossEntropyLoss] Freeing GPU resources...")
            tensors_to_free = [self.log_probs, self.d_logits, self.loss_per_sample]
            for tensor in tensors_to_free:
                if tensor:
                    try:
                        tensor.free()
                    except Exception as e:
                        print(f"[CrossEntropyLoss] Warning: Error freeing tensor {getattr(tensor,'name','Unknown')} during cleanup: {e}")
            self._is_freed = True
            if DEBUG_PRINTS: print(f"[CrossEntropyLoss] Resources freed.")


    def __del__(self):
        """Ensures resources are freed when the object is garbage collected."""
        if not getattr(self, '_is_freed', True):
            self.free()


# --- Main Model Class ---

class MyModel:
    """
    Combines the different layers (Embedding, BioInspired, Linear) into a single model.

    Manages the overall forward pass, backward pass initiation, parameter updates,
    checkpointing, and resource management for all contained layers.

    Architecture:
        Input (Token IDs) -> EmbeddingLayer -> BioInspiredAssociativeLayer -> LinearLayer -> Output (Logits)

    Attributes:
        B (int): Maximum batch size.
        S (int): Sequence length.
        M_flat (int): Maximum flattened dimension (B * S).
        embedding_dim (int): Dimension of embeddings.
        hidden_dim (int): Dimension of the BioLayer hidden state.
        vocab_size (int): Size of the vocabulary.
        num_prototypes (int): Number of prototypes used in the BioLayer.
        gpu_index (int): GPU device index.
        embedding_layer (EmbeddingLayer): The embedding layer instance.
        bio_layer (BioInspiredAssociativeLayer): The bio-inspired layer instance.
        output_layer (LinearLayer): The final linear output layer instance.
        criterion (CrossEntropyLoss): The loss calculation component.
        embedding_output_buffer (GPUTensor): Intermediate buffer holding the output
                                            of the embedding layer.
    """
    def __init__(self, batch_size: int, seq_len: int, embedding_dim: int, hidden_dim: int, vocab_size: int, num_prototypes: int, gpu_index: int = GPU_INDEX):
        """
        Initializes the MyModel instance and its constituent layers.

        Args:
            batch_size: Maximum expected batch size.
            seq_len: Sequence length.
            embedding_dim: Dimension for the embedding layer.
            hidden_dim: Dimension for the bio-inspired hidden layer.
            vocab_size: Size of the vocabulary (for embedding and output layers).
            num_prototypes: Number of prototypes for the bio-inspired layer.
            gpu_index: The index of the GPU device to use.
        """
        self.B: int = batch_size
        self.S: int = seq_len
        self.M_flat: int = batch_size * seq_len # Max flattened size
        self.embedding_dim: int = embedding_dim
        self.hidden_dim: int = hidden_dim
        self.vocab_size: int = vocab_size
        self.num_prototypes: int = num_prototypes
        self.gpu_index: int = gpu_index
        self._is_freed: bool = False
        itemsize_fp = FP_TYPE().itemsize

        print("[Model Init] Instantiating layers...")
        # Instantiate layers, passing necessary dimensions and GPU index.
        self.embedding_layer = EmbeddingLayer(self.vocab_size, self.embedding_dim, gpu_index=self.gpu_index)
        self.bio_layer = BioInspiredAssociativeLayer(
            batch_size=self.B, seq_len=self.S, embedding_dim=self.embedding_dim, hidden_dim=self.hidden_dim,
            num_prototypes=self.num_prototypes, hebbian_lr=HEBBIAN_LR, spike_threshold=SPIKE_THRESHOLD,
            prototype_lr=PROTOTYPE_LR, gpu_index=self.gpu_index
        )
        self.output_layer = LinearLayer(self.M_flat, self.hidden_dim, self.vocab_size, gpu_index=self.gpu_index)
        self.criterion = CrossEntropyLoss(self.M_flat, self.vocab_size, gpu_index=self.gpu_index)
        print("[Model Init] Layers instantiated.")

        print("[Model Init] Allocating intermediate buffers...")
        # Allocate the buffer to hold the output of the embedding layer, which is
        # the input to the bio_layer. Size is max_M_flat * embedding_dim.
        self.embedding_output_buffer = GPUTensor(self.M_flat * self.embedding_dim * itemsize_fp, gpu_index, name="EmbeddingOutputBuffer")
        print("[Model Init] Model initialization complete.")

    def _initialize_weights(self):
        """Initializes weights for all layers using their respective methods."""
        if DEBUG_PRINTS: print("[Model] Initializing weights for all layers...")
        # Initialize Embedding Layer weights
        self.embedding_layer.initialize_weights()

        # Initialize Bio Layer weights (W1) using Xavier/Glorot uniform. Prototypes random normal.
        limit_bio = np.sqrt(6.0 / (self.bio_layer.E_in + self.bio_layer.E_hidden))
        bio_w1_init = np.random.uniform(-limit_bio, limit_bio, (self.bio_layer.E_in, self.bio_layer.E_hidden)).astype(FP_TYPE)
        self.bio_layer.W1.write(bio_w1_init)
        del bio_w1_init
        print(f"[BioLayer] W1 weights initialized with U[-{limit_bio:.4f}, {limit_bio:.4f}].")
        # Initialize prototypes only if they exist (T > 0). Use small random normal values.
        if self.bio_layer.T > 0 and self.bio_layer.prototypes is not None:
            prototypes_init = np.random.randn(self.bio_layer.T, self.bio_layer.E_hidden).astype(FP_TYPE) * 0.1 # Small variance
            self.bio_layer.prototypes.write(prototypes_init)
            print(f"[BioLayer] Prototypes initialized with N(0, 0.1^2).")
            del prototypes_init

        # Initialize Output Layer weights (W) using Xavier/Glorot uniform.
        limit_out = np.sqrt(6.0 / (self.output_layer.E_in + self.output_layer.E_out))
        out_w_init = np.random.uniform(-limit_out, limit_out, (self.output_layer.E_in, self.output_layer.E_out)).astype(FP_TYPE)
        self.output_layer.W.write(out_w_init)
        del out_w_init
        print(f"[LinearLayer Out] Weights initialized with U[-{limit_out:.4f}, {limit_out:.4f}].")
        # Biases are already zero-initialized by GPUTensor.

    def set_current_batch_shape(self, b: int, s: int):
        """
        Informs all layers about the actual shape of the current batch.

        Args:
            b: Current batch size.
            s: Current sequence length.

        Raises:
            ValueError: If sequence length `s` does not match model config `self.S`.
        """
        if s != self.S:
            raise ValueError(f"Input sequence length ({s}) does not match model's configured sequence length ({self.S}).")
        # Calculate the current flattened dimension.
        current_m_flat = b * s
        # Propagate the current shape information to each layer.
        self.embedding_layer.set_current_batch_shape(b, s)
        self.bio_layer.set_current_batch_shape(b, s)
        self.output_layer.set_current_batch_m_flat(current_m_flat)
        # Note: CrossEntropyLoss uses current_m_flat passed directly to its forward method.

    def forward(self, input_indices_gpu: GPUTensor) -> GPUTensor:
        """
        Executes the forward pass through the entire model architecture.

        Input (IDs) -> Embedding -> BioLayer -> OutputLayer -> Logits

        Args:
            input_indices_gpu: GPUTensor containing the batch of input token IDs
                               (shape current_B * current_S).

        Returns:
            GPUTensor: GPUTensor containing the output logits
                       (shape current_M_flat x vocab_size).
        """
        # Ensure layers know the current batch size before executing.
        # Assumes set_current_batch_shape was called externally before this.

        # 1. Embedding Layer
        # Input: input_indices_gpu, Output: embedding_output_buffer
        embedded_input = self.embedding_layer.forward(input_indices_gpu, self.embedding_output_buffer)

        # 2. Bio-Inspired Layer
        # Input: embedding_output_buffer, Output: bio_layer.hidden_activations
        bio_output = self.bio_layer.forward(embedded_input)

        # 3. Output Linear Layer
        # Input: bio_layer.hidden_activations, Output: output_layer.output (Logits)
        logits = self.output_layer.forward(bio_output)

        return logits

    def compute_loss_and_backward(self, logits: GPUTensor, targets_gpu: GPUTensor, targets_np: np.ndarray) -> Tuple[float, None]:
        """
        Computes the loss and performs the backward pass through all layers.

        1. Computes loss using the criterion (which also calculates d_logits).
        2. Propagates gradients backward: OutputLayer -> BioLayer -> EmbeddingLayer.

        Args:
            logits: GPUTensor containing the output logits from the forward pass.
            targets_gpu: GPUTensor containing the target token IDs on the GPU.
            targets_np: NumPy array of target token IDs on the host (for padding mask).

        Returns:
            Tuple[float, None]: A tuple containing:
                - float: The calculated average loss for the batch (excluding padding).
                - None: Indicates gradients are stored internally, not returned.
        """
        # Get the actual flattened dimension for the current batch from a layer.
        current_m_flat = self.bio_layer.current_M_flat # Assumes bio_layer holds the current M_flat

        # 1. Compute Loss and d_logits (gradient w.r.t. logits)
        # The criterion.forward method calculates both loss and stores d_logits internally.
        loss = self.criterion.forward(logits, targets_gpu, current_m_flat, targets_np=targets_np)
        # Retrieve the gradient computed by the loss function.
        d_logits = self.criterion.d_logits # Shape: current_M_flat x vocab_size

        # 2. Backward pass through Output Layer
        # Input: d_logits, Output: d_bio_output (gradient w.r.t. bio_layer output)
        d_bio_output = self.output_layer.backward(d_logits)

        # 3. Backward pass through Bio-Inspired Layer
        # Input: d_bio_output, Output: d_embedded_input (gradient w.r.t. embedding output)
        d_embedded_input = self.bio_layer.backward(d_bio_output)

        # 4. Backward pass through Embedding Layer
        # Input: d_embedded_input, Output: None (accumulates dW_emb internally)
        self.embedding_layer.backward(d_embedded_input)

        # Gradients (dW, db) are now stored within each layer's respective tensors.
        return loss, None

    def clip_all_gradients(self, clip_value: Optional[float]):
         """
         Applies gradient clipping to all trainable layers in the model.

         Args:
             clip_value: The maximum L2 norm for gradients. If None or <= 0,
                         clipping is skipped.
         """
         if clip_value is not None and clip_value > 0:
             if DEBUG_PRINTS: print(f"[Model] Clipping gradients with L2 norm <= {clip_value}...")
             self.embedding_layer.clip_gradients(clip_value)
             self.bio_layer.clip_gradients(clip_value)
             self.output_layer.clip_gradients(clip_value)

    def update_trainable(self, t: int, lr: float, beta1: float = 0.9, beta2: float = 0.999, weight_decay: float = 0.0):
        """
        Updates the parameters of all trainable layers using their Adam optimizers.

        Args:
            t: Current global training step (for Adam bias correction).
            lr: Current learning rate.
            beta1: Adam beta1 parameter.
            beta2: Adam beta2 parameter.
            weight_decay: L2 regularization factor (applied selectively in layers).
        """
        if DEBUG_PRINTS: print(f"[Model] Updating trainable parameters (Adam step {t}, lr={lr:.6g}, wd={weight_decay:.4g})...")
        # Update Embedding Layer (typically no weight decay)
        self.embedding_layer.update(t, lr, beta1, beta2, weight_decay=0.0)
        # Update Bio-Inspired Layer (applies weight decay to W1)
        self.bio_layer.update(t, lr, beta1, beta2, weight_decay=weight_decay)
        # Update Output Layer (applies weight decay to W)
        self.output_layer.update(t, lr, beta1, beta2, weight_decay=weight_decay)

    def update_special(self):
        """
        Performs special update rules unique to certain layers.

        Currently calls Hebbian learning and prototype updates in the BioLayer.
        """
        if DEBUG_PRINTS: print("[Model] Performing special updates (Hebbian, Prototypes)...")
        # Hebbian learning using hidden activations and spikes from the last forward pass.
        self.bio_layer.hebbian_learn(self.bio_layer.hidden_activations, self.bio_layer.spikes)
        # Update prototypes based on assignments from the last forward pass.
        self.bio_layer.update_prototypes()

    def save_checkpoint(self, filepath: str, epoch: int, global_step: int, best_loss: Optional[float] = None):
        """
        Saves the current state of the model and training progress to a file.

        Includes layer states (weights, Adam moments), epoch number, global step,
        best validation loss achieved so far, and key hyperparameters. Creates a
        backup (.bak) of the previous checkpoint before saving.

        Args:
            filepath: The path to save the checkpoint file (usually ends in .pkl).
            epoch: The current epoch number (completed epochs + 1).
            global_step: The current global training step count.
            best_loss: The best validation loss recorded so far (optional).
        """
        print(f"[Model] Saving checkpoint to '{filepath}' (Epoch {epoch}, Step {global_step})...")
        # Gather state from all components.
        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'embedding_layer_state': self.embedding_layer.get_state(),
            'bio_layer_state': self.bio_layer.get_state(),
            'output_layer_state': self.output_layer.get_state(),
            'best_valid_loss': best_loss if best_loss is not None else float('inf'),
            # Store key hyperparameters for consistency checks during loading.
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'num_prototypes': self.num_prototypes,
            'seq_len': self.S
        }

        try:
            # Create backup of existing checkpoint file.
            backup_path = filepath + ".bak"
            if os.path.exists(filepath):
                if os.path.exists(backup_path):
                    os.remove(backup_path) # Remove old backup
                os.rename(filepath, backup_path) # Rename current to backup

            # Save the new checkpoint using pickle.
            with open(filepath, 'wb') as f:
                pickle.dump(checkpoint, f)

            if DEBUG_PRINTS: print(f"[Model] Checkpoint saved successfully: {filepath}")

        except Exception as e:
            print(f"[Model] ERROR saving checkpoint to {filepath}: {e}")
            # Optionally try to restore backup if save failed mid-way? (More complex)

    def load_checkpoint(self, filepath: str) -> Tuple[int, int, Optional[float]]:
        """
        Loads the model state and training progress from a checkpoint file.

        Performs consistency checks to ensure loaded hyperparameters match the
        current model configuration. If mismatch or file not found, initializes
        weights randomly.

        Args:
            filepath: The path to the checkpoint file (.pkl).

        Returns:
            Tuple[int, int, Optional[float]]: A tuple containing:
                - The epoch number to resume training from.
                - The global step count to resume from.
                - The best validation loss recorded in the checkpoint.
                Returns (0, 0, inf) if checkpoint not found or invalid.
        """
        if not os.path.exists(filepath):
            print(f"[Model] Checkpoint file not found: '{filepath}'. Initializing model with random weights.")
            self._initialize_weights()
            return 0, 0, float('inf') # Start from scratch

        try:
            print(f"[Model] Loading checkpoint from '{filepath}'...")
            with open(filepath, 'rb') as f:
                checkpoint = pickle.load(f)

            # --- Hyperparameter Consistency Check ---
            mismatch = False
            loaded_vs_current = [
                ('vocab_size', checkpoint.get('vocab_size'), self.vocab_size),
                ('embedding_dim', checkpoint.get('embedding_dim'), self.embedding_dim),
                ('hidden_dim', checkpoint.get('hidden_dim'), self.hidden_dim),
                # Check optional params only if present in checkpoint (for backward compat)
                ('num_prototypes', checkpoint.get('num_prototypes', self.num_prototypes), self.num_prototypes) if 'num_prototypes' in checkpoint else None,
                ('seq_len', checkpoint.get('seq_len', self.S), self.S) if 'seq_len' in checkpoint else None,
            ]
            for name, loaded_val, current_val in loaded_vs_current:
                 if name is None: continue # Skip None entries from conditional checks
                 # Allow loading if checkpoint value is missing (use current), but warn if different.
                 if loaded_val is not None and loaded_val != current_val:
                     print(f"[Model] WARNING: Checkpoint hyperparameter mismatch! '{name}': "
                           f"Checkpoint has {loaded_val}, Model configured with {current_val}.")
                     mismatch = True

            if mismatch:
                # If critical hyperparameters don't match, refuse to load parameters
                # and re-initialize to prevent unexpected behavior or errors.
                print("[Model] CRITICAL WARNING: Hyperparameter mismatch detected. "
                      "Loading parameters from this checkpoint is unsafe. "
                      "Re-initializing model weights.")
                self._initialize_weights()
                return 0, 0, float('inf') # Start from scratch
            # --- End Check ---

            # Load state into each layer.
            self.embedding_layer.set_state(checkpoint['embedding_layer_state'])
            self.bio_layer.set_state(checkpoint['bio_layer_state'])
            self.output_layer.set_state(checkpoint['output_layer_state'])

            # Retrieve training progress. Use defaults if keys are missing.
            epoch = checkpoint.get('epoch', 0)
            global_step = checkpoint.get('global_step', 0)
            best_loss = checkpoint.get('best_valid_loss', float('inf'))

            print(f"[Model] Checkpoint loaded successfully. Resuming from Epoch {epoch}, Step {global_step}. Best valid loss so far: {best_loss:.6f}")
            return epoch, global_step, best_loss

        except FileNotFoundError: # Should be caught above, but for safety
             print(f"[Model] Checkpoint file disappeared unexpectedly: '{filepath}'. Initializing model.")
             self._initialize_weights(); return 0, 0, float('inf')
        except Exception as e:
            # Catch other potential errors (pickle errors, invalid file format, errors during set_state).
            print(f"[Model] ERROR loading checkpoint from '{filepath}': {e}. Re-initializing model weights.")
            import traceback
            traceback.print_exc() # Print stack trace for debugging
            self._initialize_weights()
            return 0, 0, float('inf') # Start from scratch


    def free(self):
        """Releases GPU resources for the model and all its components."""
        if not self._is_freed:
            if DEBUG_PRINTS: print("[Model] Freeing model resources...")
            # Free resources in reverse order of creation (optional, but sometimes helpful)
            if hasattr(self, 'criterion') and self.criterion: self.criterion.free()
            if hasattr(self, 'output_layer') and self.output_layer: self.output_layer.free()
            if hasattr(self, 'bio_layer') and self.bio_layer: self.bio_layer.free()
            if hasattr(self, 'embedding_layer') and self.embedding_layer: self.embedding_layer.free()
            # Free intermediate buffers managed by the model itself
            if hasattr(self, 'embedding_output_buffer') and self.embedding_output_buffer: self.embedding_output_buffer.free()

            self._is_freed = True
            if DEBUG_PRINTS: print("[Model] Model resources freed.")

    def __del__(self):
         """Ensures resources are freed when the object is garbage collected."""
         if not getattr(self, '_is_freed', True):
             self.free()

    def train_mode(self):
         """Sets all layers to training mode."""
         if DEBUG_PRINTS: print("[Model] Setting train mode.")
         self.embedding_layer.train()
         self.bio_layer.train()
         self.output_layer.train()
         # Criterion typically doesn't have modes

    def eval_mode(self):
         """Sets all layers to evaluation mode."""
         if DEBUG_PRINTS: print("[Model] Setting eval mode.")
         self.embedding_layer.eval()
         self.bio_layer.eval()
         self.output_layer.eval()


# --- Learning Rate Scheduler ---

class StepLR:
     """
     Decays the learning rate by a multiplicative factor (`gamma`) every
     `step_size` epochs.

     Attributes:
         lr (float): The current learning rate.
         initial_lr (float): The starting learning rate.
         step_size (int): The period (in epochs) for decaying the learning rate.
         gamma (float): The multiplicative decay factor.
         last_epoch (int): The epoch number of the last update.
     """
     def __init__(self, initial_lr: float, step_size: int, gamma: float = 0.1):
          """
          Initializes the StepLR scheduler.

          Args:
              initial_lr: The initial learning rate.
              step_size: Frequency of decay (number of epochs).
              gamma: Multiplicative factor of learning rate decay.
          """
          if step_size <= 0 or not isinstance(step_size, int):
              raise ValueError("Step size must be a positive integer.")
          if gamma >= 1.0:
              print(f"Warning: StepLR gamma factor ({gamma}) is >= 1.0, learning rate will not decrease.")

          self.initial_lr: float = initial_lr
          self.lr: float = initial_lr
          self.step_size: int = step_size
          self.gamma: float = gamma
          self.last_epoch: int = -1 # Tracks the last epoch step was called for

     def step(self, epoch: Optional[int] = None):
          """
          Updates the learning rate based on the current epoch.

          Should be called after each training epoch.

          Args:
              epoch: The current epoch number (0-indexed). If None, increments
                     the internal counter.
          """
          if epoch is None:
               epoch = self.last_epoch + 1
          # Prevent multiple updates in the same epoch
          if epoch == self.last_epoch:
               return
          self.last_epoch = epoch

          # Decay learning rate if epoch is a multiple of step_size (and not the first epoch)
          if epoch > 0 and epoch % self.step_size == 0:
               new_lr = self.lr * self.gamma
               print(f"[Scheduler] Epoch {epoch}: Decaying learning rate from {self.lr:.6g} to {new_lr:.6g} (gamma={self.gamma})")
               self.lr = new_lr

     def get_lr(self) -> float:
          """Returns the current learning rate."""
          return self.lr

# --- Accuracy Calculation ---

def calculate_accuracy(logits_gpu: GPUTensor, targets_np: np.ndarray, current_m_flat: int, vocab_size: int) -> float:
     """
     Calculates the prediction accuracy based on logits and target labels.

     Reads logits from the GPU, finds the predicted class (argmax), compares
     with host-side targets, and computes accuracy, ignoring padded elements.

     Args:
         logits_gpu: GPUTensor containing the raw logits from the model
                     (shape current_m_flat x vocab_size).
         targets_np: NumPy array containing the ground truth target labels on host
                     (shape B x S or flattened B*S). Used for comparison and padding mask.
         current_m_flat: The actual flattened dimension (B * S) for the current batch.
         vocab_size: The size of the vocabulary (number of classes).

     Returns:
         float: The calculated accuracy (correct predictions / total valid samples).
                Returns 0.0 if there are no valid (non-padded) samples.
     """
     # Basic checks for empty inputs.
     if logits_gpu.size == 0 or targets_np.size == 0 or current_m_flat <= 0:
         return 0.0

     # Allocate host buffer for the relevant part of the logits.
     logits_host_flat = np.zeros(current_m_flat * vocab_size, dtype=FP_TYPE)
     try:
         # Read logits from GPU for the current batch size.
         logits_gpu.read(logits_host_flat, offset_bytes=0)
     except Exception as e:
         print(f"[Accuracy] Error reading logits from GPU: {e}")
         return 0.0 # Cannot calculate accuracy if logits cannot be read

     # Reshape logits to (current_m_flat, vocab_size) for argmax.
     try:
        logits_host = logits_host_flat.reshape(current_m_flat, vocab_size)
     except ValueError:
         print(f"[Accuracy] Error reshaping flat logits ({len(logits_host_flat)}) to ({current_m_flat}, {vocab_size}).")
         del logits_host_flat
         return 0.0

     # Find the index of the highest logit value for each sample -> predicted class.
     predictions_flat = np.argmax(logits_host, axis=1)
     # Flatten targets and select the relevant part for the current batch.
     targets_flat = targets_np.flatten()[:current_m_flat]

     # Create mask to ignore padded targets.
     valid_mask = (targets_flat != PAD_INDEX)

     # Check if there are any valid samples to calculate accuracy on.
     if not np.any(valid_mask):
         del logits_host_flat, logits_host, predictions_flat, targets_flat, valid_mask
         return 0.0 # Accuracy is 0 if the batch contains only padding

     # Count correct predictions only where the target is valid.
     correct_predictions = np.sum(predictions_flat[valid_mask] == targets_flat[valid_mask])
     # Count total number of valid (non-padded) target samples.
     total_valid = np.sum(valid_mask)

     # Calculate accuracy.
     accuracy = float(correct_predictions) / total_valid if total_valid > 0 else 0.0

     # Clean up host memory.
     del logits_host_flat, logits_host, predictions_flat, targets_flat, valid_mask
     return accuracy

# --- Helper Functions for Text Sampling/Generation ---

def encode(text: str, char_to_id: Mapping[str, int]) -> List[int]:
    """
    Converts a string into a list of integer token IDs using the vocabulary mapping.

    Characters not found in the `char_to_id` map are currently mapped to ID 0.
    Consider raising an error or using a dedicated <UNK> token ID if available.

    Args:
        text: The input string.
        char_to_id: Dictionary mapping characters to integer IDs.

    Returns:
        List[int]: A list of corresponding integer token IDs.
    """
    # Use dict.get() with a default value (e.g., 0 or a specific UNK_ID)
    # for characters not present in the vocabulary.
    unknown_char_id = 0 # Or potentially char_to_id['<UNK>'] if it exists
    return [char_to_id.get(char, unknown_char_id) for char in text]

def decode(ids: Union[List[int], np.ndarray], id_to_char: Mapping[int, str]) -> str:
    """
    Converts a sequence of integer token IDs back into a string.

    Unknown IDs (not found in `id_to_char`) are mapped to '?'.

    Args:
        ids: A list or NumPy array of integer token IDs.
        id_to_char: Dictionary mapping integer IDs back to characters.

    Returns:
        str: The resulting decoded string.
    """
    unknown_id_char = '?' # Character to use for unknown IDs
    return "".join([id_to_char.get(token_id, unknown_id_char) for token_id in ids])

def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Computes softmax probabilities from logits with temperature scaling.

    Uses numerical stability trick (subtracting max logit).

    Args:
        logits: A NumPy array of raw scores (logits).
        temperature: A scaling factor applied before exponentiation. Lower values
                     make the distribution sharper (less random), higher values
                     make it flatter (more random). Must be positive. Default: 1.0.

    Returns:
        np.ndarray: A NumPy array of probabilities that sum to 1.0.
    """
    if temperature <= 1e-9: # Avoid division by zero or near-zero
        # Handle near-zero temperature by making it deterministic (argmax)
        # or setting a minimum positive temperature. Let's use a minimum.
        temperature = 1e-9
        print(f"[Softmax] Warning: Temperature ({temperature}) too low, clamped to {1e-9}.")

    # Apply temperature scaling.
    scaled_logits = logits / temperature
    # Subtract max logit for numerical stability (prevents overflow in exp).
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    # Normalize to get probabilities. Add epsilon to prevent division by zero if all exp_logits are zero.
    sum_exp_logits = np.sum(exp_logits)
    probabilities = exp_logits / (sum_exp_logits + 1e-9) # Add epsilon for safety

    # Optional: Renormalize to ensure sum is exactly 1 due to potential float precision issues
    probabilities /= np.sum(probabilities)

    return probabilities

def sample_from_probs(probs: np.ndarray) -> int:
    """
    Samples an index randomly based on the provided probability distribution.

    Args:
        probs: A 1D NumPy array representing a probability distribution (non-negative, sums to ~1).

    Returns:
        int: The sampled index.
    """
    # Ensure probabilities sum to 1, handling potential floating point inaccuracies.
    probs = probs / np.sum(probs)
    # Use numpy's multinomial sampling or choice. `choice` is simpler for single samples.
    return np.random.choice(len(probs), p=probs)

# --- Text Generation Function ---

def generate_text(
    model_instance: MyModel,
    prompt: str,
    num_chars_to_generate: int,
    seq_len: int,
    vocab_size: int,
    char_to_id: Mapping[str, int],
    id_to_char: Mapping[int, str],
    sampling_input_gpu: GPUTensor, # Reusable GPU tensor for input IDs
    temperature: float = 0.8,
    gpu_index: int = GPU_INDEX # Ensure correct GPU index if multiple are available
) -> str:
    """
    Generates text character by character starting from a given prompt.

    Uses the trained model in evaluation mode. Samples the next character based
    on the predicted probability distribution scaled by temperature.

    Args:
        model_instance: The trained MyModel instance.
        prompt: The initial string to seed the generation.
        num_chars_to_generate: The number of new characters to generate after the prompt.
        seq_len: The sequence length the model expects as input.
        vocab_size: The size of the vocabulary.
        char_to_id: Mapping from character to ID.
        id_to_char: Mapping from ID to character.
        sampling_input_gpu: A pre-allocated GPUTensor of size (1 * seq_len * int_size)
                            to hold the input sequence for the GPU during generation.
        temperature: The sampling temperature (controls randomness).
        gpu_index: The index of the GPU to use for generation.

    Returns:
        str: The prompt string followed by the generated characters.
             Returns "[GENERATION ERROR]" if an error occurs.
    """
    print(f"\n--- Generating Text ---")
    print(f"Prompt: '{prompt}'")
    print(f"Chars to generate: {num_chars_to_generate}")
    print(f"Temperature: {temperature}")
    print("-" * 25)

    # Set model to evaluation mode (disables dropout, etc., if implemented)
    model_instance.eval_mode()

    # Encode the initial prompt into token IDs.
    tokens = encode(prompt, char_to_id)
    # Create a host buffer for the input sequence IDs (size: seq_len).
    input_buffer_host = np.full(seq_len, PAD_INDEX, dtype=INT_TYPE) # Fill with PAD initially
    # Create a host buffer to receive logits from the GPU (size: seq_len * vocab_size).
    # We only need the last logit, but reading the whole chunk might be easier/faster depending on API.
    logits_buffer_host = np.zeros(seq_len * vocab_size, dtype=FP_TYPE)

    generation_start_time = time.time()
    generated_chars_count = 0

    try:
        for i in range(num_chars_to_generate):
            # 1. Prepare Input Sequence: Take the last `seq_len` tokens.
            current_input_tokens = tokens[-seq_len:] # Get the most recent tokens
            num_current_tokens = len(current_input_tokens)

            # Place the current tokens at the END of the host buffer, padding the start.
            input_buffer_host.fill(PAD_INDEX) # Reset buffer
            start_pos = seq_len - num_current_tokens
            input_buffer_host[start_pos:] = current_input_tokens
            # Example: seq_len=5, tokens=[10, 11, 12] -> buffer = [PAD, PAD, 10, 11, 12]

            # 2. Write Input to GPU: Use the dedicated sampling tensor.
            # Ensure the tensor has the correct size (1 * seq_len * itemsize_int).
            if sampling_input_gpu.size != seq_len * INT_TYPE().itemsize:
                 raise ValueError(f"Sampling input GPU tensor has incorrect size ({sampling_input_gpu.size}) "
                                  f"Expected {seq_len * INT_TYPE().itemsize}")
            sampling_input_gpu.write(input_buffer_host) # Write the prepared buffer

            # 3. Model Forward Pass: Use batch size 1.
            model_instance.set_current_batch_shape(1, seq_len) # Set B=1, S=seq_len
            logits_gpu = model_instance.forward(sampling_input_gpu) # Pass the sampling tensor

            # 4. Read Logits from GPU: Get the full output (1*seq_len, V).
            if logits_gpu.size < seq_len * vocab_size * FP_TYPE().itemsize:
                 # This might happen if the model internally adjusts output size based on M_flat,
                 # ensure the read matches the actual output size. Read might need adjustment
                 # if the C function only fills part of the buffer based on current_m_flat.
                 # Assuming C function fills based on current_m_flat (1 * seq_len):
                 read_size = 1 * seq_len * vocab_size # Should match logits_gpu.size
                 if logits_buffer_host.size < read_size: # Check host buffer size
                      logits_buffer_host = np.zeros(read_size, dtype=FP_TYPE)
                 logits_gpu.read(logits_buffer_host[:read_size], offset_bytes=0) # Read into correctly sized host buffer
            else:
                 logits_gpu.read(logits_buffer_host, offset_bytes=0) # Read the full buffer if sizes match

            # Reshape and extract the logits corresponding to the *last* input token.
            # Output shape is (M_flat, V) -> (1 * seq_len, V). We need the logits
            # at index `seq_len - 1` (the prediction for the *next* token).
            try:
                logits_matrix = logits_buffer_host.reshape((seq_len, vocab_size))
            except ValueError:
                 print(f"[Generate] Error reshaping flat logits ({len(logits_buffer_host)}) to ({seq_len}, {vocab_size}). Skipping step {i}.")
                 continue # Skip this generation step if reshape fails

            last_logits = logits_matrix[-1] # Logits for predicting the token after the last input token.

            # 5. Apply Softmax and Sample: Get probabilities and sample the next token ID.
            probs = softmax(last_logits, temperature=temperature)
            next_token = sample_from_probs(probs)

            # 6. Append Sampled Token: Add the new token ID to our sequence.
            tokens.append(next_token)
            generated_chars_count += 1

            # Optional: Print generated character immediately.
            # sys.stdout.write(id_to_char.get(next_token, '?'))
            # sys.stdout.flush()

            # Optional: Progress indicator.
            if (i + 1) % 100 == 0:
                print(f"\n[Generate] Generated {i+1}/{num_chars_to_generate} characters...")

    except KeyboardInterrupt:
        print("\n[Generate] Text generation interrupted by user.")
    except Exception as e:
        print(f"\n[Generate] ERROR during text generation: {e}")
        import traceback
        traceback.print_exc()
        # Return the prompt and whatever was generated before the error.
        generated_sequence = decode(tokens, id_to_char)
        return generated_sequence + "\n[GENERATION ERROR]"
    finally:
        generation_duration = time.time() - generation_start_time
        print(f"\n--- Generation Finished ({generated_chars_count} chars in {generation_duration:.2f} sec) ---")
        # Restore model to training mode if needed elsewhere.
        # model_instance.train_mode() # Or handle mode switching outside this function.

    # 7. Decode Final Sequence: Convert the full list of token IDs back to a string.
    generated_sequence = decode(tokens, id_to_char)
    return generated_sequence


# --- Main Execution Block ---
if __name__ == "__main__":

    # --- Initialization and Setup ---
    model: Optional[MyModel] = None
    train_inputs: Optional[np.ndarray] = None
    train_targets: Optional[np.ndarray] = None
    valid_inputs: Optional[np.ndarray] = None
    valid_targets: Optional[np.ndarray] = None
    char_to_id: Optional[Mapping[str, int]] = None
    id_to_char: Optional[Mapping[int, str]] = None
    # GPU tensors for batch data during training/validation
    input_gpu: Optional[GPUTensor] = None
    targets_gpu: Optional[GPUTensor] = None
    # Dedicated GPU tensor for input during text generation (B=1)
    sampling_input_gpu: Optional[GPUTensor] = None
    # Training progress variables
    global_step: int = 0
    start_epoch: int = 0
    best_valid_loss: float = float('inf')
    current_epoch: int = 0 # Variable to track current epoch for cleanup/saving

    try:
        # --- GPU Selection ---
        # List available GPUs using PyOpenCL (if installed) or assume index 0.
        available_gpus = list_available_gpus()
        # Let user select GPU if multiple are available.
        selected_gpu_index = select_gpu_index(available_gpus)
        GPU_INDEX = selected_gpu_index # Update global constant
        # Display selected GPU info.
        if cl and available_gpus: # Check if PyOpenCL was available and found GPUs
             selected_gpu_info = available_gpus[GPU_INDEX][1]
             print(f"[Main] Using GPU {GPU_INDEX}: {selected_gpu_info.name}")
        else:
             print(f"[Main] Using GPU Index {GPU_INDEX} (details unavailable or PyOpenCL not used).")


        # --- GPU Initialization via C Driver ---
        # This step tries to initialize the selected GPU context in the C library.
        initialize_selected_gpu(GPU_INDEX)

        # --- Dataset Preparation/Loading ---
        # Check if input text file exists, create a dummy if not.
        if not os.path.exists(input_text_file):
             print(f"WARNING: Input file '{input_text_file}' not found.")
             print("Creating a small dummy input file for demonstration.")
             # Create some simple repetitive text.
             dummy_text = "This is a simple example text for the character model.\n" \
                          "It repeats several times to provide enough data for short sequences.\n" * 50
             with open(input_text_file, 'w', encoding='utf-8') as f:
                 f.write(dummy_text)
             print(f"Dummy file created at '{input_text_file}'.")

        # Preprocess the text data (creates .npz and _vocab.pkl if they don't exist).
        preprocess_char_data(input_text_file, processed_data_file, SEQ_LEN)
        # Load the processed data arrays and vocabulary mappings.
        train_inputs, train_targets, valid_inputs, valid_targets, \
            loaded_vocab_size, char_to_id, id_to_char = load_processed_data(processed_data_file)

        # Update global VOCAB_SIZE based on loaded data.
        VOCAB_SIZE = loaded_vocab_size
        if VOCAB_SIZE <= 0:
            raise ValueError("Vocabulary size could not be loaded or is invalid.")

        # Validate and potentially adjust BATCH_SIZE if dataset is very small.
        num_train_samples = train_inputs.shape[0]
        num_valid_samples = valid_inputs.shape[0]
        if num_train_samples == 0:
             raise ValueError("Loaded training data has 0 samples. Cannot train.")
        if BATCH_SIZE > num_train_samples:
             print(f"[Main] WARNING: Configured BATCH_SIZE ({BATCH_SIZE}) is larger than number of training samples ({num_train_samples}).")
             BATCH_SIZE = num_train_samples
             print(f"              Adjusting BATCH_SIZE to {BATCH_SIZE}.")

        # --- Model, Optimizer, Scheduler Initialization / Checkpoint Loading ---
        # Calculate max flattened size based on potentially adjusted BATCH_SIZE.
        M_flat_max = BATCH_SIZE * SEQ_LEN
        print(f"[Main] Initializing model with: B={BATCH_SIZE}, S={SEQ_LEN}, E={EMBEDDING_DIM}, H={HIDDEN_DIM}, V={VOCAB_SIZE}, T={NUM_TOKEN_PROTOTYPES}")
        model = MyModel(BATCH_SIZE, SEQ_LEN, EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, NUM_TOKEN_PROTOTYPES, gpu_index=GPU_INDEX)

        # Attempt to load the latest checkpoint.
        start_epoch, global_step, loaded_best_loss = model.load_checkpoint(checkpoint_path)
        best_valid_loss = loaded_best_loss if loaded_best_loss is not None else float('inf')

        # Additionally, try loading the 'best' checkpoint to potentially recover an older, better state.
        if os.path.exists(best_checkpoint_path):
             print(f"[Main] Found best checkpoint file: '{best_checkpoint_path}'. Loading to check best loss.")
             # Load state from best checkpoint temporarily just to get its loss value.
             _, _, best_loss_from_file = model.load_checkpoint(best_checkpoint_path)
             # Update our tracked best_valid_loss if the one from the best file is better.
             if best_loss_from_file is not None and best_loss_from_file < best_valid_loss:
                  print(f"[Main] Best checkpoint file loss ({best_loss_from_file:.6f}) is better than latest checkpoint's ({best_valid_loss:.6f}). Updating best loss.")
                  best_valid_loss = best_loss_from_file
             else:
                  print(f"[Main] Best checkpoint file loss ({best_loss_from_file:.6f}) is not better than latest's ({best_valid_loss:.6f}). Keeping latest best loss.")
             # IMPORTANT: Reload the *latest* checkpoint again to ensure we continue from the correct step count and optimizer state.
             print(f"[Main] Reloading latest checkpoint '{checkpoint_path}' to resume training state.")
             start_epoch, global_step, _ = model.load_checkpoint(checkpoint_path) # Discard loss value this time

        print(f"[Main] Effective best validation loss to beat: {best_valid_loss:.6f}")

        # Initialize the learning rate scheduler.
        scheduler = StepLR(INITIAL_LEARNING_RATE, step_size=LR_DECAY_STEP, gamma=LR_DECAY_GAMMA)
        # Advance the scheduler state to match the loaded start_epoch.
        # Set last_epoch correctly so the *first* call uses the initial LR.
        scheduler.last_epoch = start_epoch - 1
        # Manually step scheduler for epochs already completed before the checkpoint.
        # for i in range(start_epoch):
        #     scheduler.step(i) # This logic might be incorrect if step depends on last state. Better to restore LR from checkpoint if saved, or just set last_epoch.

        current_lr = scheduler.get_lr() # Get the LR for the *start* of the next epoch
        print(f"[Main] Training will resume from Epoch {start_epoch + 1}, Global Step {global_step + 1}, with initial LR {current_lr:.6g}")

        # --- Allocate GPU Tensors for Batch Data ---
        itemsize_int = INT_TYPE().itemsize
        itemsize_fp = FP_TYPE().itemsize
        print("[Main] Allocating GPU buffers for training batches...")
        # Tensor for input sequences (batch_size * seq_len)
        input_gpu = GPUTensor(BATCH_SIZE * SEQ_LEN * itemsize_int, name="InputIndices_batch", gpu_index=GPU_INDEX)
        # Tensor for target sequences (batch_size * seq_len)
        targets_gpu = GPUTensor(BATCH_SIZE * SEQ_LEN * itemsize_int, name="Targets_batch", gpu_index=GPU_INDEX)
        print("[Main] Allocating GPU buffer for sampling/generation (B=1)...")
        # Separate, smaller tensor for text generation input (1 * seq_len)
        sampling_input_gpu = GPUTensor(1 * SEQ_LEN * itemsize_int, name="SamplingInput_GPU", gpu_index=GPU_INDEX)

        # --- Training Loop ---
        print("\n" + "="*15 + f" Starting Training Loop " + "="*15)
        print(f"Target Epochs: {NUM_EPOCHS} | Starting Epoch: {start_epoch + 1}")
        print(f"Training Samples: {num_train_samples} | Validation Samples: {num_valid_samples}")
        print(f"Batch Size: {BATCH_SIZE} | Steps per Epoch approx: {num_train_samples // BATCH_SIZE}")
        print("="*50)

        for epoch in range(start_epoch, NUM_EPOCHS):
            current_epoch = epoch # Store for potential saving in exception handlers
            # Get the learning rate for the *start* of this epoch.
            # Scheduler step happens *after* the epoch.
            epoch_lr = scheduler.get_lr()
            print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} | Learning Rate: {epoch_lr:.6g} ---")
            epoch_start_time = time.time()

            # --- Training Phase ---
            model.train_mode() # Set layers to training mode
            epoch_train_loss: float = 0.0
            num_train_batches: int = 0
            # Create a generator for training batches (shuffled each epoch).
            batch_generator = create_batches(train_inputs, train_targets, BATCH_SIZE, SEQ_LEN, shuffle=True)

            for batch_idx, (batch_inputs_host, batch_targets_host) in enumerate(batch_generator):
                # Determine the actual size of the current batch (might be smaller for the last one).
                current_batch_size = batch_inputs_host.shape[0]
                if current_batch_size == 0: continue # Skip empty batches if they occur

                # Check if the batch consists only of padding (can happen if last batch logic aligns perfectly with padding).
                if np.all(batch_inputs_host == PAD_INDEX):
                    if DEBUG_PRINTS: print(f"[Train] Skipping batch {batch_idx} as it contains only PAD_INDEX.")
                    continue

                # Increment global step counter.
                global_step += 1
                num_train_batches += 1

                # Inform the model about the current batch size.
                model.set_current_batch_shape(current_batch_size, SEQ_LEN)

                # Write batch data to GPU tensors (flatten required).
                input_gpu.write(batch_inputs_host.flatten())
                targets_gpu.write(batch_targets_host.flatten())

                # --- Forward Pass ---
                logits = model.forward(input_gpu)

                # --- Loss Calculation and Backward Pass ---
                loss, _ = model.compute_loss_and_backward(logits, targets_gpu, batch_targets_host)

                # --- Gradient Sanity Check & Updates ---
                # Check for invalid loss values (NaN or Infinity).
                if not math.isfinite(loss):
                     print(f"[WARN] Invalid loss detected ({loss}) in Epoch {epoch+1}, Batch {batch_idx}. Skipping parameter updates for this batch.")
                     # Optional: Zero out gradients manually if they might be corrupted.
                     # model.embedding_layer.dW_emb._zero_initialize()
                     # model.bio_layer.dW1._zero_initialize()
                     # ... etc for all grad tensors ...
                     continue # Proceed to the next batch without updating

                # Add the valid loss to the epoch total.
                epoch_train_loss += loss

                # (Optional) Gradient Clipping
                model.clip_all_gradients(GRADIENT_CLIP_VALUE)

                # Update trainable parameters (Adam step).
                model.update_trainable(global_step, epoch_lr, weight_decay=WEIGHT_DECAY)

                # Perform special updates (Hebbian, Prototypes).
                model.update_special()

                # --- Logging ---
                # Print progress periodically within the epoch.
                if num_train_batches % 50 == 0: # Log every 50 batches
                    print(f"  [Epoch {epoch+1}, Batch {num_train_batches}/{num_train_samples // BATCH_SIZE}] Loss: {loss:.6f}")

            # --- End of Training Phase for Epoch ---
            avg_train_loss = epoch_train_loss / num_train_batches if num_train_batches > 0 else 0.0
            print(f"[Epoch {epoch + 1}] Training finished. Average Training Loss: {avg_train_loss:.6f}")

            # --- Validation Phase ---
            epoch_valid_loss: float = 0.0
            epoch_valid_accuracy: float = 0.0
            num_valid_batches: int = 0
            num_valid_batches_skipped: int = 0
            model.eval_mode() # Set layers to evaluation mode

            # Create a generator for validation batches (no shuffling needed).
            valid_batch_generator = create_batches(valid_inputs, valid_targets, BATCH_SIZE, SEQ_LEN, shuffle=False)

            for batch_inputs_host, batch_targets_host in valid_batch_generator:
                current_batch_size = batch_inputs_host.shape[0]
                if current_batch_size == 0: continue
                if np.all(batch_inputs_host == PAD_INDEX): continue # Skip padding-only batches

                # Inform model of current batch size.
                model.set_current_batch_shape(current_batch_size, SEQ_LEN)

                # Write data to GPU.
                input_gpu.write(batch_inputs_host.flatten())
                targets_gpu.write(batch_targets_host.flatten())

                # --- Forward Pass Only ---
                logits = model.forward(input_gpu)

                # --- Calculate Loss (No Backward Pass) ---
                # Use criterion.forward but ignore the gradient calculation side-effect.
                loss = model.criterion.forward(logits, targets_gpu, model.bio_layer.current_M_flat, targets_np=batch_targets_host)

                # --- Calculate Accuracy ---
                # Only calculate metrics if loss is valid.
                if math.isfinite(loss):
                    accuracy = calculate_accuracy(logits, batch_targets_host, model.bio_layer.current_M_flat, VOCAB_SIZE)
                    epoch_valid_loss += loss
                    epoch_valid_accuracy += accuracy
                    num_valid_batches += 1 # Count valid batches processed
                else:
                    # Skip batch if loss is invalid (NaN/Inf).
                    print(f"[WARN] Invalid Validation Loss ({loss}) encountered. Skipping metrics for this batch.")
                    num_valid_batches_skipped += 1

            # --- End of Validation Phase for Epoch ---
            # Calculate average validation metrics, avoiding division by zero.
            if num_valid_batches > 0:
                 avg_valid_loss = epoch_valid_loss / num_valid_batches
                 avg_valid_accuracy = epoch_valid_accuracy / num_valid_batches
            else:
                 avg_valid_loss = float('inf') # Indicate failure if no valid batches
                 avg_valid_accuracy = 0.0
                 print("[WARN] No valid validation batches were processed in this epoch.")

            epoch_duration = time.time() - epoch_start_time
            print(f"[Epoch {epoch + 1}] Validation finished.")
            print(f"  Avg Validation Loss:      {avg_valid_loss:.6f}")
            print(f"  Avg Validation Accuracy:  {avg_valid_accuracy:.4f}")
            if num_valid_batches_skipped > 0:
                print(f"  ({num_valid_batches_skipped} validation batches skipped due to invalid loss)")
            print(f"  Epoch Duration:           {epoch_duration:.2f} seconds")

            # --- Checkpointing ---
            # Save the latest checkpoint after each epoch.
            model.save_checkpoint(checkpoint_path, epoch + 1, global_step, best_loss=best_valid_loss)
            # Save a separate checkpoint if this epoch achieved the best validation loss so far.
            if avg_valid_loss < best_valid_loss:
                print(f"[Epoch {epoch + 1}] New best validation loss achieved! ({avg_valid_loss:.6f} < {best_valid_loss:.6f}). Saving best checkpoint.")
                best_valid_loss = avg_valid_loss
                # Save to the 'best' checkpoint file path.
                model.save_checkpoint(best_checkpoint_path, epoch + 1, global_step, best_loss=best_valid_loss)

            # --- Learning Rate Decay ---
            # Step the scheduler *after* the epoch is fully completed (including validation).
            scheduler.step(epoch) # Pass 0-indexed epoch number

            # --- Text Generation Sample ---
            # Generate a sample text after each epoch to monitor progress qualitatively.
            if model is not None and char_to_id is not None and id_to_char is not None and sampling_input_gpu is not None:
                # Define a fixed prompt and parameters for consistency.
                sampling_prompt = "Der Sinn des Lebens ist" # Example prompt
                num_chars_to_generate = 250
                sampling_temperature = 0.7 # Adjust as needed

                print("\n" + "-"*10 + " Generating Sample Text " + "-"*10)
                generated_text = generate_text(
                    model_instance=model,
                    prompt=sampling_prompt,
                    num_chars_to_generate=num_chars_to_generate,
                    seq_len=SEQ_LEN,
                    vocab_size=VOCAB_SIZE,
                    char_to_id=char_to_id,
                    id_to_char=id_to_char,
                    sampling_input_gpu=sampling_input_gpu,
                    temperature=sampling_temperature,
                    gpu_index=GPU_INDEX
                )
                # Print the generated text clearly demarcated.
                print("\n--- Generated Text Sample ---")
                print(generated_text)
                print("--- End Generated Text Sample ---")
            # --- End Text Generation ---


        # --- End of Training Loop ---
        print("\n" + "="*15 + " Training Loop Completed " + "="*15)

    except KeyboardInterrupt:
         # Handle graceful exit on Ctrl+C.
         print("\n[Main] Training interrupted by user (KeyboardInterrupt).")
         if model is not None and global_step > 0:
              # Attempt to save the current state before exiting.
              print("[Main] Attempting to save final checkpoint before exiting...")
              # Use the epoch number where interruption occurred.
              final_epoch = current_epoch + 1 # Save as if this epoch finished
              model.save_checkpoint(checkpoint_path, final_epoch, global_step, best_loss=best_valid_loss)
              print("[Main] Final checkpoint saved.")
    except Exception as e:
        # Catch any other unexpected errors during training.
        print(f"\n--- An Unexpected Error Occurred in the Main Loop ---")
        import traceback
        traceback.print_exc() # Print detailed error information.
        # Optionally try to save checkpoint on other errors too?
        # if model is not None and global_step > 0:
        #      print("[Main] Attempting to save checkpoint after error...")
        #      final_epoch = current_epoch + 1
        #      model.save_checkpoint(checkpoint_path + ".error", final_epoch, global_step, best_loss=best_valid_loss)

        sys.exit(1) # Exit with error code
    finally:
        # --- Resource Cleanup ---
        # This block executes whether the loop finished normally, was interrupted, or an error occurred.
        print("\n[Python] Cleaning up resources...")
        # List all major resources that need explicit cleanup.
        # model.free() handles freeing resources within the model's layers.
        resource_list = [input_gpu, targets_gpu, sampling_input_gpu, model]
        for res in resource_list:
             try:
                 if res is not None:
                      # Check if the object has a 'free' method and call it.
                      if hasattr(res, 'free') and callable(res.free):
                          if DEBUG_PRINTS: print(f"  Freeing {getattr(res, 'name', type(res).__name__)}...")
                          res.free()
             except Exception as cleanup_e:
                  # Log errors during cleanup but continue cleaning other resources.
                  item_name = getattr(res, 'name', type(res).__name__) if res else "None"
                  print(f"  ERROR during cleanup of {item_name}: {cleanup_e}")

        # Shutdown the GPU context via the C driver.
        if 'c_driver' in locals() and c_driver:
            print("[Python] Calling C driver shutdown_gpu...")
            try:
                c_driver.shutdown_gpu(GPU_INDEX)
                print("[Python] C driver shutdown_gpu called successfully.")
            except Exception as shutdown_e:
                print(f"ERROR during C driver shutdown_gpu call: {shutdown_e}")
        else:
            print("[Python] C driver not loaded or already cleaned up, skipping shutdown_gpu call.")

        print("[Python] Cleanup finished. Program exiting.")
