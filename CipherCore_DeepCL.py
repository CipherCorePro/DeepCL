# -*- coding: utf-8 -*-
"""
Python-Code zur Verwendung des C/OpenCL-Treibers für ein gehirnähnliches Netzwerk.
Angepasst für Charakter-Level Text Generation mit Embedding Layer.
"""
import ctypes
import numpy as np
import os
import platform
import time
import math
import pickle # Für Checkpoints
from typing import Optional, Tuple, Generator, Dict, Any, List, Mapping, Union # Typ-Hinweise angepasst
import sys
from collections import Counter # Für Vokabular

try:
    import pyopencl as cl # Zur GPU-Erkennung
except ImportError:
    print("FEHLER: PyOpenCL nicht gefunden. Bitte installieren: pip install pyopencl")
    sys.exit(1)


# --- Konstanten und Konfiguration ---
FP_TYPE = np.float32
FP_TYPE_C = ctypes.c_float
INT_TYPE = np.int32
INT_TYPE_C = ctypes.c_int
GPU_INDEX = 0 # Standard, wird ggf. durch Benutzerauswahl überschrieben
DEBUG_PRINTS = False # Reduziert, um Output übersichtlicher zu machen (True für Details)
ADAM_EPS = 1e-8
PAD_INDEX = -1 # Index für Padding in Targets

# --- Trainings-Hyperparameter ---
INITIAL_LEARNING_RATE = 1e-2 # Etwas niedriger starten für Textgen
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 13 # Reduziert für ersten Test
BATCH_SIZE = 64
GRADIENT_CLIP_VALUE = 1.0 # Wert für Clipping, None zum Deaktivieren
LR_DECAY_STEP = 1 # Lernrate schneller anpassen
LR_DECAY_GAMMA = 0.7

# --- Modell-Hyperparameter ---
SEQ_LEN = 64  # Längere Sequenzen für Text sinnvoll
EMBEDDING_DIM = 128 # Dimension der Vektoren nach dem Embedding (= Input für BioLayer)
HIDDEN_DIM = 384 # Eventuell vergrößern
VOCAB_SIZE = -1 # Wird aus Daten geladen!
NUM_TOKEN_PROTOTYPES = 384 # Mehr Prototypen können helfen
# BioLayer spezifisch
HEBBIAN_LR = 0.01 # Eventuell anpassen
SPIKE_THRESHOLD = 0.45
PROTOTYPE_LR = 0.005 # Eventuell anpassen
USE_GPU_PROTOTYPE_UPDATE = True # GPU Update nutzen, da es jetzt funktioniert

# --- Pfadkonfiguration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
cl_dir = os.path.join(script_dir, "CL")
data_dir = os.path.join(script_dir, "mini_data")
checkpoint_dir = os.path.join(script_dir, "mini_checkpoints")
os.makedirs(data_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# NEU: Pfade für echte Daten
input_text_file = os.path.join(data_dir, "mini_input.txt") # <<<=== Lege hier deine Textdatei ab!
processed_data_file = os.path.join(data_dir, "mini_char_dataset.npz")

checkpoint_filename = f"model_char_emb{EMBEDDING_DIM}_h{HIDDEN_DIM}_t{NUM_TOKEN_PROTOTYPES}.pkl"
checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
best_checkpoint_path = os.path.join(checkpoint_dir, f"best_{checkpoint_filename}")

# --- WICHTIG: Füge das CL-Verzeichnis zum DLL-Suchpfad hinzu (Windows, Python 3.8+) ---
# ... (unverändert) ...
dll_load_error = None
if platform.system() == "Windows":
    if hasattr(os, 'add_dll_directory'):
        try:
            if DEBUG_PRINTS: print(f"[Python] Adding DLL search directory: {cl_dir}")
            _ = os.add_dll_directory(cl_dir)
        except OSError as e:
            dll_load_error = f"Warning: Could not add DLL directory {cl_dir}: {e}. Abhängigkeiten werden möglicherweise nicht gefunden."
            print(f"[Python] {dll_load_error}")
    else:
        dll_load_error = "Warning: os.add_dll_directory nicht verfügbar (benötigt Python 3.8+). Abhängigkeiten müssen im PATH sein oder im Skriptverzeichnis."
        print(f"[Python] {dll_load_error}")

# Laden der kompilierten C-Bibliothek aus dem CL-Ordner
lib_name = "CipherCore_OpenCl.dll" if platform.system() == "Windows" else "libsimulated_driver.so"
lib_path = os.path.join(cl_dir, lib_name)

c_driver = None # Initialisieren für finally Block
if not os.path.exists(lib_path):
    raise ImportError(f"Kompilierte Bibliothek nicht gefunden unter: {lib_path}\n"
                      "Bitte kompilieren Sie zuerst den C-Code (z.B. mit gcc) und stellen Sie sicher, dass die DLL/SO im 'CL'-Unterordner liegt.")
try:
    c_driver = ctypes.CDLL(lib_path)
except OSError as e:
     print("\n--- Detaillierte Ladefehler-Info ---")
     print(f"Versuchter Pfad: {lib_path}")
     print(f"Betriebssystem: {platform.system()} {platform.architecture()}")
     print(f"Python-Version: {platform.python_version()}")
     print(f"Aktuelles Verzeichnis: {os.getcwd()}")
     print(f"CL Verzeichnis: {cl_dir}")
     if dll_load_error: print(f"Warnung zum Suchpfad: {dll_load_error}")
     print(f"Fehler von ctypes: {e}")
     print("Mögliche Ursachen:")
     print("  1. Pfad zur DLL/SO falsch.")
     print("  2. Fehlende Abhängigkeit (z.B. OpenCL.dll/libOpenCL.so) NICHT im 'CL'-Ordner oder einem anderen System-Suchpfad.")
     print("  3. Architektur-Mismatch (32-bit Python vs. 64-bit DLL/SO o.ä.?).")
     print("  4. Fehlende VC++ Redistributables (falls DLL mit MSVC kompiliert wurde).")
     print("  5. Fehlende Standard-C-Laufzeitbibliotheken (selten bei MinGW, aber möglich).")
     print("------------------------------------\n")
     raise ImportError(f"Fehler beim Laden der Bibliothek {lib_path}: {e}")

print(f"[Python] C-Treiber-Bibliothek geladen von: {lib_path}")

# --- GPU Erkennung / Auswahl ---
# ... (list_available_gpus, select_gpu_index, initialize_selected_gpu unverändert) ...
def list_available_gpus() -> List[Tuple[int, cl.Device]]:
    """ Listet verfügbare OpenCL GPUs auf. """
    platforms = cl.get_platforms()
    available_gpus = []
    print("Verfügbare OpenCL GPUs:")
    print("-" * 30)
    global_gpu_index = 0
    for p_idx, platform in enumerate(platforms):
        try:
            devices = platform.get_devices(device_type=cl.device_type.GPU)
            if not devices: continue # Keine GPUs auf dieser Plattform
            print(f"Plattform {p_idx}: {platform.name}")
            for d_idx, device in enumerate(devices):
                 # Zeige relevante Infos
                compute_units = device.max_compute_units
                clock_freq = device.max_clock_frequency
                global_mem_mb = device.global_mem_size // (1024 * 1024)
                print(f"  GPU Index {global_gpu_index}: {device.name}")
                print(f"    Compute Units: {compute_units}, Max Clock: {clock_freq} MHz, Global Memory: {global_mem_mb} MB")
                available_gpus.append((global_gpu_index, device))
                global_gpu_index += 1
        except cl.LogicError as cle:
            # Manchmal können Geräteinformationen nicht abgefragt werden
            print(f"Warnung: Konnte Geräte für Plattform {p_idx} ({platform.name}) nicht vollständig abfragen: {cle}")
        except Exception as e:
             print(f"Fehler beim Abfragen von Plattform {p_idx} ({platform.name}): {e}")
    print("-" * 30)
    if not available_gpus:
         print("Keine OpenCL GPUs gefunden!")
    return available_gpus

def select_gpu_index(available_gpus: List[Tuple[int, cl.Device]]) -> int:
    """ Lässt den Benutzer einen GPU-Index auswählen. """
    if not available_gpus:
        raise RuntimeError("Keine GPUs zur Auswahl verfügbar.")

    if len(available_gpus) == 1:
        print(f"Nur eine GPU (Index 0) gefunden: {available_gpus[0][1].name}. Wird automatisch ausgewählt.")
        return 0

    while True:
        try:
            prompt = f"Bitte wählen Sie den zu verwendenden GPU Index (0 bis {len(available_gpus) - 1}): "
            choice = input(prompt)
            if not choice: # Standard auf 0 bei leerer Eingabe
                 print("Keine Eingabe, verwende GPU 0.")
                 return 0
            gpu_index = int(choice)
            if 0 <= gpu_index < len(available_gpus):
                print(f"GPU {gpu_index}: {available_gpus[gpu_index][1].name} ausgewählt.")
                return gpu_index
            else:
                print(f"Ungültiger Index. Bitte eine Zahl zwischen 0 und {len(available_gpus) - 1} eingeben.")
        except ValueError:
            print("Ungültige Eingabe. Bitte eine Zahl eingeben.")
        except EOFError: # Falls Input umgeleitet wird
             print("Keine Eingabe möglich, verwende GPU 0.")
             return 0

def initialize_selected_gpu(gpu_index: int):
    """ Initialisiert die GPU mit dem gegebenen Index über den C-Treiber. """
    print(f"[Python] Initialisiere GPU mit Index {gpu_index} über C-Treiber...")
    if c_driver.initialize_gpu(gpu_index) == 0:
        raise RuntimeError(f"GPU-Initialisierung fehlgeschlagen für Index {gpu_index}.")
    print(f"[Python] GPU {gpu_index} erfolgreich initialisiert.")


# --- Definition der C-Funktionssignaturen mit ctypes ---
# ... (unverändert) ...
GPU_BUFFER_HANDLE = ctypes.c_void_p
c_driver.initialize_gpu.argtypes = [ctypes.c_int]; c_driver.initialize_gpu.restype = ctypes.c_int
c_driver.allocate_gpu_memory.argtypes = [ctypes.c_int, ctypes.c_size_t]; c_driver.allocate_gpu_memory.restype = GPU_BUFFER_HANDLE
c_driver.free_gpu_memory.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE]; c_driver.free_gpu_memory.restype = None
c_driver.write_host_to_gpu_blocking.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p]; c_driver.write_host_to_gpu_blocking.restype = ctypes.c_int
c_driver.read_gpu_to_host_blocking.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p]; c_driver.read_gpu_to_host_blocking.restype = ctypes.c_int
c_driver.simulated_get_compute_unit_count.argtypes = [ctypes.c_int]; c_driver.simulated_get_compute_unit_count.restype = ctypes.c_uint
c_driver.shutdown_gpu.argtypes = [ctypes.c_int]; c_driver.shutdown_gpu.restype = None
c_driver.execute_clone_on_gpu.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, ctypes.c_size_t]; c_driver.execute_clone_on_gpu.restype = ctypes.c_int
c_driver.execute_matmul_on_gpu.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]; c_driver.execute_matmul_on_gpu.restype = ctypes.c_int
c_driver.execute_add_on_gpu.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, ctypes.c_int]; c_driver.execute_add_on_gpu.restype = ctypes.c_int
c_driver.execute_add_bias_on_gpu.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, ctypes.c_int, ctypes.c_int]; c_driver.execute_add_bias_on_gpu.restype = ctypes.c_int
c_driver.execute_gelu_on_gpu.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, ctypes.c_int]; c_driver.execute_gelu_on_gpu.restype = ctypes.c_int
c_driver.execute_layernorm_on_gpu.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, ctypes.c_int, ctypes.c_int, ctypes.c_float]; c_driver.execute_layernorm_on_gpu.restype = ctypes.c_int
c_driver.execute_log_softmax_stable_gpu.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, ctypes.c_int, ctypes.c_int]; c_driver.execute_log_softmax_stable_gpu.restype = ctypes.c_int
c_driver.execute_embedding_lookup_gpu.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]; c_driver.execute_embedding_lookup_gpu.restype = ctypes.c_int
c_driver.execute_gelu_backward_on_gpu.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, ctypes.c_int]; c_driver.execute_gelu_backward_on_gpu.restype = ctypes.c_int
c_driver.execute_matmul_backward_on_gpu.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]; c_driver.execute_matmul_backward_on_gpu.restype = ctypes.c_int
c_driver.execute_reduce_sum_gpu.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, ctypes.c_int, ctypes.c_int, ctypes.c_int]; c_driver.execute_reduce_sum_gpu.restype = ctypes.c_int
c_driver.execute_cross_entropy_loss_grad_gpu.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, ctypes.c_int, ctypes.c_int]; c_driver.execute_cross_entropy_loss_grad_gpu.restype = ctypes.c_int
c_driver.execute_embedding_backward_gpu.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]; c_driver.execute_embedding_backward_gpu.restype = ctypes.c_int
print("[Python] CTypes-Definition für execute_embedding_backward_gpu (via non-atomic two-pass) geladen.")
c_driver.execute_adam_update_on_gpu.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]; c_driver.execute_adam_update_on_gpu.restype = ctypes.c_int
c_driver.execute_hebbian_update_on_gpu.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]; c_driver.execute_hebbian_update_on_gpu.restype = ctypes.c_int
c_driver.execute_threshold_spike_on_gpu.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, ctypes.c_float, ctypes.c_int]; c_driver.execute_threshold_spike_on_gpu.restype = ctypes.c_int
c_driver.execute_dynamic_token_assignment_gpu.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]; c_driver.execute_dynamic_token_assignment_gpu.restype = ctypes.c_int
c_driver.execute_pairwise_similarity_gpu.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, ctypes.c_int, ctypes.c_int]; c_driver.execute_pairwise_similarity_gpu.restype = ctypes.c_int
CAN_USE_GPU_PROTO_UPDATE = False
try:
    c_driver.execute_proto_segmented_sum_gpu.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    c_driver.execute_proto_segmented_sum_gpu.restype = ctypes.c_int
    c_driver.execute_proto_update_step_gpu.argtypes = [ctypes.c_int, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, GPU_BUFFER_HANDLE, ctypes.c_float, ctypes.c_int, ctypes.c_int]
    c_driver.execute_proto_update_step_gpu.restype = ctypes.c_int
    print("[Python] CTypes-Definitionen für GPU Prototyp-Update geladen.")
    CAN_USE_GPU_PROTO_UPDATE = True
except AttributeError:
    print("[Python] WARNUNG: GPU Prototyp-Update Funktionen nicht in DLL gefunden. Host-Fallback wird verwendet.")

if USE_GPU_PROTOTYPE_UPDATE and not CAN_USE_GPU_PROTO_UPDATE:
     print("[Python] FEHLER: GPU Prototyp-Update angefordert, aber Funktionen nicht in DLL gefunden!")
     sys.exit(1)


# --- GPUTensor Klasse ---
# ... (unverändert, nur Prints reduziert) ...
class GPUTensor:
    def __init__(self, size_in_bytes: int, gpu_index: int = GPU_INDEX, name: str = "Tensor", zero_init: bool = False):
        self.gpu_index = gpu_index
        self.size = int(size_in_bytes)
        self.name = name
        self._is_freed = False
        self.handle = None # Initialisieren

        if self.size < 0:
             raise ValueError(f"Ungültige Größe für GPUTensor '{self.name}': {self.size}")

        if self.size > 0:
            self.handle = c_driver.allocate_gpu_memory(self.gpu_index, self.size)
            if not self.handle:
                raise MemoryError(f"GPU-Speicherallokation fehlgeschlagen für {self.size} Bytes ({self.name}) auf GPU {self.gpu_index}")
            handle_int = ctypes.cast(self.handle, ctypes.c_void_p).value
            if DEBUG_PRINTS: print(f"[Python GPUTensor] Allocated '{self.name}' ({self.size} bytes), handle: {hex(handle_int) if handle_int else 'None'})")
            if zero_init:
                self._zero_initialize()
        elif DEBUG_PRINTS:
             print(f"[Python GPUTensor] Allocated '{self.name}' (0 bytes), handle: None")

    def _zero_initialize(self):
        if self.handle is None or self.size == 0: return
        item_size = 4 # Annahme FP32 oder INT32
        num_elements = self.size // item_size
        zeros_buffer = None # Initialisieren
        if num_elements * item_size != self.size:
            zeros_buffer = np.zeros(self.size, dtype=np.byte)
        else:
             # Versuche, dtype basierend auf Namen zu erraten
            if 'indices' in self.name.lower() or 'counts' in self.name.lower() or 'target' in self.name.lower() or 'InputIndices' in self.name: # <<< NEU: InputIndices
                zeros_buffer = np.zeros(num_elements, dtype=INT_TYPE)
            else:
                zeros_buffer = np.zeros(num_elements, dtype=FP_TYPE)

        try:
            if zeros_buffer is not None:
                self.write(zeros_buffer)
        except Exception as e:
             print(f"[Python GPUTensor] FEHLER bei Zero-Init für '{self.name}': {e}")
        finally:
            if zeros_buffer is not None: del zeros_buffer

    def write(self, host_data_np: np.ndarray, offset_bytes: int = 0):
        if self._is_freed: raise RuntimeError(f"Operation auf freigegebenem Tensor '{self.name}'.")
        if self.handle is None:
             if self.size == 0 and host_data_np.nbytes == 0: return
             raise RuntimeError(f"Versuch, in freigegebenen oder 0-size Tensor '{self.name}' zu schreiben.")
        if not host_data_np.flags['C_CONTIGUOUS']: host_data_np = np.ascontiguousarray(host_data_np)
        data_ptr = host_data_np.ctypes.data_as(ctypes.c_void_p)
        size_bytes = host_data_np.nbytes
        handle_int = ctypes.cast(self.handle, ctypes.c_void_p).value
        if offset_bytes < 0 or size_bytes < 0: raise ValueError(f"Offset/Größe darf nicht negativ sein für '{self.name}'.")
        if offset_bytes + size_bytes > self.size: raise ValueError(f"Schreibvorgang (Offset {offset_bytes}, Größe {size_bytes}) überschreitet allokierte Größe {self.size} für '{self.name}'.")
        if c_driver.write_host_to_gpu_blocking(self.gpu_index, self.handle, offset_bytes, size_bytes, data_ptr) == 0:
            raise RuntimeError(f"Fehler beim Schreiben auf die GPU für '{self.name}'.")

    def read(self, host_data_np: np.ndarray, offset_bytes: int = 0) -> np.ndarray:
        if self._is_freed: raise RuntimeError(f"Operation auf freigegebenem Tensor '{self.name}'.")
        if self.handle is None:
             if self.size == 0 and host_data_np.nbytes == 0: return host_data_np
             raise RuntimeError(f"Versuch, aus freigegebenen oder 0-size Tensor '{self.name}' zu lesen.")
        if not host_data_np.flags['C_CONTIGUOUS']: raise ValueError("Ziel-NumPy-Array muss C-kontinuierlich sein.")
        if not host_data_np.flags['WRITEABLE']: raise ValueError("Ziel-NumPy-Array muss beschreibbar sein.")
        data_ptr = host_data_np.ctypes.data_as(ctypes.c_void_p)
        size_bytes = host_data_np.nbytes
        handle_int = ctypes.cast(self.handle, ctypes.c_void_p).value
        if offset_bytes < 0 or size_bytes < 0: raise ValueError(f"Offset/Größe darf nicht negativ sein für '{self.name}'.")
        if offset_bytes + size_bytes > self.size: raise ValueError(f"Lesevorgang (Offset {offset_bytes}, Größe {size_bytes}) überschreitet allokierte Größe {self.size} für '{self.name}'.")
        if c_driver.read_gpu_to_host_blocking(self.gpu_index, self.handle, offset_bytes, size_bytes, data_ptr) == 0:
            raise RuntimeError(f"Fehler beim Lesen von der GPU für '{self.name}'.")
        return host_data_np

    def free(self):
        if not self._is_freed and self.handle:
            handle_to_free = self.handle
            handle_int = ctypes.cast(handle_to_free, ctypes.c_void_p).value
            if DEBUG_PRINTS: print(f"[Python GPUTensor] Freeing '{self.name}' (handle: {hex(handle_int)})")
            try: c_driver.free_gpu_memory(self.gpu_index, handle_to_free)
            except Exception as e: print(f"[Python GPUTensor] WARNUNG: Fehler beim Freigeben von '{self.name}': {e}")
            finally: self.handle = None; self.size = 0
        self._is_freed = True

    def __del__(self):
        if not getattr(self, '_is_freed', True):
             if DEBUG_PRINTS: print(f"[Python GPUTensor] Destructor called for '{self.name}' - ensuring memory is freed.")
             self.free()

# --- NEU: Datenverarbeitung für Charakter-Level ---
# ... (preprocess_char_data, load_processed_data, create_batches unverändert) ...
def preprocess_char_data(input_path: str, output_path: str, seq_len: int, val_split: float = 0.1):
    """
    Liest eine Textdatei, erstellt ein Charakter-Vokabular, erzeugt Input/Target-Paare
    für Next-Character-Prediction und speichert alles in einer .npz-Datei.
    """
    if os.path.exists(output_path):
        print(f"[DataPrep] Verarbeitete Datendatei '{output_path}' existiert bereits. Überspringe Vorverarbeitung.")
        return

    print(f"[DataPrep] Starte Vorverarbeitung für '{input_path}'...")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input-Textdatei nicht gefunden: {input_path}")

    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"[DataPrep] Text geladen ({len(text)} Zeichen). Erstelle Vokabular...")
    # Vokabular erstellen
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_id = {ch: i for i, ch in enumerate(chars)}
    id_to_char = {i: ch for i, ch in enumerate(chars)}
    print(f"[DataPrep] Vokabulargröße: {vocab_size}")

    # Text in IDs umwandeln
    data_ids = np.array([char_to_id[ch] for ch in text], dtype=INT_TYPE)
    print(f"[DataPrep] Text in {len(data_ids)} IDs umgewandelt.")

    # Input/Target-Paare erstellen
    num_sequences = len(data_ids) - seq_len
    if num_sequences <= 0:
        raise ValueError(f"Text ist zu kurz ({len(data_ids)} IDs) für die Sequenzlänge {seq_len}.")

    inputs = np.zeros((num_sequences, seq_len), dtype=INT_TYPE)
    targets = np.zeros((num_sequences, seq_len), dtype=INT_TYPE)

    print(f"[DataPrep] Erstelle {num_sequences} Input/Target-Sequenzpaare...")
    for i in range(num_sequences):
        inputs[i] = data_ids[i : i + seq_len]
        targets[i] = data_ids[i + 1 : i + 1 + seq_len] # Next character prediction

    # Aufteilen in Training/Validierung
    val_idx = int(len(inputs) * (1 - val_split))
    train_inputs = inputs[:val_idx]
    train_targets = targets[:val_idx]
    valid_inputs = inputs[val_idx:]
    valid_targets = targets[val_idx:]

    print(f"[DataPrep] Aufgeteilt: {len(train_inputs)} Trainings-, {len(valid_inputs)} Validierungssequenzen.")

    # Speichern
    # Pickle kann direkt zum Speichern von Dicts verwendet werden
    vocab_data = {'char_to_id': char_to_id, 'id_to_char': id_to_char}
    with open(output_path + "_vocab.pkl", 'wb') as f:
        pickle.dump(vocab_data, f)

    np.savez(output_path,
             train_inputs=train_inputs,
             train_targets=train_targets,
             valid_inputs=valid_inputs,
             valid_targets=valid_targets)
    print(f"[DataPrep] Verarbeitete Daten und Vokabular gespeichert in '{output_path}' und '{output_path}_vocab.pkl'.")

def load_processed_data(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, dict, dict]:
    """ Lädt die vorverarbeiteten Integer-Daten und das Vokabular. """
    vocab_path = filepath + "_vocab.pkl"
    if not os.path.exists(filepath) or not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Verarbeitete Datendatei '{filepath}' oder Vokabulardatei '{vocab_path}' nicht gefunden.")

    print(f"[DataLoader] Lade verarbeitete Daten aus '{filepath}'...")
    data = np.load(filepath)
    train_inputs = data['train_inputs'].astype(INT_TYPE)
    train_targets = data['train_targets'].astype(INT_TYPE)
    valid_inputs = data['valid_inputs'].astype(INT_TYPE)
    valid_targets = data['valid_targets'].astype(INT_TYPE)

    print(f"[DataLoader] Lade Vokabular aus '{vocab_path}'...")
    with open(vocab_path, 'rb') as f:
        vocab_data = pickle.load(f)
    char_to_id = vocab_data['char_to_id']
    id_to_char = vocab_data['id_to_char']
    vocab_size = len(char_to_id)

    # Form überprüfen (sollte jetzt 2D sein)
    if train_inputs.ndim != 2 or train_targets.ndim != 2 or valid_inputs.ndim != 2 or valid_targets.ndim != 2:
        raise ValueError(f"Geladene Daten haben nicht die erwarteten 2 Dimensionen.")
    if train_inputs.shape != train_targets.shape or valid_inputs.shape != valid_targets.shape:
         raise ValueError(f"Input- und Target-Shapes in geladenen Daten stimmen nicht überein.")

    print(f"[DataLoader] Verarbeitete Daten geladen:")
    print(f"  Train Inputs Shape: {train_inputs.shape}, Typ: {train_inputs.dtype}")
    print(f"  Train Targets Shape: {train_targets.shape}, Typ: {train_targets.dtype}")
    print(f"  Valid Inputs Shape: {valid_inputs.shape}, Typ: {valid_inputs.dtype}")
    print(f"  Valid Targets Shape: {valid_targets.shape}, Typ: {valid_targets.dtype}")
    print(f"  Vokabulargröße: {vocab_size}")

    return train_inputs, train_targets, valid_inputs, valid_targets, vocab_size, char_to_id, id_to_char

def create_batches(inputs: np.ndarray, targets: np.ndarray, batch_size: int, seq_len: int, shuffle: bool = True) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """ Erzeugt Batches (2D Integer) und füllt den letzten mit Padding auf. """
    num_samples = inputs.shape[0]
    indices = np.arange(num_samples)
    if shuffle: np.random.shuffle(indices)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        current_batch_size = len(batch_indices)

        # Direkte Indizierung für 2D-Arrays
        batch_inputs = inputs[batch_indices]
        batch_targets = targets[batch_indices]

        # Padding, falls der letzte Batch kleiner ist
        if current_batch_size < batch_size:
            num_padding = batch_size - current_batch_size
            # Padding für 2D-Arrays
            input_padding_shape = (num_padding, seq_len)
            # Input mit PAD_INDEX padden (da 0 eine gültige ID sein kann)
            input_padding = np.full(input_padding_shape, PAD_INDEX, dtype=inputs.dtype)
            batch_inputs = np.concatenate([batch_inputs, input_padding], axis=0)

            target_padding_shape = (num_padding, seq_len)
            target_padding = np.full(target_padding_shape, PAD_INDEX, dtype=targets.dtype)
            batch_targets = np.concatenate([batch_targets, target_padding], axis=0)

        yield batch_inputs, batch_targets


# --- NEU: Embedding Layer Klasse ---
# ... (unverändert, nur Prints reduziert) ...
class EmbeddingLayer:
    def __init__(self, vocab_size: int, embedding_dim: int, gpu_index: int = GPU_INDEX):
        self.V = vocab_size
        self.E = embedding_dim
        self.gpu_index = gpu_index
        self._is_freed = False
        itemsize_fp = FP_TYPE().itemsize
        itemsize_adam = np.float32().itemsize

        # Embedding Matrix und Gradient/Adam States
        self.W_emb = GPUTensor(self.V * self.E * itemsize_fp, gpu_index, name="EmbeddingW")
        self.dW_emb = GPUTensor(self.V * self.E * itemsize_fp, gpu_index, name="dEmbeddingW", zero_init=True)
        self.m_W_emb = GPUTensor(self.V * self.E * itemsize_adam, gpu_index, name="m_EmbeddingW", zero_init=True)
        self.v_W_emb = GPUTensor(self.V * self.E * itemsize_adam, gpu_index, name="v_EmbeddingW", zero_init=True)

        # Output-Tensor wird von MyModel verwaltet
        self.output: Optional[GPUTensor] = None
        self.input_indices_handle_cache: Optional[GPU_BUFFER_HANDLE] = None
        self.current_B = 0 # Wird von MyModel gesetzt
        self.current_S = 0 # Wird von MyModel gesetzt

        # Minimalerer Print
        # print(f"[EmbeddingLayer] Initialisiert: VocabSize={self.V}, EmbeddingDim={self.E}")

    def _check_freed(self):
        if self._is_freed: raise RuntimeError("Operation on freed EmbeddingLayer.")

    def set_current_batch_shape(self, b: int, s: int):
        self.current_B = b
        self.current_S = s

    def forward(self, input_indices_gpu: GPUTensor, output_buffer_gpu: GPUTensor) -> GPUTensor:
        self._check_freed()
        self.input_indices_handle_cache = input_indices_gpu.handle
        self.output = output_buffer_gpu # Verwende den vom Model bereitgestellten Buffer

        if self.output.handle is None: raise RuntimeError("Output buffer handle in EmbeddingLayer forward is None.")
        if self.input_indices_handle_cache is None: raise RuntimeError("Input indices handle cache in EmbeddingLayer forward is None.")
        if self.W_emb.handle is None: raise RuntimeError("Embedding weight handle in EmbeddingLayer forward is None.")

        if c_driver.execute_embedding_lookup_gpu(
            self.gpu_index,
            self.input_indices_handle_cache,
            self.W_emb.handle,
            self.output.handle,
            self.current_B,
            self.current_S,
            self.E, # Embedding Dimension (d in C)
            self.V  # Vocab Size (v in C)
        ) == 0:
            raise RuntimeError("Embedding Lookup fehlgeschlagen.")
        return self.output

    def backward(self, d_output: GPUTensor):
        """ d_output ist der Gradient, der von der nächsten Schicht kommt (shape B*S x E) """
        self._check_freed()
        if self.input_indices_handle_cache is None: raise RuntimeError("Cannot perform backward pass on EmbeddingLayer without cached input indices.")
        if d_output.handle is None or self.dW_emb.handle is None: raise RuntimeError("Handles for d_output or dW_emb are None in EmbeddingLayer backward.")

        if c_driver.execute_embedding_backward_gpu(
            self.gpu_index,
            d_output.handle,           # d_o
            self.input_indices_handle_cache, # idx
            self.dW_emb.handle,        # d_w (Ziel für Gradienten)
            self.current_B,            # b
            self.current_S,            # s
            self.E,                    # d (embedding dim)
            self.V                     # v (vocab size)
        ) == 0:
            raise RuntimeError("Embedding Backward (non-atomic) fehlgeschlagen.")
        return None # Gibt keinen Gradienten weiter

    def clip_gradients(self, clip_value: float):
         self._check_freed(); grad_W_host = None
         try:
             if clip_value <= 0 or self.dW_emb.size == 0: return
             grad_W_host = np.zeros((self.V, self.E), dtype=FP_TYPE)
             self.dW_emb.read(grad_W_host)
             norm_W = np.linalg.norm(grad_W_host)
             scale_W = clip_value / (norm_W + 1e-6) if norm_W > clip_value else 1.0
             if scale_W < 1.0: grad_W_host *= scale_W; self.dW_emb.write(grad_W_host)
         finally:
             if grad_W_host is not None: del grad_W_host

    def update(self, t: int, lr: float, beta1: float = 0.9, beta2: float = 0.999, weight_decay: float = 0.0):
        self._check_freed(); num_elements_W = self.V * self.E
        if num_elements_W > 0:
             if c_driver.execute_adam_update_on_gpu(
                 self.gpu_index, self.W_emb.handle, self.dW_emb.handle, self.m_W_emb.handle, self.v_W_emb.handle,
                 num_elements_W, t, lr, beta1, beta2, ADAM_EPS, weight_decay # WD wird hier übergeben, aber unten beim Aufruf mit 0 gesetzt
             ) == 0: raise RuntimeError("Adam Update für Embedding W fehlgeschlagen.")

    def get_state(self) -> Dict[str, np.ndarray]:
        self._check_freed(); state = {}
        state['W_emb'] = np.zeros((self.V, self.E), dtype=FP_TYPE)
        if self.W_emb.size > 0: self.W_emb.read(state['W_emb'])
        adam_shape = (self.V, self.E);
        state['m_W_emb'] = np.zeros(adam_shape, dtype=np.float32); state['v_W_emb'] = np.zeros(adam_shape, dtype=np.float32)
        if self.m_W_emb.size > 0: self.m_W_emb.read(state['m_W_emb'])
        if self.v_W_emb.size > 0: self.v_W_emb.read(state['v_W_emb'])
        return state

    def set_state(self, state: Dict[str, np.ndarray]):
        self._check_freed()
        try:
            if 'W_emb' in state and self.W_emb.size > 0: self.W_emb.write(state['W_emb'].astype(FP_TYPE))
            if 'm_W_emb' in state and self.m_W_emb.size > 0: self.m_W_emb.write(state['m_W_emb'].astype(np.float32))
            if 'v_W_emb' in state and self.v_W_emb.size > 0: self.v_W_emb.write(state['v_W_emb'].astype(np.float32))
            # Minimalerer Print
            # if DEBUG_PRINTS: print(f"[EmbeddingLayer] Zustand geladen für {self.V} x {self.E}.")
        except KeyError as e: print(f"[EmbeddingLayer] WARNUNG: Fehlender Key im Checkpoint-State: {e}")
        except Exception as e: print(f"[EmbeddingLayer] FEHLER beim Setzen des Zustands: {e}")

    def initialize_weights(self):
         limit = np.sqrt(1.0 / self.E); weights = np.random.uniform(-limit, limit, (self.V, self.E)).astype(FP_TYPE)
         self.W_emb.write(weights)
         print(f"[EmbeddingLayer] Weights initialized ({self.V}x{self.E})")

    def free(self):
        if not self._is_freed:
            if DEBUG_PRINTS: print(f"[EmbeddingLayer] Freeing resources...")
            tensors_to_free = [self.W_emb, self.dW_emb, self.m_W_emb, self.v_W_emb]
            for tensor in tensors_to_free:
                if tensor: tensor.free()
            self._is_freed = True

    def __del__(self):
        if not getattr(self, '_is_freed', True): self.free()
    def train(self): pass
    def eval(self): pass

# --- Layer Klassen (Linear, BioInspired, CrossEntropyLoss) ---
# ... (LinearLayer, BioInspiredAssociativeLayer, CrossEntropyLoss unverändert, nur Prints reduziert) ...
class LinearLayer:
    def __init__(self, batch_size_seq_len_flat: int, input_dim: int, output_dim: int, gpu_index: int = GPU_INDEX):
        self.M_flat = batch_size_seq_len_flat; self.E_in = input_dim; self.E_out = output_dim
        self.gpu_index = gpu_index; self._is_freed = False
        itemsize_fp = FP_TYPE().itemsize; itemsize_adam = np.float32().itemsize
        self.W = GPUTensor(self.E_in * self.E_out * itemsize_fp, gpu_index, name="LinearW"); self.b = GPUTensor(self.E_out * itemsize_fp, gpu_index, name="LinearB", zero_init=True)
        self.dW = GPUTensor(self.E_in * self.E_out * itemsize_fp, gpu_index, name="dLinearW", zero_init=True); self.db = GPUTensor(self.E_out * itemsize_fp, gpu_index, name="dLinearB", zero_init=True)
        self.m_W = GPUTensor(self.E_in * self.E_out * itemsize_adam, gpu_index, name="m_LinearW", zero_init=True); self.v_W = GPUTensor(self.E_in * self.E_out * itemsize_adam, gpu_index, name="v_LinearW", zero_init=True)
        self.m_b = GPUTensor(self.E_out * itemsize_adam, gpu_index, name="m_LinearB", zero_init=True); self.v_b = GPUTensor(self.E_out * itemsize_adam, gpu_index, name="v_LinearB", zero_init=True)
        self.output = GPUTensor(self.M_flat * self.E_out * itemsize_fp, gpu_index, name="LinearOutput"); self.d_input = GPUTensor(self.M_flat * self.E_in * itemsize_fp, gpu_index, name="dLinearInput")
        self.input_handle_cache = None; self.current_M_flat = self.M_flat
        # Minimalerer Print
        # print(f"[LinearLayer] Initialisiert: In={self.E_in}, Out={self.E_out}, Max M_flat={self.M_flat}")
    def _check_freed(self):
        if self._is_freed: raise RuntimeError("Operation on freed LinearLayer.")
    def set_current_batch_m_flat(self, m_flat: int):
         if m_flat > self.M_flat: raise ValueError(f"Aktuelles M_flat ({m_flat}) > Max M_flat ({self.M_flat})")
         self.current_M_flat = m_flat
    def forward(self, input_tensor: GPUTensor) -> GPUTensor:
        self._check_freed(); self.input_handle_cache = input_tensor.handle
        if c_driver.execute_matmul_on_gpu(self.gpu_index, self.input_handle_cache, self.W.handle, self.output.handle, 1, self.current_M_flat, self.E_out, self.E_in) == 0: raise RuntimeError("Linear Forward MatMul fehlgeschlagen.")
        if c_driver.execute_add_bias_on_gpu(self.gpu_index, self.output.handle, self.b.handle, self.current_M_flat, self.E_out) == 0: raise RuntimeError("Linear Forward Bias Add fehlgeschlagen.")
        return self.output
    def backward(self, d_output: GPUTensor) -> GPUTensor:
        self._check_freed(); assert self.input_handle_cache is not None
        if c_driver.execute_reduce_sum_gpu(self.gpu_index, d_output.handle, self.db.handle, 1, self.current_M_flat, self.E_out) == 0: raise RuntimeError("Linear Backward Bias (ReduceSum) fehlgeschlagen.")
        if c_driver.execute_matmul_backward_on_gpu(self.gpu_index, self.input_handle_cache, self.W.handle, d_output.handle, self.d_input.handle, self.dW.handle, 1, self.current_M_flat, self.E_out, self.E_in) == 0: raise RuntimeError("Linear Backward MatMul fehlgeschlagen.")
        return self.d_input
    def clip_gradients(self, clip_value: float):
        self._check_freed(); grad_W_host = grad_b_host = None
        try:
            if clip_value <= 0 or (self.dW.size == 0 and self.db.size == 0): return
            if self.dW.size > 0:
                grad_W_host = np.zeros((self.E_in, self.E_out), dtype=FP_TYPE); self.dW.read(grad_W_host)
                norm_W = np.linalg.norm(grad_W_host); scale_W = clip_value / (norm_W + 1e-6) if norm_W > clip_value else 1.0
                if scale_W < 1.0: grad_W_host *= scale_W; self.dW.write(grad_W_host)
            if self.db.size > 0:
                grad_b_host = np.zeros(self.E_out, dtype=FP_TYPE); self.db.read(grad_b_host)
                norm_b = np.linalg.norm(grad_b_host); scale_b = clip_value / (norm_b + 1e-6) if norm_b > clip_value else 1.0
                if scale_b < 1.0: grad_b_host *= scale_b; self.db.write(grad_b_host)
        finally:
             if grad_W_host is not None: del grad_W_host
             if grad_b_host is not None: del grad_b_host
    def update(self, t: int, lr: float, beta1: float = 0.9, beta2: float = 0.999, weight_decay: float = 0.0):
        self._check_freed(); num_elements_W = self.E_in * self.E_out; num_elements_b = self.E_out
        if num_elements_W > 0:
            if c_driver.execute_adam_update_on_gpu(self.gpu_index, self.W.handle, self.dW.handle, self.m_W.handle, self.v_W.handle, num_elements_W, t, lr, beta1, beta2, ADAM_EPS, weight_decay) == 0: raise RuntimeError("Adam Update für Linear W fehlgeschlagen.")
        if num_elements_b > 0: # Kein WD für Bias
            if c_driver.execute_adam_update_on_gpu(self.gpu_index, self.b.handle, self.db.handle, self.m_b.handle, self.v_b.handle, num_elements_b, t, lr, beta1, beta2, ADAM_EPS, 0.0) == 0: raise RuntimeError("Adam Update für Linear b fehlgeschlagen.")
    def get_state(self) -> Dict[str, np.ndarray]:
        self._check_freed(); state = {};
        state['W'] = np.zeros((self.E_in, self.E_out), dtype=FP_TYPE); state['b'] = np.zeros(self.E_out, dtype=FP_TYPE)
        if self.W.size > 0: self.W.read(state['W']);
        if self.b.size > 0: self.b.read(state['b'])
        adam_shape_W = (self.E_in, self.E_out); adam_shape_b = (self.E_out,)
        state['m_W'] = np.zeros(adam_shape_W, dtype=np.float32); state['v_W'] = np.zeros(adam_shape_W, dtype=np.float32)
        state['m_b'] = np.zeros(adam_shape_b, dtype=np.float32); state['v_b'] = np.zeros(adam_shape_b, dtype=np.float32)
        if self.m_W.size > 0: self.m_W.read(state['m_W']);
        if self.v_W.size > 0: self.v_W.read(state['v_W'])
        if self.m_b.size > 0: self.m_b.read(state['m_b']);
        if self.v_b.size > 0: self.v_b.read(state['v_b'])
        return state
    def set_state(self, state: Dict[str, np.ndarray]):
        self._check_freed();
        try:
            if 'W' in state and self.W.size > 0: self.W.write(state['W'].astype(FP_TYPE))
            if 'b' in state and self.b.size > 0: self.b.write(state['b'].astype(FP_TYPE))
            if 'm_W' in state and self.m_W.size > 0: self.m_W.write(state['m_W'].astype(np.float32))
            if 'v_W' in state and self.v_W.size > 0: self.v_W.write(state['v_W'].astype(np.float32))
            if 'm_b' in state and self.m_b.size > 0: self.m_b.write(state['m_b'].astype(np.float32))
            if 'v_b' in state and self.v_b.size > 0: self.v_b.write(state['v_b'].astype(np.float32))
        except KeyError as e: print(f"[LinearLayer] WARNUNG: Fehlender Key im Checkpoint-State: {e}")
        except Exception as e: print(f"[LinearLayer] FEHLER beim Setzen des Zustands: {e}")
    def free(self):
        if not self._is_freed:
            if DEBUG_PRINTS: print(f"[LinearLayer] Freeing resources...")
            tensors_to_free = [self.W, self.b, self.dW, self.db, self.m_W, self.v_W, self.m_b, self.v_b, self.output, self.d_input]
            for tensor in tensors_to_free:
                if tensor: tensor.free()
            self._is_freed = True
    def __del__(self):
        if not getattr(self, '_is_freed', True): self.free()
    def train(self): pass
    def eval(self): pass

class BioInspiredAssociativeLayer:
    def __init__(self, batch_size: int, seq_len: int, embedding_dim: int, hidden_dim: int, num_prototypes: int,
                 hebbian_lr: float = 0.001, spike_threshold: float = 0.5, prototype_lr: float = 0.01, gpu_index: int = GPU_INDEX):
        self.B = batch_size; self.S = seq_len; self.E_in = embedding_dim
        self.E_hidden = hidden_dim; self.T = num_prototypes
        self.hebbian_lr = hebbian_lr; self.spike_threshold = spike_threshold; self.prototype_lr = prototype_lr
        self.gpu_index = gpu_index; self._is_freed = False
        itemsize_fp = FP_TYPE().itemsize; itemsize_adam = np.float32().itemsize; itemsize_int = INT_TYPE().itemsize
        self.max_M_flat = self.B * self.S; self.current_B = self.B; self.current_M_flat = self.max_M_flat
        self.W1 = GPUTensor(self.E_in * self.E_hidden * itemsize_fp, gpu_index, name="BioW1"); self.b1 = GPUTensor(self.E_hidden * itemsize_fp, gpu_index, name="BioB1", zero_init=True)
        self.dW1 = GPUTensor(self.E_in * self.E_hidden * itemsize_fp, gpu_index, name="dBioW1", zero_init=True); self.db1 = GPUTensor(self.E_hidden * itemsize_fp, gpu_index, name="dBioB1", zero_init=True)
        self.m_W1 = GPUTensor(self.E_in * self.E_hidden * itemsize_adam, gpu_index, name="m_BioW1", zero_init=True); self.v_W1 = GPUTensor(self.E_in * self.E_hidden * itemsize_adam, gpu_index, name="v_BioW1", zero_init=True)
        self.m_b1 = GPUTensor(self.E_hidden * itemsize_adam, gpu_index, name="m_BioB1", zero_init=True); self.v_b1 = GPUTensor(self.E_hidden * itemsize_adam, gpu_index, name="v_BioB1", zero_init=True)
        self.W_hebb = GPUTensor(self.E_hidden * self.E_hidden * itemsize_fp, gpu_index, name="BioW_hebb", zero_init=True); self.prototypes = GPUTensor(self.T * self.E_hidden * itemsize_fp, gpu_index, name="BioPrototypes") if self.T > 0 else None # Handle T=0
        self.num_elements_hidden_flat_max = self.max_M_flat * self.E_hidden; self.num_elements_tokens_max = self.max_M_flat; self.num_elements_input_flat_max = self.max_M_flat * self.E_in
        self.input_handle_cache = None; self.pre_gelu_activations = GPUTensor(self.num_elements_hidden_flat_max * itemsize_fp, gpu_index, name="BioPreGELUActs")
        self.hidden_activations = GPUTensor(self.num_elements_hidden_flat_max * itemsize_fp, gpu_index, name="BioHiddenActs"); self.d_pre_gelu = GPUTensor(self.num_elements_hidden_flat_max * itemsize_fp, gpu_index, name="dBioPreGELU")
        self.d_input = GPUTensor(self.num_elements_input_flat_max * itemsize_fp, gpu_index, name="dBioInput"); self.spikes = GPUTensor(self.num_elements_hidden_flat_max * itemsize_fp, gpu_index, name="BioSpikes")
        self.token_indices = GPUTensor(self.num_elements_tokens_max * itemsize_int, gpu_index, name="BioTokenIndices") if self.T > 0 else None # Handle T=0
        self.proto_sums_gpu = None; self.proto_counts_gpu = None
        if self.T > 0 and USE_GPU_PROTOTYPE_UPDATE and CAN_USE_GPU_PROTO_UPDATE:
             self.proto_sums_gpu = GPUTensor(self.T * self.E_hidden * itemsize_fp, gpu_index, name="ProtoSumsGPU", zero_init=True)
             self.proto_counts_gpu = GPUTensor(self.T * itemsize_int, gpu_index, name="ProtoCountsGPU", zero_init=True)
        # Minimalerer Print
        # print(f"[BioLayer] Initialisiert: Max B={self.B}, S={self.S}, E_in(Embed)={self.E_in}, E_hidden={self.E_hidden}, T={self.T}")
        # print(f"[BioLayer] Max Flattened M dimension: {self.max_M_flat}")
        # print(f"[BioLayer] GPU Prototype Update: {'Aktiviert' if self.T > 0 and USE_GPU_PROTOTYPE_UPDATE and CAN_USE_GPU_PROTO_UPDATE else 'Deaktiviert'}")
    def _check_freed(self):
        if self._is_freed: raise RuntimeError("Operation on freed BioInspiredAssociativeLayer.")
    def set_current_batch_shape(self, b: int, s: int):
         if b > self.B or s != self.S: raise ValueError(f"Aktuelles B ({b}) > Max B ({self.B}) oder S ({s}) != fixed S ({self.S})")
         self.current_B = b; self.current_M_flat = b * s
    def forward(self, input_tensor: GPUTensor) -> GPUTensor:
        self._check_freed(); self.input_handle_cache = input_tensor.handle; current_num_elements_hidden = self.current_M_flat * self.E_hidden
        if c_driver.execute_matmul_on_gpu(self.gpu_index, self.input_handle_cache, self.W1.handle, self.pre_gelu_activations.handle, 1, self.current_M_flat, self.E_hidden, self.E_in) == 0: raise RuntimeError("BioLayer Forward MatMul fehlgeschlagen.")
        if c_driver.execute_add_bias_on_gpu(self.gpu_index, self.pre_gelu_activations.handle, self.b1.handle, self.current_M_flat, self.E_hidden) == 0: raise RuntimeError("BioLayer Forward Bias Add fehlgeschlagen.")
        if c_driver.execute_gelu_on_gpu(self.gpu_index, self.pre_gelu_activations.handle, self.hidden_activations.handle, current_num_elements_hidden) == 0: raise RuntimeError("BioLayer Forward GELU fehlgeschlagen.")
        if self.T > 0 and self.prototypes is not None and self.token_indices is not None:
             if c_driver.execute_dynamic_token_assignment_gpu(self.gpu_index, self.hidden_activations.handle, self.prototypes.handle, self.token_indices.handle, self.current_B, self.S, self.E_hidden, self.T) == 0: raise RuntimeError("BioLayer Dynamic token assignment fehlgeschlagen.")
        if c_driver.execute_threshold_spike_on_gpu(self.gpu_index, self.hidden_activations.handle, self.spikes.handle, self.spike_threshold, current_num_elements_hidden) == 0: raise RuntimeError("BioLayer Threshold spike fehlgeschlagen.")
        return self.hidden_activations
    def backward(self, d_output: GPUTensor) -> GPUTensor:
        self._check_freed(); assert self.input_handle_cache is not None; current_num_elements_hidden = self.current_M_flat * self.E_hidden
        if c_driver.execute_gelu_backward_on_gpu(self.gpu_index, self.pre_gelu_activations.handle, d_output.handle, self.d_pre_gelu.handle, current_num_elements_hidden) == 0: raise RuntimeError("BioLayer Backward GELU fehlgeschlagen.")
        if c_driver.execute_reduce_sum_gpu(self.gpu_index, self.d_pre_gelu.handle, self.db1.handle, 1, self.current_M_flat, self.E_hidden) == 0: raise RuntimeError("BioLayer Backward Bias (ReduceSum) fehlgeschlagen.")
        if c_driver.execute_matmul_backward_on_gpu(self.gpu_index, self.input_handle_cache, self.W1.handle, self.d_pre_gelu.handle, self.d_input.handle, self.dW1.handle, 1, self.current_M_flat, self.E_hidden, self.E_in) == 0: raise RuntimeError("BioLayer Backward MatMul fehlgeschlagen.")
        return self.d_input
    def clip_gradients(self, clip_value: float):
        self._check_freed(); grad_W1_host = grad_b1_host = None
        try:
            if clip_value <= 0 or (self.dW1.size == 0 and self.db1.size == 0) : return
            if self.dW1.size > 0:
                grad_W1_host = np.zeros((self.E_in, self.E_hidden), dtype=FP_TYPE); self.dW1.read(grad_W1_host)
                norm_W1 = np.linalg.norm(grad_W1_host); scale_W1 = clip_value / (norm_W1 + 1e-6) if norm_W1 > clip_value else 1.0
                if scale_W1 < 1.0: grad_W1_host *= scale_W1; self.dW1.write(grad_W1_host);
            if self.db1.size > 0:
                grad_b1_host = np.zeros(self.E_hidden, dtype=FP_TYPE); self.db1.read(grad_b1_host)
                norm_b1 = np.linalg.norm(grad_b1_host); scale_b1 = clip_value / (norm_b1 + 1e-6) if norm_b1 > clip_value else 1.0
                if scale_b1 < 1.0: grad_b1_host *= scale_b1; self.db1.write(grad_b1_host);
        finally:
             if grad_W1_host is not None: del grad_W1_host
             if grad_b1_host is not None: del grad_b1_host
    def update(self, t: int, lr: float, beta1: float = 0.9, beta2: float = 0.999, weight_decay: float = 0.0):
        self._check_freed(); assert t > 0;
        num_elements_W1 = self.E_in * self.E_hidden; num_elements_b1 = self.E_hidden
        if num_elements_W1 > 0:
            if c_driver.execute_adam_update_on_gpu(self.gpu_index, self.W1.handle, self.dW1.handle, self.m_W1.handle, self.v_W1.handle, num_elements_W1, t, lr, beta1, beta2, ADAM_EPS, weight_decay) == 0: raise RuntimeError("Adam Update für BioW1 fehlgeschlagen.")
        if num_elements_b1 > 0:
            if c_driver.execute_adam_update_on_gpu(self.gpu_index, self.b1.handle, self.db1.handle, self.m_b1.handle, self.v_b1.handle, num_elements_b1, t, lr, beta1, beta2, ADAM_EPS, 0.0) == 0: raise RuntimeError("Adam Update für BioB1 fehlgeschlagen.")
    def hebbian_learn(self, pre_synaptic_activations: GPUTensor, post_synaptic_activations: GPUTensor):
        self._check_freed(); assert pre_synaptic_activations.handle is not None and post_synaptic_activations.handle is not None;
        if self.W_hebb.size == 0: return
        if c_driver.execute_hebbian_update_on_gpu(self.gpu_index, pre_synaptic_activations.handle, post_synaptic_activations.handle, self.W_hebb.handle, self.hebbian_lr, self.current_B, self.S, self.E_hidden, self.E_hidden) == 0: raise RuntimeError("execute_hebbian_update_on_gpu fehlgeschlagen.")
    def update_prototypes(self):
        self._check_freed();
        if self.T == 0: return
        if USE_GPU_PROTOTYPE_UPDATE and CAN_USE_GPU_PROTO_UPDATE: self._update_prototypes_gpu()
        else:
            if USE_GPU_PROTOTYPE_UPDATE and not CAN_USE_GPU_PROTO_UPDATE and DEBUG_PRINTS: print("[BioLayer] WARNUNG: GPU Prototyp-Update nicht verfügbar. Nutze Host-Fallback.")
            self._update_prototypes_host()
    def _update_prototypes_gpu(self):
        assert self.proto_sums_gpu is not None and self.proto_counts_gpu is not None and self.prototypes is not None and self.token_indices is not None
        self.proto_sums_gpu._zero_initialize(); self.proto_counts_gpu._zero_initialize()
        if c_driver.execute_proto_segmented_sum_gpu(self.gpu_index, self.hidden_activations.handle, self.token_indices.handle, self.proto_sums_gpu.handle, self.proto_counts_gpu.handle, self.current_M_flat, self.E_hidden, self.T) == 0: raise RuntimeError("GPU Prototype Segmented Sum fehlgeschlagen.")
        if c_driver.execute_proto_update_step_gpu(self.gpu_index, self.prototypes.handle, self.proto_sums_gpu.handle, self.proto_counts_gpu.handle, self.prototype_lr, self.E_hidden, self.T) == 0: raise RuntimeError("GPU Prototype Update Step fehlgeschlagen.")
    def _update_prototypes_host(self):
        assert self.token_indices is not None and self.prototypes is not None
        host_indices = np.zeros(self.current_M_flat, dtype=INT_TYPE); host_activations_flat = np.zeros(self.current_M_flat * self.E_hidden, dtype=FP_TYPE)
        self.token_indices.read(host_indices, offset_bytes=0); self.hidden_activations.read(host_activations_flat, offset_bytes=0)
        host_indices_rs = host_indices.reshape(self.current_B, self.S); host_activations_rs = host_activations_flat.reshape(self.current_B, self.S, self.E_hidden)
        current_prototypes_host = np.zeros((self.T, self.E_hidden), dtype=FP_TYPE); self.prototypes.read(current_prototypes_host)
        updated_prototypes_host = current_prototypes_host.copy(); update_counts = np.zeros(self.T, dtype=int)
        for p_idx in range(self.T):
            assigned_mask = (host_indices_rs == p_idx); num_assigned = np.sum(assigned_mask); update_counts[p_idx] = num_assigned
            if num_assigned > 0:
                assigned_activations = host_activations_rs[assigned_mask]; mean_activation = np.mean(assigned_activations, axis=0)
                updated_prototypes_host[p_idx] = ((1.0 - self.prototype_lr) * current_prototypes_host[p_idx] + self.prototype_lr * mean_activation)
        self.prototypes.write(updated_prototypes_host)
        del host_indices, host_activations_flat, host_indices_rs, host_activations_rs, current_prototypes_host, updated_prototypes_host, update_counts
    def get_state(self) -> Dict[str, np.ndarray]:
        self._check_freed(); state = {};
        state['W1'] = np.zeros((self.E_in, self.E_hidden), dtype=FP_TYPE); state['b1'] = np.zeros(self.E_hidden, dtype=FP_TYPE)
        if self.W1.size > 0: self.W1.read(state['W1']);
        if self.b1.size > 0: self.b1.read(state['b1'])
        adam_shape_W1 = (self.E_in, self.E_hidden); adam_shape_b1 = (self.E_hidden,)
        state['m_W1'] = np.zeros(adam_shape_W1, dtype=np.float32); state['v_W1'] = np.zeros(adam_shape_W1, dtype=np.float32)
        state['m_b1'] = np.zeros(adam_shape_b1, dtype=np.float32); state['v_b1'] = np.zeros(adam_shape_b1, dtype=np.float32)
        if self.m_W1.size > 0: self.m_W1.read(state['m_W1']);
        if self.v_W1.size > 0: self.v_W1.read(state['v_W1'])
        if self.m_b1.size > 0: self.m_b1.read(state['m_b1']);
        if self.v_b1.size > 0: self.v_b1.read(state['v_b1'])
        state['W_hebb'] = self.get_hebbian_weights();
        if self.T > 0: state['prototypes'] = self.get_prototypes()
        return state
    def set_state(self, state: Dict[str, np.ndarray]):
        self._check_freed();
        try:
            if 'W1' in state and self.W1.size > 0: self.W1.write(state['W1'].astype(FP_TYPE))
            if 'b1' in state and self.b1.size > 0: self.b1.write(state['b1'].astype(FP_TYPE))
            if 'm_W1' in state and self.m_W1.size > 0: self.m_W1.write(state['m_W1'].astype(np.float32))
            if 'v_W1' in state and self.v_W1.size > 0: self.v_W1.write(state['v_W1'].astype(np.float32))
            if 'm_b1' in state and self.m_b1.size > 0: self.m_b1.write(state['m_b1'].astype(np.float32))
            if 'v_b1' in state and self.v_b1.size > 0: self.v_b1.write(state['v_b1'].astype(np.float32))
            if 'W_hebb' in state and self.W_hebb.size > 0: self.W_hebb.write(state['W_hebb'].astype(FP_TYPE))
            if 'prototypes' in state and self.T > 0 and self.prototypes is not None: self.prototypes.write(state['prototypes'].astype(FP_TYPE))
        except KeyError as e: print(f"[BioLayer] WARNUNG: Fehlender Key im Checkpoint-State: {e}")
        except Exception as e: print(f"[BioLayer] FEHLER beim Setzen des Zustands: {e}")
    def get_dynamic_tokens(self) -> Optional[np.ndarray]:
        self._check_freed();
        if self.token_indices is None: return None
        host_indices = np.zeros(self.current_M_flat, dtype=INT_TYPE); self.token_indices.read(host_indices, offset_bytes=0); return host_indices.reshape(self.current_B, self.S)
    def get_spikes(self) -> np.ndarray:
        self._check_freed(); host_spikes_flat = np.zeros(self.current_M_flat * self.E_hidden, dtype=FP_TYPE); self.spikes.read(host_spikes_flat, offset_bytes=0); return host_spikes_flat.reshape(self.current_B, self.S, self.E_hidden)
    def get_prototypes(self) -> Optional[np.ndarray]:
        self._check_freed();
        if self.prototypes is None: return None
        host_prototypes = np.zeros((self.T, self.E_hidden), dtype=FP_TYPE);
        self.prototypes.read(host_prototypes); return host_prototypes
    def get_hebbian_weights(self) -> np.ndarray:
        self._check_freed(); host_W_hebb = np.zeros((self.E_hidden, self.E_hidden), dtype=FP_TYPE);
        if self.W_hebb.size > 0: self.W_hebb.read(host_W_hebb); return host_W_hebb
    def get_W1_grad(self) -> np.ndarray:
        self._check_freed(); grad = np.zeros((self.E_in, self.E_hidden), dtype=FP_TYPE); self.dW1.read(grad); return grad
    def get_b1_grad(self) -> np.ndarray:
        self._check_freed(); grad = np.zeros(self.E_hidden, dtype=FP_TYPE); self.db1.read(grad); return grad
    def free(self):
        if not self._is_freed:
            if DEBUG_PRINTS: print(f"[BioLayer] Freeing resources...")
            tensors_to_free = [
                self.W1, self.b1, self.dW1, self.db1, self.m_W1, self.v_W1, self.m_b1, self.v_b1,
                self.W_hebb, self.prototypes, self.pre_gelu_activations, self.hidden_activations,
                self.d_pre_gelu, self.d_input, self.spikes, self.token_indices,
                self.proto_sums_gpu, self.proto_counts_gpu
            ]
            for tensor in tensors_to_free:
                if tensor: tensor.free()
            self._is_freed = True
    def __del__(self):
        if not getattr(self, '_is_freed', True): self.free()
    def train(self): pass
    def eval(self): pass

class CrossEntropyLoss:
    def __init__(self, max_batch_size_seq_len_flat: int, vocab_size: int, gpu_index: int = GPU_INDEX):
        self.max_M_flat = max_batch_size_seq_len_flat
        self.V = vocab_size
        self.gpu_index = gpu_index
        self._is_freed = False
        itemsize_fp = FP_TYPE().itemsize
        self.log_probs = GPUTensor(self.max_M_flat * self.V * itemsize_fp, gpu_index, name="LogProbs")
        self.d_logits = GPUTensor(self.max_M_flat * self.V * itemsize_fp, gpu_index, name="dLogits", zero_init=True) # Zero init grads
        self.loss_per_sample = GPUTensor(self.max_M_flat * itemsize_fp, gpu_index, name="LossPerSample")
        # Minimalerer Print
        # print(f"[CrossEntropyLoss] Initialisiert: VocabSize={self.V}, Max M_flat={self.max_M_flat}")

    def _check_freed(self):
        if self._is_freed: raise RuntimeError("Operation on freed CrossEntropyLoss.")

    def forward(self, logits: GPUTensor, targets_gpu: GPUTensor, current_m_flat: int, targets_np: np.ndarray) -> float:
        """ Berechnet Loss UND Gradient d_logits. Gibt avg Loss zurück. """
        self._check_freed()
        if logits.handle is None or targets_gpu.handle is None: raise ValueError("Input handles for CrossEntropyLoss forward are None.")
        if current_m_flat > self.max_M_flat: raise ValueError(f"Aktuelles M_flat ({current_m_flat}) > Max M_flat ({self.max_M_flat})")

        if c_driver.execute_log_softmax_stable_gpu(self.gpu_index, logits.handle, self.log_probs.handle, current_m_flat, self.V) == 0:
             print("[FATAL] execute_log_softmax_stable_gpu returned 0. Check C error messages above.")
             raise RuntimeError("Loss Forward LogSoftmax fehlgeschlagen.")

        if c_driver.execute_cross_entropy_loss_grad_gpu(
            self.gpu_index, self.log_probs.handle, targets_gpu.handle,
            self.d_logits.handle, self.loss_per_sample.handle,
            current_m_flat, self.V
        ) == 0: raise RuntimeError("Loss Forward/Backward (CrossEntropy Kernel) fehlgeschlagen.")

        host_loss_per_sample = np.zeros(current_m_flat, dtype=FP_TYPE); self.loss_per_sample.read(host_loss_per_sample, offset_bytes=0)
        mean_loss = 0.0
        targets_np_flat = targets_np.flatten()[:current_m_flat]
        valid_mask = targets_np_flat != PAD_INDEX
        valid_losses = host_loss_per_sample[valid_mask]
        if np.any(valid_mask) and len(valid_losses) > 0:
             mean_loss = np.mean(valid_losses)

        del host_loss_per_sample, targets_np_flat, valid_mask, valid_losses
        return float(mean_loss)

    def free(self):
        if not self._is_freed:
            tensors_to_free = [self.log_probs, self.d_logits, self.loss_per_sample];
            for tensor in tensors_to_free:
                if tensor: tensor.free()
            self._is_freed = True

    def __del__(self):
        if not getattr(self, '_is_freed', True): self.free()


# MyModel Klasse
class MyModel:
    def __init__(self, batch_size: int, seq_len: int, embedding_dim: int, hidden_dim: int, vocab_size: int, num_prototypes: int, gpu_index: int = GPU_INDEX):
        self.B = batch_size; self.S = seq_len; self.M_flat = batch_size * seq_len;
        self.embedding_dim = embedding_dim; self.hidden_dim = hidden_dim; self.vocab_size = vocab_size;
        self.num_prototypes = num_prototypes # <<< NEU: Store num_prototypes
        self.gpu_index = gpu_index; self._is_freed = False
        itemsize_fp = FP_TYPE().itemsize

        print("[Model Init] Erstelle Layer...")
        self.embedding_layer = EmbeddingLayer(vocab_size, embedding_dim, gpu_index=gpu_index)
        # Pass NUM_TOKEN_PROTOTYPES hier korrekt
        self.bio_layer = BioInspiredAssociativeLayer(batch_size, seq_len, embedding_dim, hidden_dim, self.num_prototypes, gpu_index=gpu_index, hebbian_lr=HEBBIAN_LR, spike_threshold=SPIKE_THRESHOLD, prototype_lr=PROTOTYPE_LR)
        self.output_layer = LinearLayer(self.M_flat, hidden_dim, vocab_size, gpu_index=gpu_index)
        self.criterion = CrossEntropyLoss(self.M_flat, vocab_size, gpu_index=gpu_index)

        print("[Model Init] Erstelle Buffer...")
        # Buffer für Embedding-Output (wird von EmbeddingLayer genutzt)
        self.embedding_output_buffer = GPUTensor(self.M_flat * self.embedding_dim * itemsize_fp, gpu_index, name="EmbeddingOutputBuffer")
        print("[Model Init] Initialisierung abgeschlossen.")

    def _initialize_weights(self):
        if DEBUG_PRINTS: print("[Model] Initializing weights...")
        self.embedding_layer.initialize_weights()
        limit_bio = np.sqrt(6.0 / (self.bio_layer.E_in + self.bio_layer.E_hidden)); self.bio_layer.W1.write(np.random.uniform(-limit_bio, limit_bio, (self.bio_layer.E_in, self.bio_layer.E_hidden)).astype(FP_TYPE))
        # Nur initialisieren, wenn Prototypen existieren
        if self.bio_layer.T > 0 and self.bio_layer.prototypes is not None:
            prototypes_init = np.random.randn(self.bio_layer.T, self.bio_layer.E_hidden).astype(FP_TYPE) * 0.1; self.bio_layer.prototypes.write(prototypes_init)
        limit_out = np.sqrt(6.0 / (self.output_layer.E_in + self.output_layer.E_out)); self.output_layer.W.write(np.random.uniform(-limit_out, limit_out, (self.output_layer.E_in, self.output_layer.E_out)).astype(FP_TYPE))

    def set_current_batch_shape(self, b: int, s: int):
        if s != self.S: raise ValueError(f"Sequenzlänge ({s}) stimmt nicht mit Modellkonfiguration ({self.S}) überein.")
        current_m_flat = b * s
        self.embedding_layer.set_current_batch_shape(b, s)
        self.bio_layer.set_current_batch_shape(b, s)
        self.output_layer.set_current_batch_m_flat(current_m_flat)

    def forward(self, input_indices_gpu: GPUTensor) -> GPUTensor:
        """ Nimmt Integer-IDs als Input (GPUTensor) und gibt Logits aus (GPUTensor). """
        embedded_input = self.embedding_layer.forward(input_indices_gpu, self.embedding_output_buffer)
        bio_output = self.bio_layer.forward(embedded_input)
        logits = self.output_layer.forward(bio_output)
        return logits

    def compute_loss_and_backward(self, logits: GPUTensor, targets_gpu: GPUTensor, targets_np: np.ndarray) -> Tuple[float, None]:
        """ Berechnet Loss und führt Backward-Pass durch alle Schichten aus """
        current_m_flat = self.bio_layer.current_M_flat # Hole aktuelle Batch-Größe

        loss = self.criterion.forward(logits, targets_gpu, current_m_flat, targets_np=targets_np)
        d_logits = self.criterion.d_logits # Gradient aus dem Loss-Kernel holen

        d_bio_output = self.output_layer.backward(d_logits)
        d_embedded_input = self.bio_layer.backward(d_bio_output)
        self.embedding_layer.backward(d_embedded_input) # Gradient endet hier

        return loss, None

    def clip_all_gradients(self, clip_value: Optional[float]):
         if clip_value is not None and clip_value > 0:
             self.embedding_layer.clip_gradients(clip_value)
             self.bio_layer.clip_gradients(clip_value)
             self.output_layer.clip_gradients(clip_value)

    def update_trainable(self, t: int, lr: float, beta1: float = 0.9, beta2: float = 0.999, weight_decay: float = 0.0):
        # Kein WD für Embeddings
        self.embedding_layer.update(t, lr, beta1, beta2, 0.0)
        self.bio_layer.update(t, lr, beta1, beta2, weight_decay)
        self.output_layer.update(t, lr, beta1, beta2, weight_decay)

    def update_special(self):
        # Rufe Hebbian Learning und Prototypen-Update auf
        self.bio_layer.hebbian_learn(self.bio_layer.hidden_activations, self.bio_layer.spikes)
        self.bio_layer.update_prototypes()

    def save_checkpoint(self, filepath: str, epoch: int, global_step: int, best_loss: Optional[float] = None):
        checkpoint = {
            'epoch': epoch, 'global_step': global_step,
            'embedding_layer_state': self.embedding_layer.get_state(),
            'bio_layer_state': self.bio_layer.get_state(),
            'output_layer_state': self.output_layer.get_state(),
            'best_valid_loss': best_loss if best_loss is not None else float('inf'),
            'vocab_size': self.vocab_size, 'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim, 'num_prototypes': self.num_prototypes, # <<< NEU: num_prototypes speichern
            'seq_len': self.S # <<< NEU: seq_len speichern
        }
        try:
            backup_path = filepath + ".bak";
            if os.path.exists(filepath):
                if os.path.exists(backup_path): os.remove(backup_path)
                os.rename(filepath, backup_path)
            with open(filepath, 'wb') as f: pickle.dump(checkpoint, f)
            # Minimalerer Print
            # if DEBUG_PRINTS: print(f"[Model] Checkpoint gespeichert: {filepath} (Epoche {epoch}, Schritt {global_step})")
        except Exception as e: print(f"[Model] FEHLER beim Speichern des Checkpoints nach {filepath}: {e}")

    def load_checkpoint(self, filepath: str) -> Tuple[int, int, Optional[float]]:
        if not os.path.exists(filepath):
            print(f"[Model] WARNUNG: Checkpoint-Datei nicht gefunden: {filepath}. Initialisiere Gewichte neu.")
            self._initialize_weights(); return 0, 0, float('inf')
        try:
            print(f"[Model] Lade Checkpoint aus {filepath}...")
            with open(filepath, 'rb') as f: checkpoint = pickle.load(f)
            # Konsistenzprüfung (gegen aktuelle Modellparameter)
            mismatch = False
            if checkpoint.get('vocab_size') != self.vocab_size:
                print(f"[Model] WARNUNG: Checkpoint Vocab Size ({checkpoint.get('vocab_size')}) != Modell ({self.vocab_size})")
                mismatch = True
            if checkpoint.get('embedding_dim') != self.embedding_dim:
                print(f"[Model] WARNUNG: Checkpoint Embedding Dim ({checkpoint.get('embedding_dim')}) != Modell ({self.embedding_dim})")
                mismatch = True
            if checkpoint.get('hidden_dim') != self.hidden_dim:
                print(f"[Model] WARNUNG: Checkpoint Hidden Dim ({checkpoint.get('hidden_dim')}) != Modell ({self.hidden_dim})")
                mismatch = True
            # Prüfe num_prototypes nur, wenn es im Checkpoint existiert (für Abwärtskompatibilität)
            if 'num_prototypes' in checkpoint and checkpoint.get('num_prototypes') != self.num_prototypes:
                print(f"[Model] WARNUNG: Checkpoint Num Prototypes ({checkpoint.get('num_prototypes')}) != Modell ({self.num_prototypes})")
                mismatch = True
            # Prüfe seq_len nur, wenn es im Checkpoint existiert
            if 'seq_len' in checkpoint and checkpoint.get('seq_len') != self.S:
                 print(f"[Model] WARNUNG: Checkpoint Seq Len ({checkpoint.get('seq_len')}) != Modell ({self.S})")
                 mismatch = True

            if mismatch:
                print("[Model] WARNUNG: Checkpoint-Hyperparameter stimmen nicht überein! Initialisiere Gewichte neu.")
                self._initialize_weights(); return 0, 0, float('inf')

            self.embedding_layer.set_state(checkpoint['embedding_layer_state'])
            self.bio_layer.set_state(checkpoint['bio_layer_state'])
            self.output_layer.set_state(checkpoint['output_layer_state'])
            epoch = checkpoint.get('epoch', 0); global_step = checkpoint.get('global_step', 0); best_loss = checkpoint.get('best_valid_loss', float('inf'))
            print(f"[Model] Checkpoint erfolgreich geladen (Epoche {epoch}, Schritt {global_step}, Bester Loss {best_loss:.6f}).")
            return epoch, global_step, best_loss
        except Exception as e:
            print(f"[Model] FEHLER beim Laden des Checkpoints aus {filepath}: {e}. Initialisiere Gewichte neu.")
            self._initialize_weights(); return 0, 0, float('inf')

    def free(self):
        if not self._is_freed:
            if DEBUG_PRINTS: print("[Model] Freeing resources...")
            if hasattr(self, 'embedding_layer') and self.embedding_layer: self.embedding_layer.free()
            if hasattr(self, 'bio_layer') and self.bio_layer: self.bio_layer.free()
            if hasattr(self, 'output_layer') and self.output_layer: self.output_layer.free()
            if hasattr(self, 'criterion') and self.criterion: self.criterion.free()
            if hasattr(self, 'embedding_output_buffer') and self.embedding_output_buffer: self.embedding_output_buffer.free()
            self._is_freed = True

    def __del__(self):
         if not getattr(self, '_is_freed', True): self.free()

    def train_mode(self):
         self.embedding_layer.train(); self.bio_layer.train(); self.output_layer.train()
    def eval_mode(self):
         self.embedding_layer.eval(); self.bio_layer.eval(); self.output_layer.eval()


# --- Lernraten-Scheduler ---
class StepLR:
     def __init__(self, initial_lr: float, step_size: int, gamma: float = 0.1):
          self.lr = initial_lr; self.step_size = step_size; self.gamma = gamma; self.last_epoch = -1
     def step(self, epoch: int):
          if epoch == self.last_epoch: return
          self.last_epoch = epoch
          if epoch > 0 and epoch % self.step_size == 0:
               self.lr *= self.gamma
               print(f"[Scheduler] Lernrate reduziert auf {self.lr:.6g} für Epoche {epoch+1}")
     def get_lr(self) -> float: return self.lr

# --- Genauigkeitsberechnung (angepasst für Padding) ---
def calculate_accuracy(logits_gpu: GPUTensor, targets_np: np.ndarray, current_m_flat: int, vocab_size: int) -> float:
     if logits_gpu.size == 0 or targets_np.size == 0 or current_m_flat == 0: return 0.0
     logits_host_flat = np.zeros(current_m_flat * vocab_size, dtype=FP_TYPE)
     logits_gpu.read(logits_host_flat, offset_bytes=0) # Lese nur relevanten Teil
     logits_host = logits_host_flat.reshape(current_m_flat, vocab_size)

     predictions_flat = np.argmax(logits_host, axis=1)
     targets_flat = targets_np.flatten()[:current_m_flat]

     valid_mask = targets_flat != PAD_INDEX
     if not np.any(valid_mask): return 0.0

     correct_predictions = np.sum(predictions_flat[valid_mask] == targets_flat[valid_mask])
     total_valid = np.sum(valid_mask)

     accuracy = correct_predictions / total_valid if total_valid > 0 else 0.0
     del logits_host_flat, logits_host, predictions_flat, targets_flat, valid_mask # Speicher freigeben
     return accuracy

# --- NEU: Helper-Funktionen für Sampling ---
def encode(text: str, char_to_id: Mapping[str, int]) -> List[int]:
    """ Wandelt einen String in eine Liste von Token-IDs um. """
    return [char_to_id.get(char, 0) for char in text] # Map unknown to 0? Oder Error?

def decode(ids: Union[List[int], np.ndarray], id_to_char: Mapping[int, str]) -> str:
    """ Wandelt eine Liste/Array von Token-IDs zurück in einen String. """
    return "".join([id_to_char.get(token_id, '?') for token_id in ids]) # Map unknown ID to '?'

def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """ Berechnet Softmax-Wahrscheinlichkeiten (numerisch stabil). """
    if temperature <= 0: temperature = 1.0 # Verhindere Division durch Null/Negativ
    logits = logits / temperature
    exp_logits = np.exp(logits - np.max(logits)) # Stabilitäts-Trick
    return exp_logits / np.sum(exp_logits)

def sample_from_probs(probs: np.ndarray) -> int:
    """ Wählt einen Index basierend auf Wahrscheinlichkeiten. """
    # Sicherstellen, dass probs eine gültige Verteilung ist (manchmal gibt es Rundungsfehler)
    probs = probs / np.sum(probs)
    return np.random.choice(len(probs), p=probs)

# --- NEU: Sampling-Funktion ---
def generate_text(
    model_instance: MyModel,
    prompt: str,
    num_chars_to_generate: int,
    seq_len: int,
    vocab_size: int,
    char_to_id: Mapping[str, int],
    id_to_char: Mapping[int, str],
    sampling_input_gpu: GPUTensor, # Wiederverwendbarer GPU-Tensor
    temperature: float = 0.8,      # Temperatur für Sampling
    gpu_index: int = GPU_INDEX
) -> str:
    """ Generiert Text basierend auf einem Prompt mit dem trainierten Modell. """
    print(f"\n--- Generiere Text (Prompt: '{prompt}', Länge: {num_chars_to_generate}, Temp: {temperature}) ---")
    model_instance.eval_mode() # Sicherstellen, dass das Modell im Eval-Modus ist

    tokens = encode(prompt, char_to_id)
    input_buffer_host = np.zeros(seq_len, dtype=INT_TYPE) # Host-Buffer für Input-IDs
    logits_buffer_host = np.zeros(seq_len * vocab_size, dtype=FP_TYPE) # Host-Buffer für Output-Logits

    generation_start_time = time.time()
    try:
        for i in range(num_chars_to_generate):
            # 1. Input vorbereiten (die letzten seq_len Tokens)
            current_input_tokens = tokens[-seq_len:]
            # Padding, falls der Input kürzer als seq_len ist
            num_padding = seq_len - len(current_input_tokens)
            padded_input_tokens = [PAD_INDEX] * num_padding + current_input_tokens # Pre-padding
            # Übertrage in Host-Buffer (1D)
            input_buffer_host[:] = padded_input_tokens

            # 2. Input auf GPU schreiben
            sampling_input_gpu.write(input_buffer_host)

            # 3. Modell-Forward-Pass (Batchsize 1)
            model_instance.set_current_batch_shape(1, seq_len)
            logits_gpu = model_instance.forward(sampling_input_gpu)

            # 4. Logits von GPU lesen (nur die letzte Position ist relevant)
            # Da der Output (M_flat, V) ist, lesen wir den ganzen Output (1*seq_len, V)
            logits_gpu.read(logits_buffer_host)
            # Reshape und letzte Logits holen
            logits_matrix = logits_buffer_host.reshape((seq_len, vocab_size))
            last_logits = logits_matrix[-1] # Die Logits für die Vorhersage des nächsten Zeichens

            # 5. Softmax und Sampling
            probs = softmax(last_logits, temperature=temperature)
            next_token = sample_from_probs(probs)

            # 6. Token hinzufügen
            tokens.append(next_token)

            # Optional: Fortschritt anzeigen
            # if (i + 1) % 50 == 0: print(f"  Generiert {i+1}/{num_chars_to_generate} Zeichen...")

    except Exception as e:
        print(f"\nFEHLER während der Textgenerierung: {e}")
        import traceback
        traceback.print_exc()
        return "[GENERIERUNGSFEHLER]"
    finally:
        generation_duration = time.time() - generation_start_time
        print(f"--- Generierung abgeschlossen ({generation_duration:.2f} sec) ---")

    # 7. Ergebnis dekodieren
    generated_sequence = decode(tokens, id_to_char)
    return generated_sequence


# --- Haupt-Ausführung ---
if __name__ == "__main__":
    # --- Initialisierung & Setup ---
    model: Optional[MyModel] = None
    train_inputs, train_targets = None, None
    valid_inputs, valid_targets = None, None
    char_to_id: Optional[Mapping[str, int]] = None
    id_to_char: Optional[Mapping[int, str]] = None
    input_gpu, targets_gpu = None, None
    sampling_input_gpu: Optional[GPUTensor] = None ### NEU ###
    global_step = 0; start_epoch = 0; best_valid_loss = float('inf')

    try:
        # --- GPU Auswahl ---
        available_gpus = list_available_gpus()
        if not available_gpus: raise RuntimeError("Keine OpenCL GPUs gefunden.")
        # ... (Rest der GPU-Auswahl unverändert) ...
        if len(available_gpus) > 1:
            try:
                choice_str = input(f"Wählen Sie GPU Index (0 bis {len(available_gpus) - 1}) [Enter für 0]: ")
                GPU_INDEX = int(choice_str) if choice_str and 0 <= int(choice_str) < len(available_gpus) else 0
            except (ValueError, EOFError):
                print(f"Ungültige Eingabe oder keine Eingabe, verwende GPU {GPU_INDEX}.")
                GPU_INDEX = 0
        else: GPU_INDEX = 0
        selected_gpu_info = available_gpus[GPU_INDEX][1]
        print(f"[Main] Verwende GPU {GPU_INDEX}: {selected_gpu_info.name}")


        # --- GPU Initialisierung ---
        initialize_selected_gpu(GPU_INDEX)

        # --- Datensätze vorbereiten/laden ---
        if not os.path.exists(input_text_file):
             print(f"WARNUNG: '{input_text_file}' nicht gefunden. Erstelle eine Dummy-Datei.")
             dummy_text = "Dies ist ein kurzer Beispieltext. Er enthält einige Zeichen, Wiederholungen und Zeilenumbrüche.\n" * 50
             with open(input_text_file, 'w', encoding='utf-8') as f: f.write(dummy_text)

        preprocess_char_data(input_text_file, processed_data_file, SEQ_LEN)
        train_inputs, train_targets, valid_inputs, valid_targets, VOCAB_SIZE, char_to_id, id_to_char = load_processed_data(processed_data_file)

        num_train_samples = train_inputs.shape[0]; num_valid_samples = valid_inputs.shape[0]
        if BATCH_SIZE > num_train_samples:
             BATCH_SIZE = num_train_samples if num_train_samples > 0 else 1
             print(f"[Main] WARNUNG: Batch size reduziert auf {BATCH_SIZE}")
        if VOCAB_SIZE <= 0: raise ValueError("Vokabulargröße konnte nicht geladen werden oder ist ungültig.")

        # --- Modell & Scheduler Initialisierung / Laden ---
        M_flat_max = BATCH_SIZE * SEQ_LEN
        model = MyModel(BATCH_SIZE, SEQ_LEN, EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, NUM_TOKEN_PROTOTYPES, gpu_index=GPU_INDEX)

        start_epoch, global_step, best_valid_loss = model.load_checkpoint(checkpoint_path)
        if os.path.exists(best_checkpoint_path):
             _, _, loaded_best_loss = model.load_checkpoint(best_checkpoint_path)
             best_valid_loss = min(best_valid_loss, loaded_best_loss if loaded_best_loss is not None else float('inf'))
             print(f"[Main] Besten bekannten Validierungs-Loss geladen: {best_valid_loss:.6f}")
             start_epoch, global_step, _ = model.load_checkpoint(checkpoint_path)

        scheduler = StepLR(INITIAL_LEARNING_RATE, step_size=LR_DECAY_STEP, gamma=LR_DECAY_GAMMA)
        scheduler.last_epoch = start_epoch -1
        for i in range(start_epoch): scheduler.step(i)
        current_lr = scheduler.get_lr()
        print(f"[Main] Training startet bei Epoche {start_epoch+1}, Schritt {global_step+1}, LR {current_lr:.6g}")

        # --- GPU-Tensoren für Batches und Sampling --- ### MODIFIZIERT ###
        itemsize_int = INT_TYPE().itemsize
        itemsize_fp = FP_TYPE().itemsize
        input_gpu = GPUTensor(BATCH_SIZE * SEQ_LEN * itemsize_int, name="InputIndices_batch", gpu_index=GPU_INDEX)
        targets_gpu = GPUTensor(BATCH_SIZE * SEQ_LEN * itemsize_int, name="Targets_batch", gpu_index=GPU_INDEX)
        # Separater Tensor für Sampling (Batchsize 1)
        sampling_input_gpu = GPUTensor(1 * SEQ_LEN * itemsize_int, name="SamplingInput_GPU", gpu_index=GPU_INDEX) ### NEU ###

        # --- Trainings-Loop ---
        print("\n" + "="*10 + f" Starte Trainings-Loop ({NUM_EPOCHS} Epochen, Start bei {start_epoch+1}) " + "="*10)
        for epoch in range(start_epoch, NUM_EPOCHS):
            epoch_lr = scheduler.get_lr()
            print(f"\n--- Epoche {epoch + 1}/{NUM_EPOCHS} (LR: {epoch_lr:.6g}) ---")
            epoch_start_time = time.time()
            model.train_mode()

            epoch_train_loss = 0.0; num_train_batches = 0
            batch_generator = create_batches(train_inputs, train_targets, BATCH_SIZE, SEQ_LEN, shuffle=True)

            for batch_idx, (batch_inputs_host, batch_targets_host) in enumerate(batch_generator):
                current_batch_size = batch_inputs_host.shape[0]
                if current_batch_size == 0: continue

                # Ignoriere Batches, die nur aus Padding bestehen (kann am Ende vorkommen)
                if np.all(batch_inputs_host == PAD_INDEX): continue

                global_step += 1; num_train_batches += 1
                model.set_current_batch_shape(current_batch_size, SEQ_LEN)

                # Handle potenziell kleinere letzte Batch-Größen beim Schreiben
                input_gpu.write(batch_inputs_host.flatten())
                targets_gpu.write(batch_targets_host.flatten())

                logits = model.forward(input_gpu)

                loss, _ = model.compute_loss_and_backward(logits, targets_gpu, batch_targets_host)

                # Überprüfe auf NaN/Inf Loss
                if not math.isfinite(loss):
                     print(f"[WARNUNG] Ungültiger Loss ({loss}) in Epoche {epoch+1}, Batch {batch_idx}. Überspringe Updates für diesen Batch.")
                     # Optional: Gradienten zurücksetzen, um zu verhindern, dass sie weiter verwendet werden
                     model.embedding_layer.dW_emb._zero_initialize()
                     model.bio_layer.dW1._zero_initialize()
                     model.bio_layer.db1._zero_initialize()
                     model.output_layer.dW._zero_initialize()
                     model.output_layer.db._zero_initialize()
                     continue # Gehe zum nächsten Batch

                epoch_train_loss += loss

                model.clip_all_gradients(GRADIENT_CLIP_VALUE)
                model.update_trainable(global_step, epoch_lr, weight_decay=WEIGHT_DECAY)
                model.update_special()

                if num_train_batches % 1500 == 0:
                    print(f"  [Epoche {epoch+1}, Batch {num_train_batches}/{len(train_inputs)//BATCH_SIZE}] Loss: {loss:.6f}")

            avg_train_loss = epoch_train_loss / num_train_batches if num_train_batches > 0 else 0.0
            print(f"[Epoche {epoch + 1}] Durchschnittlicher Trainings-Loss: {avg_train_loss:.6f}")

            # --- Validierungs-Loop ---
            epoch_valid_loss = 0.0; epoch_valid_accuracy = 0.0; num_valid_batches = 0
            model.eval_mode()
            valid_batch_generator = create_batches(valid_inputs, valid_targets, BATCH_SIZE, SEQ_LEN, shuffle=False)

            for batch_inputs_host, batch_targets_host in valid_batch_generator:
                current_batch_size = batch_inputs_host.shape[0]
                if current_batch_size == 0: continue
                if np.all(batch_inputs_host == PAD_INDEX): continue

                num_valid_batches += 1
                model.set_current_batch_shape(current_batch_size, SEQ_LEN)

                input_gpu.write(batch_inputs_host.flatten())
                targets_gpu.write(batch_targets_host.flatten())

                logits = model.forward(input_gpu)
                loss = model.criterion.forward(logits, targets_gpu, model.bio_layer.current_M_flat, targets_np=batch_targets_host)

                # Nur Loss berechnen, wenn er gültig ist
                if math.isfinite(loss):
                    accuracy = calculate_accuracy(logits, batch_targets_host, model.bio_layer.current_M_flat, VOCAB_SIZE)
                    epoch_valid_loss += loss
                    epoch_valid_accuracy += accuracy
                else:
                    # Wenn Loss ungültig ist, zählt der Batch nicht zur Durchschnittsberechnung
                    # und die Genauigkeit wird auch nicht berechnet.
                    print(f"[WARNUNG] Ungültiger Validierungs-Loss ({loss}) - Batch wird ignoriert.")
                    num_valid_batches -= 1 # Reduziere die Anzahl, um den Durchschnitt nicht zu verfälschen


            avg_valid_loss = epoch_valid_loss / num_valid_batches if num_valid_batches > 0 else float('inf') # Inf wenn keine validen Batches
            avg_valid_accuracy = epoch_valid_accuracy / num_valid_batches if num_valid_batches > 0 else 0.0
            epoch_duration = time.time() - epoch_start_time
            print(f"[Epoche {epoch + 1}] Durchschnittlicher Validierungs-Loss: {avg_valid_loss:.6f}")
            print(f"[Epoche {epoch + 1}] Durchschnittliche Validierungs-Genauigkeit: {avg_valid_accuracy:.4f}")
            print(f"[Epoche {epoch + 1}] Dauer: {epoch_duration:.2f} sec")

            # Checkpoints speichern
            model.save_checkpoint(checkpoint_path, epoch + 1, global_step, best_loss=best_valid_loss)
            if avg_valid_loss < best_valid_loss:
                print(f"[Epoche {epoch + 1}] Neuer bester Validierungs-Loss! Speichere besten Checkpoint.")
                best_valid_loss = avg_valid_loss
                model.save_checkpoint(best_checkpoint_path, epoch + 1, global_step, best_loss=best_valid_loss)

            # Lernrate für nächste Epoche anpassen
            scheduler.step(epoch + 1) # Step *nach* der Epoche aufrufen

            ### NEU: Textgenerierung nach jeder Epoche ###
            if model is not None and char_to_id is not None and id_to_char is not None and sampling_input_gpu is not None:
                # Beispiel-Prompt und Länge
                sampling_prompt = "Ein schöner"
                num_chars_to_generate = 200
                sampling_temperature = 0.6 # Kann angepasst werden

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
                print("-" * 20 + " Generierter Text: " + "-" * 20)
                print(generated_text)
                print("-" * (40 + len(" Generierter Text: ")))
            ### ENDE NEU ###


        print("\n" + "="*10 + " Trainings-Loop beendet " + "="*10)

    except KeyboardInterrupt:
         print("\n[Main] Training durch Benutzer unterbrochen.")
         if model is not None and global_step > 0:
              print("[Main] Speichere aktuellen Stand vor dem Beenden...")
              current_epoch_or_start = epoch + 1 if 'epoch' in locals() and epoch >= start_epoch else start_epoch
              model.save_checkpoint(checkpoint_path, current_epoch_or_start, global_step, best_loss=best_valid_loss)
    except Exception as e:
        print(f"\n--- Ein unerwarteter Fehler ist im Haupt-Loop aufgetreten ---")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        print("\n[Python] Gebe Ressourcen frei...")
        # Liste muss angepasst werden ### MODIFIZIERT ###
        resource_list = [input_gpu, targets_gpu, sampling_input_gpu, model] # model.free() kümmert sich um interne Layer etc.
        for res in resource_list:
             try:
                 if res is not None:
                      if hasattr(res, 'free') and callable(res.free): res.free()
             except Exception as e:
                  item_name = getattr(res, 'name', type(res).__name__) if res else "None"
                  print(f"Fehler beim Freigeben von {item_name}: {e}")
        if 'c_driver' in locals() and c_driver:
            print("[Python] Rufe shutdown_gpu auf...")
            try: c_driver.shutdown_gpu(GPU_INDEX); print("[Python] shutdown_gpu erfolgreich aufgerufen.")
            except Exception as e: print(f"Fehler beim Aufruf von shutdown_gpu: {e}")
        else: print("[Python] c_driver wurde nicht geladen, shutdown_gpu wird übersprungen.")
        print("[Python] Programm beendet.")

