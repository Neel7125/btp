import subprocess
import json
import time
import psutil
import platform
import numpy as np

FILE_PATH = "/data/data/com.termux/files/home//storage/dcim/lab/device_info.json"

# ================= DEVICE NAME =================
def get_device_name():
    try:
        model = subprocess.check_output(["getprop", "ro.product.model"]).decode().strip()
        brand = subprocess.check_output(["getprop", "ro.product.brand"]).decode().strip()
        return f"{brand} {model}"
    except:
        return platform.node()

# ================= BATTERY =================
def get_battery():
    try:
        out = subprocess.check_output(["termux-battery-status"])
        return json.loads(out)["percentage"]
    except:
        return -1

# ================= CPU SPEED =================
def get_cpu_speed():
    try:
        freq = psutil.cpu_freq()
        if freq:
            return round(freq.max / 1000, 2)  # GHz
        return 2.0
    except:
        return 2.0

# ================= CORES =================
def get_cores():
    try:
        return psutil.cpu_count()
    except:
        return 1

# ================= MEMORY =================
def get_memory():
    try:
        return round(psutil.virtual_memory().total / (1024**3), 2)  # GB
    except:
        return 2.0

def compute_frame_processing_time():
    # Step 1: Generate random 5x5 matrix
    A = np.random.randint(1, 100, (5, 5))

    start = time.time()

    # Step 2: Transpose
    A_T = A.T

    # Step 3: Multiply
    C = np.dot(A, A_T)

    # Step 4: Determinant
    _ = np.linalg.det(C)

    end = time.time()

    return end - start

# ================= MAIN LOOP =================
while True:
    data = {
        "name": get_device_name(),
        "processor_speed": get_cpu_speed(),
        "cores": get_cores(),
        "memory": get_memory(),
        "battery": get_battery(),
        "frame_processing_time": compute_frame_processing_time(),
        "timestamp": time.time()
    }

    try:
        with open(FILE_PATH, "w") as f:
            json.dump(data, f)
        print("Updated:", data)
    except Exception as e:
        print("Write error:", e)

    time.sleep(30)
