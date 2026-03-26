import socket
import json
import platform
import psutil
import subprocess
import time
import numpy as np

SERVER_IP = input("Enter server IP: ")
PORT = int(input("Enter port: "))

# ================= DEVICE INFO =================
import json

def get_device_info():
    try:
        file_path = "/storage/emulated/0/dcim/lab/device_info.json"

        with open(file_path, "r") as f:
            data = json.load(f)

        return data

    except Exception as e:
        print("Read error:", e)
        return {
            "name": "Unknown",
            "processor_speed": 2,
            "cores": 1,
            "memory": 2,
            "battery": -1,
            "frame_processing_time": 0.3
        }

# ================= USE IT =================
device_info = get_device_info()

# ================= SOCKET =================
def send_json(sock, data):
    sock.send((json.dumps(data) + "\n").encode())

def recv_json(sock):
    buffer = ""
    while True:
        chunk = sock.recv(4096).decode()
        if not chunk:
            raise Exception("Disconnected")
        buffer += chunk
        if "\n" in buffer:
            msg, buffer = buffer.split("\n", 1)
            return json.loads(msg)

# ================= CONNECT =================
s = socket.socket()
s.connect((SERVER_IP, PORT))

send_json(s, device_info)

print("Connected to server")

# ================= COMPUTATION =================
def process_matrix(A):
    start = time.time()

    A = np.array(A, dtype=np.float64)

    # Step 1: Transpose
    A_T = A.T

    # Step 2: Multiply
    B = np.matmul(A, A_T)

    # Step 3: Determinant
    det = np.linalg.det(B)

    processing_time = time.time() - start

    return det, processing_time

# ================= LOOP =================
while True:
    try:
        task = recv_json(s)

        if task["type"] == "MATRIX_TASK":

            task_id = task["task_id"]
            A = task["A"]

            print(f"\n[CLIENT] Processing Task {task_id}")

            det_value, processing_time = process_matrix(A)

            print(f"[CLIENT] Determinant: {det_value:.4f}")

            # ===== SEND RESULT =====
            send_json(s, {
                "task_id": task_id,
                "device": device_info["name"],
                "determinant": float(det_value),
                "processing_time": processing_time
            })

        elif task["type"] == "EXPERIMENT_FINISHED":
            print("✅ Experiment Finished")
            break

    except Exception as e:
        print("Error:", e)
        break

s.close()
