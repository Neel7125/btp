import socket
import json
import time
import numpy as np
import psutil
import platform

SERVER_IP = input("Enter server IP: ")
PORT = 7070

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

def compute_processing_time():
    A = np.random.rand(10,10)
    start = time.time()
    np.linalg.det(A @ A.T)
    return time.time() - start

device_info = {
    "name": platform.node(),
    "processor_speed": psutil.cpu_freq().max/1000 if psutil.cpu_freq() else 2,
    "cores": psutil.cpu_count(),
    "memory": round(psutil.virtual_memory().total/(1024**3),2),
    "frame_processing_time": compute_processing_time()
}

def process(A):
    A = np.array(A)
    start = time.time()
    np.linalg.det(A @ A.T)
    return time.time() - start

s = socket.socket()
s.connect((SERVER_IP, PORT))

send_json(s, device_info)

print("Connected to server")

while True:
    task = recv_json(s)

    if task["type"] == "MATRIX_TASK":
        t = process(task["A"])

        send_json(s, {
            "task_id": task["task_id"],
            "device": device_info["name"],
            "processing_time": t
        })

    elif task["type"] == "EXPERIMENT_FINISHED":
        print("Finished")
        break

s.close()