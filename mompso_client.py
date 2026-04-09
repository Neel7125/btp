# ===== client_optimized.py =====

import socket, json, numpy as np, time, platform, random

SERVER_IP = "10.200.5.120"
PORT = 7070

# ---------- SOCKET ----------
def send_json(sock, data):
    sock.send((json.dumps(data) + "\n").encode())

def recv_json(sock):
    buffer = ""
    while True:
        buffer += sock.recv(4096).decode()
        if "\n" in buffer:
            msg, buffer = buffer.split("\n", 1)
            return json.loads(msg)

# ---------- COMPUTE ----------
def multiply(A, B):
    A = np.array(A)
    B = np.array(B)
    return np.dot(A, B)

# ---------- CLIENT ----------
def start():
    s = socket.socket()
    s.connect((SERVER_IP, PORT))

    # 🔥 Strong heterogeneity
    device_profiles = [0.3, 0.7, 1.5, 3.0, 5.0]

    device_info = {
        "name": platform.node(),
        "speed": random.choice(device_profiles),
        "latency": random.uniform(0.01, 0.2),
        "bandwidth": random.uniform(5, 50)
    }

    send_json(s, device_info)
    print("Connected:", device_info)

    while True:
        try:
            msg = recv_json(s)

            if msg["type"] == "TASK":
                start = time.time()

                multiply(msg["A"], msg["B"])

                end = time.time()

                send_json(s, {
                    "task_id": msg["task_id"],
                    "device": device_info["name"],
                    "processing_time": end - start
                })

            elif msg["type"] == "STOP":
                break

        except Exception as e:
            print("Error:", e)
            break

if __name__ == "__main__":
    start()