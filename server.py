import numpy as np
import time
import random
import socket
import threading
import json
import os
from collections import Counter
from deap import base, creator, tools, algorithms

# ================= GLOBAL =================
RESULT_DIR = "./Result"
os.makedirs(RESULT_DIR, exist_ok=True)

devices = []
client_sockets = {}
device_capacities = {}
device_result_files = {}

matrix_tasks = []
TOTAL_FRAMES = 0
NUM_DEVICES = 0

last_connection_time = time.time()
experiment_running = False

lock = threading.Lock()

# ================= SOCKET =================
def send_json(sock, data):
    sock.send((json.dumps(data) + "\n").encode())

def recv_json(sock):
    buffer = ""
    while True:
        chunk = sock.recv(4096).decode()
        if not chunk:
            raise Exception("Client disconnected")
        buffer += chunk
        if "\n" in buffer:
            msg, buffer = buffer.split("\n", 1)
            return json.loads(msg)

# ================= MATRIX =================
def generate_matrices():
    global TOTAL_FRAMES
    tasks = []

    for i in range(50):
        A = np.random.randint(1, 10, (3,3)).tolist()
        B = np.random.randint(1, 10, (3,3)).tolist()

        tasks.append({"id": i, "A": A, "B": B})

    TOTAL_FRAMES = len(tasks)
    return tasks

# ================= RESULT =================
def create_result_file(device_id, name):
    path = os.path.join(RESULT_DIR, name.replace(" ", "_") + "_results.txt")
    device_result_files[device_id] = path
    with open(path, "w") as f:
        f.write("===== Scheduler Evaluation =====\n\n")

def save_result(device_id, text):
    with open(device_result_files[device_id], "a") as f:
        f.write(text + "\n")

# ================= OBJECTIVE =================
def compute_objectives(assign):
    counts = Counter(assign)

    times = []
    energy = 0
    util = []

    for i in range(NUM_DEVICES):
        n = counts.get(i, 0)
        t = devices[i]["frame_processing_time"]

        Tc = n * t
        Ec = n * t * 0.5

        times.append(Tc)
        energy += Ec
        util.append(Tc / (TOTAL_FRAMES + 1))

    comp = max(times)
    avg = sum(util) / NUM_DEVICES
    thr = TOTAL_FRAMES / comp if comp > 0 else 0

    return comp, energy, avg, thr

# ================= CAPACITY =================
def capacity_scheduler():
    assign = []

    for device_id in range(NUM_DEVICES):
        count = device_capacities[device_id]
        assign.extend([device_id] * count)

    # shuffle for randomness
    random.shuffle(assign)

    return assign

def compute_device_capacities():
    scores = []

    for d in devices:
        cpu = d["cores"] * d["processor_speed"]
        mem = d["memory"]
        bat = max(d["battery"], 0) / 100

        scores.append(0.5*cpu + 0.3*mem + 0.2*bat)

    total = sum(scores)

    caps = {}
    for i, s in enumerate(scores):
        caps[i] = int((s / total) * TOTAL_FRAMES)

    while sum(caps.values()) < TOTAL_FRAMES:
        caps[np.argmax(scores)] += 1

    return caps

def print_task_distribution(assign):
    from collections import Counter

    counts = Counter(assign)

    print("\n📊 Task Distribution (Out of 50):\n")

    for i in range(NUM_DEVICES):
        device_name = devices[i]["name"]
        task_count = counts.get(i, 0)

        print(f"{device_name} → {task_count} tasks")

    print("-" * 40)

# ================= EXECUTION =================
def send_matrix_task(did, tid):
    task = matrix_tasks[tid]

    send_json(client_sockets[did], {
        "type": "MATRIX_TASK",
        "task_id": task["id"],
        "A": task["A"]
    })

def receive_result(did):
    return recv_json(client_sockets[did])

def execute_assignment(assign):
    counts = {i: 0 for i in range(NUM_DEVICES)}

    for tid, did in enumerate(assign):

        send_matrix_task(did, tid)
        res = receive_result(did)

        print(f"\n[SERVER] Task {res['task_id']} processed by {res.get('device')}")
        print(f"[SERVER] Time: {res.get('processing_time', 0):.6f}s")

        counts[did] += 1


def capacity_based_scheduler():
    assign = []

    for device_id in range(NUM_DEVICES):
        count = device_capacities[device_id]
        assign.extend([device_id] * count)

    # Shuffle so tasks are not sequentially grouped
    random.shuffle(assign)

    return assign


# ================= SCHEDULERS =================
def greedy_scheduler():
    load = [0]*NUM_DEVICES
    assign = []

    for _ in range(TOTAL_FRAMES):
        i = np.argmin(load)
        assign.append(i)
        load[i]+=1

    return assign

# (KEEP YOUR PSO / MOPSO / GA SAME HERE)
# ================= PSO =================
class Particle:

    def __init__(self):
        self.position = np.random.randint(0, NUM_DEVICES, TOTAL_FRAMES)
        self.velocity = np.random.randint(-1, 2, TOTAL_FRAMES)
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')

    def evaluate(self):
        comp, ene, util, _ = compute_objectives(self.position.tolist())
        fitness = 0.4 * comp + 0.3 * ene + 0.3 * (1 - util)

        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_position = self.position.copy()

        return fitness

    def update(self, global_best):
        for i in range(len(self.velocity)):
            self.velocity[i] = int(
                0.5*self.velocity[i]
                +1.5*random.random()*(self.best_position[i]-self.position[i])
                +1.5*random.random()*(global_best[i]-self.position[i])
            )
            self.position[i] = (self.position[i] + self.velocity[i]) % NUM_DEVICES


def pso_scheduler():

    swarm = [Particle() for _ in range(10)]

    best = swarm[0].position.copy()
    best_fit = float('inf')

    for _ in range(10):

        for p in swarm:
            fit = p.evaluate()
            if fit < best_fit:
                best_fit = fit
                best = p.position.copy()

        for p in swarm:
            p.update(best)

    return best.tolist()


# ================= MOPSO =================
def mopso_scheduler():

    best = None
    best_score = float("inf")

    for _ in range(10):
        candidate = [random.randint(0, NUM_DEVICES-1) for _ in range(TOTAL_FRAMES)]

        comp, ene, util, _ = compute_objectives(candidate)

        fitness = 0.4*comp + 0.3*ene + 0.3*(1-util)

        if fitness < best_score:
            best_score = fitness
            best = candidate

    return best


# ================= MOMPSO-GA =================
if not hasattr(creator, "FitnessMulti"):
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0,-1.0,-1.0))

if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMulti)


def mompso_ga_scheduler():

    toolbox = base.Toolbox()

    toolbox.register("attr_int", lambda: random.randint(0, NUM_DEVICES-1))
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, TOTAL_FRAMES)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=NUM_DEVICES-1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", lambda ind: compute_objectives(ind)[:3])

    pop = toolbox.population(n=10)

    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=5, verbose=False)

    best = tools.selBest(pop, k=1)[0]

    return list(best)

# ================= RUN =================
def run_experiment():
    global matrix_tasks, NUM_DEVICES, device_capacities

    print("\n🚀 Starting Experiment...\n")

    NUM_DEVICES = len(devices)
    matrix_tasks = generate_matrices()
    device_capacities = compute_device_capacities()

    sched = {
        "Greedy": greedy_scheduler,
        "PSO": pso_scheduler,
        "MOPSO": mopso_scheduler,
        "MOMPSO-GA": mompso_ga_scheduler
    }

    results = ""

    for name, fn in sched.items():
        st = time.time()

        assign = capacity_scheduler()

        print_task_distribution(assign)

        execute_assignment(assign)

        c,e,u,t = compute_objectives(assign)
        el = time.time()-st

        out = f"""{name}
Completion Time: {c:.2f}
Energy: {e:.2f}
Utilization: {u:.4f}
Throughput: {t:.2f}
Time: {el:.2f}
"""
        print(out)
        results += out + "\n"

    for d in range(len(devices)):
        save_result(d, results)

    for s in client_sockets.values():
        send_json(s, {"type": "EXPERIMENT_FINISHED"})

# ================= RESET =================
def reset_state():
    global devices, client_sockets, device_result_files

    devices = []
    client_sockets = {}
    device_result_files = {}

# ================= CLIENT HANDLER =================
def handle_client(sock, addr):
    global last_connection_time

    print("\nClient connected:", addr)

    info = recv_json(sock)
    print("Device registered:", info)

    with lock:
        did = len(devices)
        devices.append(info)
        client_sockets[did] = sock
        create_result_file(did, info["name"])

        last_connection_time = time.time()

# ================= MONITOR =================
def monitor():
    global experiment_running

    last_prompt_time = time.time()

    while True:
        time.sleep(2)

        if experiment_running:
            continue

        # ⏱️ Check fixed 1-minute interval
        if time.time() - last_prompt_time >= 30:

            choice = input("Do you want to start Experiment (Y/N): ")

            # Reset timer ALWAYS after asking
            last_prompt_time = time.time()

            if choice.lower() == 'y':

                if len(devices) == 0:
                    print("⚠️ No clients connected!\n")
                    continue

                experiment_running = True

                run_experiment()

                reset_state()
                experiment_running = False

                print("\n🔄 Server ready for next batch...\n")

            else:
                print("⏳ Waiting for next 1-minute interval...\n")

# ================= START =================
def start_server():
    s = socket.socket()
    s.bind(("0.0.0.0", 9090))
    s.listen(10)

    print("Server running on 0.0.0.0 9090")

    threading.Thread(target=monitor, daemon=True).start()

    while True:
        c, addr = s.accept()
        threading.Thread(target=handle_client, args=(c, addr)).start()

if __name__ == "__main__":
    start_server()
