import numpy as np
import time
import random
import socket
import threading
import json
import os
from collections import Counter
from deap import base, creator, tools, algorithms
from concurrent.futures import ThreadPoolExecutor

# ================= GLOBAL =================
RESULT_DIR = "./Result"
os.makedirs(RESULT_DIR, exist_ok=True)
RESULT_FILE = os.path.join(RESULT_DIR, "final_results.txt")

# ================= INPUT SIZE =================
FRAME_SIZE_MB = 32.6
FRAME_SIZE_BITS = FRAME_SIZE_MB * 8 * 1024 * 1024

Twin = []
Twex = []
TTproc = []
Ttransmit = []

epsilon_recv = []
epsilon_proc = []
epsilon_send = []

MOVIE_DURATION_SECONDS = 10  # you can tune this

devices = []
client_sockets = {}
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

    for i in range(100):
        A = np.random.randint(1, 100, (5,5)).tolist()
        B = np.random.randint(1, 100, (5,5)).tolist()

        tasks.append({"id": i, "A": A, "B": B})

    TOTAL_FRAMES = len(tasks)
    return tasks

# ================= OBJECTIVE =================
def compute_objectives(assignment):
    device_times = [0.0] * NUM_DEVICES
    energy = 0.0
    utilizations = []

    for i in range(NUM_DEVICES):
        num_tasks = assignment.count(i)

        T_cij = num_tasks * (Twin[i] + Twex[i] + TTproc[i] + Ttransmit[i])

        E_ij = num_tasks * (
            Twin[i] * epsilon_recv[i]
            + TTproc[i] * epsilon_proc[i]
            + Ttransmit[i] * epsilon_send[i]
        )

        device_times[i] = T_cij
        energy += E_ij

        utilizations.append(
            T_cij / MOVIE_DURATION_SECONDS if MOVIE_DURATION_SECONDS else 0
        )

    completion_time = max(device_times)
    avg_util = sum(utilizations) / NUM_DEVICES
    throughput = TOTAL_FRAMES / completion_time if completion_time > 0 else 0

    return completion_time, energy, avg_util, throughput




def print_task_distribution(assign):
    from collections import Counter

    counts = Counter(assign)

    print("\n📊 Task Distribution (Out of 100):\n")

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


def schedule_based_distribution():
    assignment = []

    for i in range(TOTAL_FRAMES):
        device_id = i % NUM_DEVICES   # round-robin scheduling
        assignment.append(device_id)

    return assignment


# ================= SCHEDULERS =================
def greedy_scheduler():
    assignment = []
    load = [0.0]*NUM_DEVICES

    for _ in range(TOTAL_FRAMES):
        costs = []

        for i in range(NUM_DEVICES):
            base = devices[i]["frame_processing_time"]
            costs.append(base + load[i]*0.5)

        idx = np.argmin(costs)
        assignment.append(idx)
        load[idx] += costs[idx]

    return assignment

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
    swarm = [Particle() for _ in range(30)]
    gbest = swarm[0].pos.copy()
    gbest_fit = float("inf")

    for _ in range(30):

        # 🔥 parallel fitness
        with ThreadPoolExecutor(max_workers=8) as ex:
            fits = list(ex.map(lambda p: p.evaluate(), swarm))

        for p, f in zip(swarm, fits):
            if f < gbest_fit:
                gbest_fit = f
                gbest = p.pos.copy()

        for p in swarm:
            p.update(gbest)

    return gbest.tolist()


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


def compute_total_input_load():
    matrix_size_bytes = FRAME_SIZE_BITS / 8
    total_load = matrix_size_bytes * TOTAL_FRAMES

    print(f"\n📦 Total Input Load: {total_load / (1024**3):.2f} GB")

# ================= RUN =================
def run_experiment():
    global matrix_tasks, NUM_DEVICES
    global Twin, Twex, TTproc, Ttransmit
    global epsilon_recv, epsilon_proc, epsilon_send

    print("\n🚀 Starting Experiment...\n")

    NUM_DEVICES = len(devices)

    bandwidths = [54e6] * NUM_DEVICES

    Ttransmit = [FRAME_SIZE_BITS / bw for bw in bandwidths]
    TTproc = [
        d["frame_processing_time"] * (1 + 0.2 * i)
        for i, d in enumerate(devices)
    ]

    epsilon_send = [0.5] * NUM_DEVICES
    epsilon_recv = [0.3] * NUM_DEVICES
    epsilon_proc = [0.4] * NUM_DEVICES

    Twin = [0.1] * NUM_DEVICES
    Twex = [0.1] * NUM_DEVICES

    bandwidths = [54e6] * NUM_DEVICES  # or different per device if you want

    for i, d in enumerate(devices):

        # ✅ Processing time (from client benchmark)
        proc_time = d["frame_processing_time"]

        # ✅ Transmission time (based on matrix size)
        Tt = FRAME_SIZE_BITS / bandwidths[i]

        # ✅ Receiving time (assume same as transmit or slightly lower)
        Tr = 0.8 * Tt

        # ✅ Execution overhead (small constant)
        Tex = 0.05

        # ✅ Store values
        Twin.append(Tr)
        Twex.append(Tex)
        TTproc.append(proc_time)
        Ttransmit.append(Tt)

        # ✅ Energy model (proportional, NOT random)
        epsilon_recv.append(0.2 * Tr)
        epsilon_proc.append(1.5 * proc_time)
        epsilon_send.append(0.3 * Tt)

        matrix_tasks = generate_matrices()
        matrix_tasks = generate_matrices()
        compute_total_input_load()

    sched = {
    "Schedule-Based": schedule_based_distribution,
    "Greedy": greedy_scheduler,
    "PSO": pso_scheduler,
    "MOPSO": mopso_scheduler,
    "MOMPSO-GA": mompso_ga_scheduler
}

    results = ""

    for name, fn in sched.items():
        st = time.time()

        result = fn()

        if isinstance(result, tuple):
            assign = result[0]
        else:
            assign = result

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

    with open(RESULT_FILE, "w") as f:
        f.write("===== Scheduler Evaluation (Server Side) =====\n\n")
        f.write(results)


    for s in client_sockets.values():
        send_json(s, {"type": "EXPERIMENT_FINISHED"})

# ================= RESET =================
def reset_state():
    global devices, client_sockets, device_result_files

    devices = []
    client_sockets = {}
    

# ================= CLIENT HANDLER =================
def handle_client(sock, addr):
    global last_connection_time

    print("\nClient connected:", addr)

    info = recv_json(sock)
    print("Device registered:", info)

    with lock:
        did = len(devices)
        info["ip"] = addr[0]
        devices.append(info)
        client_sockets[did] = sock
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
        if time.time() - last_prompt_time >= 60:

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
    s.bind(("0.0.0.0", 7070))
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.listen(10)

    print("Server running on 0.0.0.0 7070")

    threading.Thread(target=monitor, daemon=True).start()

    while True:
        c, addr = s.accept()
        threading.Thread(target=handle_client, args=(c, addr)).start()

if __name__ == "__main__":
    start_server()