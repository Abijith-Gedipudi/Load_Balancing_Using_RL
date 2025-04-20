import numpy as np
import matplotlib.pyplot as plt
import random

# Environment settings
num_servers = 5
num_tasks = 100
tasks = np.random.randint(5, 20, num_tasks)  # Task loads
server_capacity = np.random.randint(50, 100, num_servers)  # Server capacities

# Load Balancing Algorithms
def round_robin(servers, tasks):
    index = 0
    for task in tasks:
        servers[index]["load"] += task
        index = (index + 1) % num_servers  # Move to next server
    return servers

def migrating_birds_optimization(servers, tasks):
    sorted_servers = sorted(servers, key=lambda x: x["load"])
    for task in tasks:
        sorted_servers[0]["load"] += task  # Assign to least loaded server
        sorted_servers.sort(key=lambda x: x["load"])
    return servers

def dolphin_optimization(servers, tasks):
    for _ in range(5):  # Iterative load redistribution
        high_load_servers = [s for s in servers if s["load"] > np.mean([s["load"] for s in servers])]
        low_load_servers = [s for s in servers if s["load"] < np.mean([s["load"] for s in servers])]
        for hl in high_load_servers:
            if low_load_servers:
                target = random.choice(low_load_servers)
                transfer_load = (hl["load"] - target["load"]) // 2
                hl["load"] -= transfer_load
                target["load"] += transfer_load
    return servers

# Q-learning settings
Q_table = np.zeros((num_tasks, num_servers))
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.2  # Exploration rate

def q_learning_load_balancing(servers, tasks, Q_table):
    for task in tasks:
        state = min(int(np.mean([s["load"] for s in servers])), num_tasks - 1)  # Fix index issue
        action = np.argmax(Q_table[state]) if np.random.rand() > epsilon else np.random.randint(num_servers)
        servers[action]["load"] += task
        reward = -abs(servers[action]["load"] - np.mean([s["load"] for s in servers]))
        Q_table[state, action] += alpha * (reward + gamma * np.max(Q_table[state]) - Q_table[state, action])
    return servers, Q_table

# Hybrid MARL (Combining RL + MBO + BDO)
def hybrid_marl(servers, tasks, Q_table):
    servers = migrating_birds_optimization(servers, tasks)
    servers = dolphin_optimization(servers, tasks)
    servers, Q_table = q_learning_load_balancing(servers, tasks, Q_table)
    return servers, Q_table

# Simulation
methods = {
    "Round Robin": round_robin,
    "Migrating Birds": migrating_birds_optimization,
    "Dolphin Optimization": dolphin_optimization,
    "Q-Learning": lambda servers, tasks: q_learning_load_balancing(servers, tasks, Q_table)[0],
    "Hybrid MARL": lambda servers, tasks: hybrid_marl(servers, tasks, Q_table)[0]
}

metrics = {method: {"cpu_util": [], "response_time": [], "load_balance": [], "convergence": []} for method in methods}

for method, func in methods.items():
    servers = [{"id": i, "capacity": server_capacity[i], "load": 0} for i in range(num_servers)]
    server_load_history = []

    for episode in range(100):  # Simulate 100 iterations
        tasks = np.clip(tasks + np.random.normal(0, 3, size=num_tasks), 5, 20).astype(int)  # Task fluctuation
        servers = func(servers, tasks)  # Run the algorithm
        server_loads = [s["load"] for s in servers]

        # Metrics Calculation
        metrics[method]["cpu_util"].append(np.mean([s["load"] / s["capacity"] for s in servers]) * 100)  # FIXED
        metrics[method]["response_time"].append(np.max(server_loads) / num_tasks)  # Approximate response time
        metrics[method]["load_balance"].append(np.std(server_loads))  # Standard deviation of load
        server_load_history.append(np.mean(server_loads))

    metrics[method]["convergence"] = server_load_history  # Track load balancing over time

# Visualization
plt.figure(figsize=(12, 10))

# CPU Utilization
plt.subplot(2, 2, 1)
for method in methods:
    plt.plot(range(100), metrics[method]["cpu_util"], label=method)
plt.xlabel("Episodes")
plt.ylabel("CPU Utilization (%)")
plt.legend()
plt.title("CPU Utilization Comparison")

# Response Time
plt.subplot(2, 2, 2)
for method in methods:
    plt.plot(range(100), metrics[method]["response_time"], label=method)
plt.xlabel("Episodes")
plt.ylabel("Response Time (ms)")
plt.legend()
plt.title("Response Time Comparison")

# Load Balancing Efficiency (Lower is Better)
plt.subplot(2, 2, 3)
for method in methods:
    plt.plot(range(100), metrics[method]["load_balance"], label=method)
plt.xlabel("Episodes")
plt.ylabel("Load Balance Efficiency (Std Dev)")
plt.legend()
plt.title("Load Balancing Efficiency")

# Convergence Speed
plt.subplot(2, 2, 4)
for method in methods:
    plt.plot(range(100), metrics[method]["convergence"], label=method)
plt.xlabel("Episodes")
plt.ylabel("Average Load per Server")
plt.legend()
plt.title("Convergence Speed Comparison")

plt.tight_layout()
plt.show()
