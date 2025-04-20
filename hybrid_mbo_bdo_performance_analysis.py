import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Number of episodes
episodes = np.arange(1, 101)
# Simulated Reward Values (Replace with actual training values if available)
rewards_mbo = np.cumsum(np.random.randint(5, 20, 100))  # Simulated data for MBO
rewards_bdo = np.cumsum(np.random.randint(4, 18, 100))  # Simulated data for BDO
rewards_hybrid = np.cumsum(np.random.randint(6, 22, 100))  # Simulated data for Hybrid MBO-BDO
# Simulated Loss Values and Epsilon Decay (For Training Analysis)
loss_values = np.random.uniform(0.1, 0.5, 100)  # Example loss values
epsilon_values = np.linspace(1, 0.01, 100)  # Epsilon decay over episodes
# Simulated Energy Consumption Data Over 100 Rounds (Lower is Better)
rounds = np.arange(1, 101)
energy_mbo = np.linspace(100, 30, 100)  # Energy consumption in MBO
energy_bdo = np.linspace(100, 35, 100)  # Energy consumption in BDO
energy_hybrid = np.linspace(100, 25, 100)  # Hybrid approach conserves more energy
# Simulated Packet Delivery Ratio (PDR) for Different Node Counts
nodes = np.array([10, 20, 30, 40, 50])
pdr_mbo = np.random.uniform(0.7, 0.95, 5)  # PDR in MBO
pdr_bdo = np.random.uniform(0.75, 0.97, 5)  # PDR in BDO
pdr_hybrid = np.random.uniform(0.8, 0.99, 5)  # Hybrid achieves best PDR
# Simulated Delay vs. Number of Nodes (Lower is Better)
delay_mbo = np.random.uniform(10, 25, 5)  # Delay in MBO
delay_bdo = np.random.uniform(8, 20, 5)  # Delay in BDO
delay_hybrid = np.random.uniform(5, 15, 5)  # Hybrid minimizes delay
# Simulated Training Times (Seconds) for Comparison
training_times = [150, 135, 120]  # MBO, BDO, Hybrid
#Total Reward Per Episode
plt.figure(figsize=(10, 5))
plt.plot(episodes, rewards_mbo, label='MBO', linestyle='--', marker='o')
plt.plot(episodes, rewards_bdo, label='BDO', linestyle='-.', marker='s')
plt.plot(episodes, rewards_hybrid, label='Hybrid MBO-BDO', linestyle='-', marker='d', linewidth=2)
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Total Reward per Episode")
plt.legend()
plt.grid(True)
plt.savefig("reward_per_episode.png", dpi=300)
plt.show()
#Epsilon Decay Over Training
plt.figure(figsize=(10, 5))
plt.plot(episodes, epsilon_values, color='red', marker='o', linestyle='--')
plt.xlabel("Episodes")
plt.ylabel("Exploration Rate (Epsilon)")
plt.title("Exploration vs. Exploitation (Epsilon Decay)")
plt.grid(True)
plt.savefig("epsilon_decay.png", dpi=300)
plt.show()
#Energy Consumption Over Rounds
plt.figure(figsize=(10, 5))
plt.plot(rounds, energy_mbo, label='MBO', linestyle='--', marker='o')
plt.plot(rounds, energy_bdo, label='BDO', linestyle='-.', marker='s')
plt.plot(rounds, energy_hybrid, label='Hybrid MBO-BDO', linestyle='-', marker='d', linewidth=2)
plt.xlabel("Rounds")
plt.ylabel("Energy Consumption (%)")
plt.title("Energy Consumption vs. Rounds")
plt.legend()
plt.grid(True)
plt.savefig("energy_consumption.png", dpi=300)
plt.show()
#Network Lifetime Analysis (Alive Nodes)
plt.figure(figsize=(10, 5))
plt.plot(rounds, 100 - energy_mbo, label='MBO', linestyle='--', marker='o')
plt.plot(rounds, 100 - energy_bdo, label='BDO', linestyle='-.', marker='s')
plt.plot(rounds, 100 - energy_hybrid, label='Hybrid MBO-BDO', linestyle='-', marker='d', linewidth=2)
plt.xlabel("Rounds")
plt.ylabel("Alive Nodes (%)")
plt.title("Network Lifetime Analysis (Alive Nodes Over Rounds)")
plt.legend()
plt.grid(True)
plt.savefig("alive_nodes.png", dpi=300)
plt.show()
#Packet Delivery Ratio (PDR) Comparison
plt.figure(figsize=(10, 5))
plt.plot(nodes, pdr_mbo, label='MBO', marker='o', linestyle='--')
plt.plot(nodes, pdr_bdo, label='BDO', marker='s', linestyle='-.')
plt.plot(nodes, pdr_hybrid, label='Hybrid MBO-BDO', marker='d', linestyle='-')
plt.xlabel("Number of Nodes")
plt.ylabel("Packet Delivery Ratio (PDR)")
plt.title("Packet Delivery Ratio (PDR) Comparison")
plt.legend()
plt.grid(True)
plt.savefig("packet_delivery_ratio.png", dpi=300)
plt.show()
#Loss Function Over Training
plt.figure(figsize=(10, 5))
plt.plot(episodes, loss_values, color='blue', linestyle='-', linewidth=2)
plt.xlabel("Episodes")
plt.ylabel("Loss Value")
plt.title("Loss Function Over Training")
plt.grid(True)
plt.savefig("loss_function.png", dpi=300)
plt.show()
#Delay vs. Number of Nodes
plt.figure(figsize=(10, 5))
plt.plot(nodes, delay_mbo, label='MBO', marker='o', linestyle='--')
plt.plot(nodes, delay_bdo, label='BDO', marker='s', linestyle='-.')
plt.plot(nodes, delay_hybrid, label='Hybrid MBO-BDO', marker='d', linestyle='-')
plt.xlabel("Number of Nodes")
plt.ylabel("Average Delay (ms)")
plt.title("Delay vs. Number of Nodes")
plt.legend()
plt.grid(True)
plt.savefig("delay_vs_nodes.png", dpi=300)
plt.show()
#Training Time Comparison
plt.figure(figsize=(8, 5))
plt.bar(["MBO", "BDO", "Hybrid MBO-BDO"], training_times, color=["blue", "green", "red"])
plt.xlabel("Optimization Algorithm")
plt.ylabel("Training Time (Seconds)")
plt.title("Training Time Comparison")
plt.grid(True)
plt.savefig("training_time_comparison.png", dpi=300)
plt.show()
