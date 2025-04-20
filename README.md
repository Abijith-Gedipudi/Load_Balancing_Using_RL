#Load Balancing and Smart City Optimization using Hybrid MARL (MBO + BDO + DQN)

Description:
This project focuses on optimizing load balancing and smart city resource allocation using a hybrid Multi-Agent Reinforcement Learning (MARL) strategy. It combines Deep Q-Networks (DQN) with Migrating Birds Optimization (MBO) and Bottlenose Dolphin Optimization (BDO) to enhance performance in terms of energy efficiency, reward maximization, and network metrics.

Core Techniques Used:

1)Deep Q-Network (DQN): Used as the learning agent to interact with the environment.

2)Migrating Birds Optimization (MBO): A metaheuristic inspired by the V-formation flight pattern for exploration.

3)Bottlenose Dolphin Optimization (BDO): Used to refine solutions using intelligent search.

4)Hybrid MBO-BDO: Combines MBO and BDO to balance global and local search strategies, enhancing the RL agent's efficiency.

Files Included:

1)load_balancing_rl_simulation.py: Simulation of different load balancing methods using RL and heuristics.

2)load_balancing_results_summary.py: Script to summarize average CPU utilization, response time, and load distribution.

3)smartcity_hybrid_rl.py: Smart city environment with a DQN agent optimized using MBO and BDO.

4)hybrid_mbo_bdo_performance_analysis.py: Visualization of reward, energy usage, delay, and other metrics.

5)Output images: PNG files generated from the performance analysis script.

Key Metrics Visualized:

1)Total Reward per Episode

2)Epsilon Decay over Training

3)Energy Consumption vs. Rounds

4)Network Lifetime (Alive Nodes)

5)Packet Delivery Ratio (PDR)

6)Average Delay vs. Number of Nodes

7)Loss Function during Training

8)Training Time Comparison between MBO, BDO, and Hybrid approaches

Conclusion:
The hybrid MARL approach (DQN + MBO + BDO) shows improved performance over standalone methods in smart city load balancing simulations. It reduces energy consumption, enhances PDR, and improves training efficiency.
