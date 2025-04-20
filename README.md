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
Results:

![image](https://github.com/user-attachments/assets/658a88dc-5943-49ec-9444-2ed962199e6c)

![image](https://github.com/user-attachments/assets/d4d283cf-1ee9-4c12-aa9b-d2fd0f78e2dd)

![image](https://github.com/user-attachments/assets/8f3bb38e-b6a1-43dc-b429-803611ac02a7)

![image](https://github.com/user-attachments/assets/2d0648c2-aa0f-479f-971a-ed67a5984656)

![image](https://github.com/user-attachments/assets/4187088c-d3ae-480a-af7c-32aa0a5069eb)

![image](https://github.com/user-attachments/assets/f44ebd38-a23a-4a41-91d5-d5b1041ff193)

![image](https://github.com/user-attachments/assets/6faeb539-c821-4a8e-8336-d4aa05d05f6c)

![image](https://github.com/user-attachments/assets/21cefc85-773a-4358-b4d1-129aa9fcb916)

![image](https://github.com/user-attachments/assets/d445dd8c-14d9-401f-b732-8b2849b26f2d)









Conclusion:
The hybrid MARL approach (DQN + MBO + BDO) shows improved performance over standalone methods in smart city load balancing simulations. It reduces energy consumption, enhances PDR, and improves training efficiency.
