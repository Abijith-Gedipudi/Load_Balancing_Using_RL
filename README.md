#  Load Balancing and Smart City Optimization using Hybrid MARL (MBO + BDO + DQN)

##  Description  
This project optimizes **load balancing** and **smart city resource allocation** using a **Hybrid Multi-Agent Reinforcement Learning (MARL)** strategy. It combines:

- **Deep Q-Networks (DQN)** for intelligent decision-making  
- **Migrating Birds Optimization (MBO)** for global exploration  
- **Bottlenose Dolphin Optimization (BDO)** for fine-tuned exploitation  

Together, these techniques enhance performance by improving energy efficiency, maximizing rewards, and stabilizing network metrics.

---

##  Core Techniques

- ** Deep Q-Network (DQN):** RL agent interacting with environment
- ** Migrating Birds Optimization (MBO):** Bird flight-inspired metaheuristic for exploration
- ** Bottlenose Dolphin Optimization (BDO):** Dolphin hunting-inspired optimization for refinement
- ** Hybrid MBO-BDO:** Balances global and local search strategies for better learning

---

##  Files Included

- `load_balancing_rl_simulation.py`: Simulates RL and heuristic-based load balancing  
- `load_balancing_results_summary.py`: Summarizes CPU usage, response time, and load distribution  
- `smartcity_hybrid_rl.py`: Implements smart city environment with DQN + MBO + BDO  
- `hybrid_mbo_bdo_performance_analysis.py`: Analyzes performance metrics  
- `*.png`: Generated performance graphs and charts

---

##  Key Metrics Visualized

-  Total Reward per Episode  
-  Epsilon Decay during Training  
-  Energy Consumption vs. Simulation Rounds  
-  Network Lifetime (Alive Nodes Over Time)  
-  Packet Delivery Ratio (PDR)  
-  Average Delay vs. Number of Nodes  
-  Loss Function During Training  
-  Training Time Comparison (MBO vs. BDO vs. Hybrid)

---

##  Results (Graphs & Visualizations)

![Total Reward](https://github.com/user-attachments/assets/658a88dc-5943-49ec-9444-2ed962199e6c)  
![Epsilon Decay](https://github.com/user-attachments/assets/d4d283cf-1ee9-4c12-aa9b-d2fd0f78e2dd)  
![Energy Consumption](https://github.com/user-attachments/assets/8f3bb38e-b6a1-43dc-b429-803611ac02a7)  
![Alive Nodes](https://github.com/user-attachments/assets/2d0648c2-aa0f-479f-971a-ed67a5984656)  
![PDR](https://github.com/user-attachments/assets/4187088c-d3ae-480a-af7c-32aa0a5069eb)  
![Average Delay](https://github.com/user-attachments/assets/f44ebd38-a23a-4a41-91d5-d5b1041ff193)  
![Loss Function](https://github.com/user-attachments/assets/6faeb539-c821-4a8e-8336-d4aa05d05f6c)  
![Training Time Comparison](https://github.com/user-attachments/assets/21cefc85-773a-4358-b4d1-129aa9fcb916)  
![Final Comparison](https://github.com/user-attachments/assets/d445dd8c-14d9-401f-b732-8b2849b26f2d)

---

##  Conclusion

The **Hybrid MARL** approach (DQN + MBO + BDO) significantly outperforms standalone methods in smart city load balancing simulations. Key improvements include:

-  Reduced energy consumption  
-  Higher Packet Delivery Ratio (PDR)  
-  Better training efficiency and reward convergence  
-  Lower average delays

This strategy is ideal for real-time, scalable smart city systems.
