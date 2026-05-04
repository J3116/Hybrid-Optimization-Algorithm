# Orca Predation Optimization Algorithm
### **Intelligent Systems Engineering | Metaheuristic Research**

---

## **1. Overview**
This repository contains a Python implementation of a nature-inspired metaheuristic optimization algorithm based on the social hierarchy and collaborative hunting behaviors of **Orcas** (*Orcinus orca*). Developed as part of the fourth-year **Intelligent Systems Engineering** curriculum at **Middle East University (MEU)**, this project investigates the efficiency of simulated predatory tactics in solving high-dimensional search space problems.

The algorithm specifically focuses on improving convergence accuracy and search efficiency through a hybrid model that integrates elements of **Wolf Optimization**.

---

## **2. Mathematical Framework**
The optimization model mimics the pod's movement to identify the global optimum of a given objective function.

### **Objective Function**
The goal is to minimize the function $f(\mathbf{x})$ within the search space $\mathbb{R}^n$:
$$ \min f(\mathbf{x}), \quad \mathbf{x} = [x_1, x_2, \dots, x_n] \in \mathbb{R}^n $$

### **Movement Logic**
The position of a search agent $i$ at iteration $t+1$ is updated based on its proximity to the current best solution and the predatory vector $\vec{P}$:
$$ \mathbf{x}_{i}(t+1) = \mathbf{x}_{best}(t) + \alpha \cdot \vec{P} $$

> **Note:** Here, $\mathbf{x}_{best}(t)$ represents the optimal position identified by the pod at iteration $t$, and $\alpha$ is the step size coefficient derived from the hunting strategy.

---

## **3. Features**
*   **Hybrid Logic**: Combines Orca Predation strategies with Wolf Optimization for enhanced search resilience
*   **Visual Analysis**: Includes real-time generation of convergence plots using Matplotlib to track performance.
*   **OOP Architecture**: Built using Object-Oriented Programming principles, making the code modular and easy to integrate into larger intelligent systems.

---

## **4. Technical Stack**
*   **Environment**: Developed using **Microsoft Visual Studio 2022** (Solution Version 17.14).
*   **Language**: Python 3.x.
*   **Libraries**:
    *   **NumPy**: High-performance matrix calculus and numerical analysis.
    *   **Matplotlib**: Data visualization and convergence tracking.

---

## **5. Project Structure**
The repository follows a professional Visual Studio solution layout:

```bash
OrcaPredation/
├── OrcaPredation.sln          # Primary Visual Studio Solution
└── OrcaPredation/             # Source Directory
    ├── OrcaPredation.py       # Core Algorithm Implementation
    └── OrcaPredation.pyproj   # Python Project Configuration


git clone [https://github.com/J3116/OrcaPredation.git](https://github.com/J3116/OrcaPredation.git)
2.  **Open in IDE**: 
    Launch `OrcaPredation.sln` using **Visual Studio 2022**
3.  **Execute**:
    Ensure the required libraries are installed and run `OrcaPredation.py`.

---

## **About the Author**
**Jana Alsuheimat**[cite: 1]
*   **Education**: 4th-year Intelligent Systems Engineering Student at **Middle East University (MEU)**.
*   **Expertise**: Robotics, AI Optimization, and Full-Stack Development (ASP.NET Core/Flutter).
*   **Location**: Amman, Jordan
