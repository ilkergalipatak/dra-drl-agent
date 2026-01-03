# LLM-Hybrid DRA Feature Selection Framework

This project implements a novel **Hyper-Heuristic Feature Selection** framework that combines the **Divine Religions Algorithm (DRA)** with **Deep Reinforcement Learning (DRL)** and **Large Language Models (LLMs)**.

The "Prophet" (DRL Agent) learns to dynamically switch between standard DRA operators and LLM-generated heuristics based on the optimization state (stagnation, diversity, etc.), aiming to find optimal feature subsets for machine learning tasks.

## 🚀 Key Features

*   **Hybrid Optimization:** Combines meta-heuristic search (DRA) with AI-guided decision making (DRL/LLM).
*   **DRL Prophet Agent:** A PPO-based agent (using Stable-Baselines3) that observes the search landscape and decides when to intervene.
*   **LLM Heuristics:** Uses **Qwen-Coder** (via Ollama) to generate Python code for novel optimization operators on-the-fly.
*   **Safe Execution:** All LLM-generated code runs in a secure, restricted sandbox environment.
*   **Parallel Training:** Supports multi-process training for the DRL agent (`SubprocVecEnv`).
*   **Live Monitoring:** Streamlit-based dashboard to watch LLM interactions and search progress in real-time.
*   **Comprehensive Benchmark:** Automated benchmarking suite comparing Native classifiers, Pure DRA, and Hybrid DRA across multiple datasets.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ilkergalipatak/dra-drl-agent.git
    cd dra-drl-agent
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Setup Ollama (LLM Server):**
    *   Install [Ollama](https://ollama.ai).
    *   Pull the model: `ollama pull qwen2.5-coder:32b` (or your preferred model).
    *   Start the server with parallel support:
        ```bash
        # Windows
        start_ollama.bat
        ```

4.  **Configuration:**
    *   Copy `config.example.yaml` to `config.yaml`.
    *   Update `config.yaml` with your Ollama URL and API key (if needed).

## 🏃‍♂️ Usage

### 1. Train the DRL Agent (Prophet)
Train the PPO agent to learn the optimization strategy.
```bash
python train_prophet.py --dataset heart_failure --timesteps 50000 --n-envs 4
```

### 2. Run Benchmarks
Compare the performance of the trained agent against baselines.
```bash
python run_full_benchmark.py --datasets heart_failure breast_cancer --runs 5
```

### 3. Live Monitoring
Monitor LLM prompts and responses during training/benchmarking.
```bash
streamlit run monitor_llm.py
```

### 4. EDA (Exploratory Data Analysis)
Generate summary statistics for all datasets.
```bash
python run_eda.py
```

## 📂 Project Structure

```
├── models/             # Saved PPO models and checkpoints
├── datasets/           # CSV datasets
├── logs/               # Log files (LLM history, training logs)
├── results/            # Benchmark results (CSV, Plots)
├── src/
│   ├── agents/         # DRL Agent (PPO) and Environment
│   ├── core/           # DRA Algorithm implementation
│   ├── data/           # Data loading and preprocessing
│   ├── evaluation/     # Fitness functions (KNN, SVM, etc.)
│   ├── llm/            # LLM interaction and Code Sandbox
│   └── hybrid_dra_fs.py # Main Integration Class
├── config.yaml         # Configuration file
├── train_prophet.py    # Training script
└── run_full_benchmark.py # Benchmarking script
```

## 🤖 How It Works

1.  **Observation:** The DRL agent observes the searching population's state (Diversity, Stagnation Counter, Best Fitness).
2.  **Action:** The agent selects an action:
    *   *Action 0:* Continue with standard DRA Operator.
    *   *Action 1:* Request a "Exploration" heuristic from LLM.
    *   *Action 2:* Request a "Exploitation" heuristic from LLM.
3.  **LLM Generation:** If an LLM action is chosen, the system constructs a prompt describing the current problem state and asks Qwen to write a Python function to perturb the solution.
4.  **Execution:** The generated code is executed in a sandbox and applied to the population.
5.  **Reward:** The agent receives a reward based on the improvement in fitness (accuracy/feature reduction).

## 📝 License

This project is licensed under the MIT License.
