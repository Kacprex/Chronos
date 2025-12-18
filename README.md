# ♟️ Chronos

**Chronos** is a complete **AlphaZero-style chess engine implemented from scratch**.

The project combines **neural networks**, **Monte Carlo Tree Search (MCTS)**, **supervised learning**, and **reinforcement learning through self-play**.  
The system learns chess **without handcrafted evaluation rules**, improving purely through data and search.

> This is **not a toy project**.  
> Chronos is a **research-grade implementation** designed to be modular, restartable, observable, and safe to run on consumer hardware.

---

## 🧠 Core Concepts

Chronos is built around the following principles:

- Neural network with **policy and value heads**
- **Monte Carlo Tree Search** guided by the neural network
- **Supervised learning** from strong human games
- **Reinforcement learning** via self-play
- Continuous evaluation against **Stockfish**
- Explicit **diversity monitoring** to prevent policy collapse

---

## ✨ Features

### AlphaZero-Style MCTS
- Dirichlet noise at the root
- Temperature scheduling
- Opening randomness

### Training & Learning
- Supervised learning pipeline with **shard-based datasets**
- Self-play engine producing training samples and PGNs
- Reinforcement learning loop

### Evaluation & Safety
- AI vs AI and Stockfish vs AI match generation
- PGN export for all played games
- Diversity testing for policy health
- Model promotion system (**latest vs best**)
- Safe, RAM-aware memory usage
- GPU acceleration (CUDA supported)

---

## 🗂️ Project Structure

### Top-Level Files
hub.py # Central command-line interface
config.py # Global configuration, paths, and defaults
data/ # Datasets, training shards, PGNs

css
Skopiuj kod

### Source Code (`src/`)
src/
├── nn/ # Neural network architecture & board/move encoding
├── mcts/ # Monte Carlo Tree Search implementation
├── selfplay/ # Self-play engine and PGN encoding
├── training/ # Supervised and reinforcement learning pipelines
├── evaluation/ # Stockfish evaluation and diversity testing
└── utils/ # Logging, device handling, shared utilities

yaml
Skopiuj kod

---

## ▶️ How to Use

Run the project via:

```bash
python hub.py
This launches an interactive menu with options to:

Run self-play followed by reinforcement learning

Generate self-play games only

Play AI vs AI and export PGNs

Play Stockfish vs AI and export PGNs

Run diversity tests

Run Stockfish evaluation on the model

Evaluate and promote models

Exit

When prompted, always provide:

Number of games

MCTS simulations per move

Stockfish depth (when applicable)

🔄 Training Workflow
Recommended workflow for new experiments:

Run supervised learning to initialize the model

Verify basic behavior using AI vs AI games

Run diversity tests to ensure exploration

Start self-play game generation

Train using reinforcement learning

Periodically evaluate against Stockfish

Promote stronger models automatically

All steps are restartable and safe to interrupt.

🧪 Diversity & Stability
Chronos includes an explicit diversity testing module that analyzes PGN files and reports:

Game result distribution

Color balance

Game length statistics

Opening diversity

This prevents silent failure modes such as:

Repetitive openings

Deterministic move loops

Policy collapse

💻 Hardware Requirements
Designed and tested for consumer hardware:

CPU: Modern multi-core consumer CPUs

GPU: NVIDIA GPU with CUDA support (optional)

RAM: Moderate usage via shard-based loading

Parallelization is intentionally conservative to avoid memory spikes.

📈 Current Status
Core engine: ✅ complete and functional

MCTS: ✅ stable with exploration mechanisms

Self-play: ✅ running correctly

Reinforcement learning: ✅ operational

Evaluation tools: ✅ operational

Strength is expected to increase with additional self-play and training time.

🎯 Goals
Chronos aims to:

Demonstrate a full AlphaZero-style system end-to-end

Serve as a learning and research platform

Remain understandable and maintainable

Avoid hidden magic or black-box behavior

📜 License
This project is intended for educational and research purposes.
License details can be added as needed.

📝 Final Note
Chronos represents a serious implementation of a modern chess AI system.