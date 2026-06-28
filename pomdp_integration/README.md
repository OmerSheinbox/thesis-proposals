# POMDP Integration: The Two Paradigms

This folder contains the complete body of work for integrating Partially Observable Markov Decision Processes (POMDPs) into LLM agent loops. 

To better understand the field and the novel contributions of this thesis, the code and notes have been neatly separated into **two distinct spaces**, representing a fundamental conceptual divide:

## 1. `01_intent_uncertainty_ptbench/`
**The Classic Approach (Uncertainty over *User Intent*)**

This space represents the current edge of the literature, where the "hidden state" of the POMDP is what the user actually wants. The tools are assumed to return perfect truth, but the agent must decide when it's safe to act vs. when it must ask for clarification.

**Key Concepts Explored Here:**
- **PTBench:** The benchmarking framework testing LLMs against ambiguous and deceptive tasks.
- **POMDP_V3 (Agentic Tool POMDP):** The system that uses MCTS, Particle Filters, and Conditional Value at Risk (CVaR) to prevent catastrophic irreversible actions.
- **EVPI:** Expected Value of Perfect Information used to gate clarification questions.
- **Dec-POMDP (Multi-Agent):** A swarm of agents maintaining private Bayesian beliefs about user intent, coordinating via KL-divergence thresholds.

## 2. `02_world_state_uncertainty_tasm/`
**The Novel Lacuna (Uncertainty over *World State*)**

This space represents the structural gap found in all prior work. Here, the "hidden state" is NOT the user's intent, but rather the *true state of the world* (e.g., the file system) after multiple tool calls. 

**Key Concepts Explored Here:**
- **TASM (Tool-use As SLAM):** Framing LLM tool execution as Simultaneous Localization and Mapping.
- **The Core Problem:** Tool outputs are treated as *noisy evidence* (stale reads, buffered writes, phantom data) rather than absolute truth.
- **TASM Experiment 1:** Particle filters running over file system configurations, proving that maintaining a belief distribution over world states achieves 100% accuracy in highly noisy environments where standard LLM point-estimate reasoning catastrophically degrades.