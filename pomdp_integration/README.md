# POMDP Integration: The Two Paradigms

This folder contains the complete body of work for integrating Partially Observable Markov Decision Processes (POMDPs) into LLM agent loops. 

To better understand the field and the novel contributions of this thesis, the code and notes have been neatly separated into **two distinct spaces**, representing a fundamental conceptual divide:

## 1. `01_user_wishes_and_intent/`
**Uncertainty over *User Intent, Emotions, and Wishes***

This space represents the inward-facing uncertainty. The agent assumes it understands the world perfectly, but the "hidden state" of the POMDP is what the user actually wants or feels. The agent must decode ambiguous prompts, evaluate urgency, and decide when it's safe to act vs. when it must ask for clarification.

**Key Concepts Explored Here:**
- **PTBench:** The benchmarking framework testing LLMs against ambiguous and deceptive tasks.
- **POMDP_V3 (Agentic Tool POMDP):** The system that uses MCTS, Particle Filters, and Conditional Value at Risk (CVaR) to prevent catastrophic irreversible actions.
- **EVPI:** Expected Value of Perfect Information used to gate clarification questions.
- **Dec-POMDP (Multi-Agent):** A swarm of specialized agents (like AffectAgent and UrgencyAgent) maintaining private Bayesian beliefs about user intent and emotions, coordinating via KL-divergence thresholds.

## 2. `02_tool_usage_and_world_state/`
**Uncertainty over *Tool Reliability and World State***

This space represents the outward-facing uncertainty. The agent assumes it knows exactly what the user wants, but the "hidden state" is the *true state of the world* (e.g., the file system, APIs, databases) after multiple tool calls. Tool outputs are treated as clues rather than absolute truth.

**Key Concepts Explored Here:**
- **TASM (Tool-use As SLAM):** Framing LLM tool execution as Simultaneous Localization and Mapping.
- **The Core Problem:** Dealing with unreliability, noisy evidence, stale reads, buffered writes, and phantom data.
- **TASM Experiment 1:** Particle filters running over file system configurations, proving that mapping the environment probabilistically achieves 100% accuracy in highly noisy environments where standard LLM point-estimate reasoning catastrophically fails.