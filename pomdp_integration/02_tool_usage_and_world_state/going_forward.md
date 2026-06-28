# Going Forward: The Master's Thesis Blueprint

This document summarizes the strategic decisions made for taking the TASM (Tool-use As SLAM) concept and elevating it into a top-tier CS conference submission (e.g., NeurIPS, ICLR).

## 1. The Core Idea: TASM (Tool-use As SLAM)
We are pivoting away from focusing on *User Intent* and instead focusing exclusively on *World State Uncertainty*. We are treating LLM tool use as a noisy state estimation problem, applying the mathematics of Simultaneous Localization and Mapping (SLAM) to digital tool execution.

## 2. The Evaluation Domain: Real-World Benchmarks
We abandon the synthetic 5-file benchmark. Instead, we will prove the architecture on a standard, highly respected benchmark like **WebArena** or **Mind2Web**, demonstrating that TASM beats standard architectures (like ReAct or LATS) when dealing with real-world unreliability like stale DOMs and delayed APIs.

## 3. The Architecture: Latent Space Particle Filter
To handle the massive dimensionality of a web browser DOM or API state, we take the deep learning path. We will use an encoder (like a frozen LLM embedding) to compress raw observations into dense vectors, running the entire Bayesian particle filter mathematically within a continuous latent space.

## 4. The Data Pipeline: Existing Offline Trajectories
To solve the dependency of needing a "Transition Model" (to predict how an action changes a dense vector), we will leverage the massive datasets of offline trajectories already provided by WebArena or Mind2Web. We will use these logs to pre-train a lightweight neural "world model."

---

**The Final Pitch:** 
We are building a *Latent Space Particle Filter* that tracks the true state of complex web environments, pre-trained on *Mind2Web/WebArena offline trajectories*, to prove that modeling tool-use as *noisy SLAM* prevents catastrophic failures in LLM agents.
