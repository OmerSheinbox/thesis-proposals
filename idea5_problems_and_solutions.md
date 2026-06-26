# Idea 5: Problems, Failure Modes, and Paper-Worthy Solutions

> Report written after implementing and debugging two prototype versions
> ([agentic_tool_pomdp_v2.py](file:///home/osha/.gemini/antigravity/scratch/pomdp-llm-integration/agentic_tool_pomdp_v2.py),
> [agentic_tool_pomdp_v3.py](file:///home/osha/.gemini/antigravity/scratch/pomdp-llm-integration/agentic_tool_pomdp_v3.py))
> and grounded in the relevant literature (LATS, ToolTree, POMCP, SAGE-Agent, DESPOT, CVaR-MDPs).

---

## Summary Table

| # | Problem | Severity | Creative Solution | Novel? |
|---|---------|----------|-------------------|--------|
| 1 | Unknown tool space | 🔴 Critical | LLM-as-tool-discoverer + hierarchical abstraction | ✅ High |
| 2 | Exponential branching | 🔴 Critical | Irreversibility-stratified pruning | ✅ High |
| 3 | Surrogate world model faithfulness | 🔴 Critical | Uncertainty-gated simulation with hallucination detection | ✅ High |
| 4 | EVPI-stalling (never commits) | 🟠 High | Adaptive ε-annealing + entropy floor commit | ✅ Medium |
| 5 | Continuous / combinatorial argument space | 🟠 High | Argument-space embedding + latent action clustering | ✅ High |
| 6 | Non-stationarity of user intent | 🟠 High | Drift-aware belief with forgetting factor | ✅ Medium |
| 7 | Credit assignment over multi-tool trajectories | 🟠 High | Hindsight relabeling + counterfactual side-effect traces | ✅ High |
| 8 | Observation aliasing (distinct states → same obs) | 🟡 Medium | Multi-modal observation fusion | ✅ Medium |
| 9 | Computational cost at inference time | 🟡 Medium | Amortized MCTS via policy distillation | ✅ Medium |
| 10 | Mixed reversibility / undo semantics | 🟡 Medium | Reversibility certificates + undo-tree planning | ✅ High |

---

## Problem 1 — Unknown Tool Space

### The Problem

The POMDP formulation requires the action space $\mathcal{A}$ to be fully enumerated at planning time. In practice, real agents have access to thousands of tools (APIs, shell commands, code interpreters, file operations) and the full set is never known in advance. A coding agent hitting a new repository may discover new tools (Makefiles, custom scripts, Docker commands) that were not in its training distribution. This is not just an engineering inconvenience — it mathematically breaks the MCTS tree structure, since you cannot UCB1-select over an action space you cannot enumerate.

Observed in v2: the prototype hardcoded 9 actions. Real agentic settings have 50–5000.

### Creative Solution: **Hierarchical Tool Abstraction with LLM-as-Tool-Discoverer**

Instead of enumerating tools at planning time, treat tool discovery itself as a POMDP action. Define a meta-action `discover_tools(context)` that queries an LLM to generate a *compressed schema* of available tools from docstrings, API specs, or README files. This schema is not a flat list of tools but a **semantic hierarchy**:

```
Level 0 (abstract): "file_mutation" | "information_gathering" | "external_comm"
Level 1 (category): "delete" | "rename" | "compress" | ...
Level 2 (concrete): "delete_temp::recursive" | "delete_all::confirmed" | ...
```

The MCTS operates on Level 0 and Level 1 nodes first, with Level 2 only expanded when the belief is concentrated enough to warrant specificity. This is essentially a **hierarchical POMDP (HPOMDP)** where macro-actions compress the exponentially large flat action space.

**Paper angle**: Formalize tool discovery as a Bayesian active learning problem over an action-space prior $P(\mathcal{A})$. Show that the optimal discovery strategy maximises the *expected reduction in planning regret* (not just entropy), and that the hierarchical abstraction achieves polynomial instead of exponential branching.

---

## Problem 2 — Exponential Branching of the Decision Tree

### The Problem

With $N$ tools and $K$ argument variants each, and a planning horizon of $d$ steps, the full tree has $(NK)^d$ nodes. With $N=50, K=5, d=6$: that's $250^6 \approx 2.4 \times 10^{14}$ nodes — completely intractable, even with MCTS pruning. Standard UCB1 exploration is designed for bandit problems with small action sets; in high-dimensional action spaces it does not concentrate visits meaningfully within a realistic simulation budget.

Observed in v3: even with only 9 actions and 800 MCTS simulations, the tree explored fewer than 100 unique trajectories.

### Creative Solution: **Irreversibility-Stratified Progressive Widening**

The key insight is that not all branches deserve equal exploration budget. An action's *reversibility* defines how catastrophic premature exploration of that branch is. Propose **Irreversibility-Stratified Progressive Widening (ISPW)**:

1. The tree expands reversible actions first and exhaustively.
2. Irreversible actions are introduced only after a threshold visit count on the parent node: $N_\text{parent} \geq \lceil C / \text{reversibility}_a \rceil$, where low-reversibility actions require exponentially more parent visits before their subtrees are opened.
3. Within each reversibility stratum, a secondary $\epsilon$-greedy exploration bonus proportional to the *expected information gain* of the action (not just UCT score) further concentrates budget on informative branches.

**Paper angle**: Prove that ISPW achieves the same asymptotic optimality as standard MCTS (converges to optimal policy as $N_\text{sims} \to \infty$) while reducing the *expected number of catastrophic rollouts* by a factor of $O(\text{reversibility}_{\min}^{-d})$. Benchmark on ToolTree and LATS environments.

---

## Problem 3 — Surrogate World Model Faithfulness

### The Problem

The MCTS requires simulating the *consequences* of tool calls without executing them in the real world. This demands a surrogate world model $\hat{T}(s'|s,a)$ — typically an LLM prompted to predict what would happen if the tool were called. This is the single most dangerous failure mode of the entire architecture.

If the surrogate hallucinates that `delete_all_files` returns `{"status": "success", "files_deleted": 0}` (i.e., falsely predicts a no-op), the MCTS will assign that action high value and the agent will commit to it in the real world. The resulting catastrophe is not recoverable.

Two classes of failure exist:
- **Over-optimistic hallucination**: surrogate predicts success where reality returns an error or destructive effect.
- **Silent incorrect state transition**: surrogate transitions to a plausible-sounding but factually wrong world state, and the downstream MCTS rollout is corrupted silently.

### Creative Solution: **Uncertainty-Gated Simulation with Ensemble Disagreement Scoring**

Run $M$ independent surrogate models (e.g., $M=5$ LLM calls with temperature>0) for each simulated tool invocation. Compute the *ensemble disagreement score* $\Delta_a$ as the JS-divergence between the predicted observation distributions:

$$\Delta_a = \text{JSD}(\hat{Z}_1(o|s,a), \hat{Z}_2(o|s,a), \ldots, \hat{Z}_M(o|s,a))$$

When $\Delta_a$ exceeds a threshold $\delta$, the MCTS assigns that subtree a *hallucination discount*: its value is multiplied by $(1 - \Delta_a)$, causing the planner to treat it as an unreliable branch. Additionally, any subtree rooted at a high-$\Delta_a$ node triggers the agent to take a real-world *information-gathering* action (a safe probe) rather than commit.

**Paper angle**: Demonstrate that ensemble disagreement is a well-calibrated proxy for surrogate model error on a set of API mock environments. Show that hallucination-discounted MCTS achieves strictly lower catastrophic action rate than undiscounted MCTS, with only a linear overhead of $O(M)$ in simulation cost.

---

## Problem 4 — EVPI-Stalling (The Agent Never Commits)

### The Problem

This was the primary failure mode of v2: all 3 scenarios failed because EVPI never dropped below the commit threshold ε=0.08. The agent kept calling `list_files` in a loop, gathering diminishing marginal information, never committing.

Root cause: the EVPI formula $\text{EVPI} = E_s[\max_a R(s,a)] - \max_a E_s[R(s,a)]$ measures the *theoretical* gap between oracle performance and current best action. Even with 90% belief in the correct intent, two competing actions with similar E[R] scores (e.g., `del_temp::recursive` vs `list::deep`) create a persistent EVPI floor because the oracle picks the dominant action per *state*, not per *belief*.

### Creative Solution: **Adaptive ε-Annealing + Entropy Floor Commit**

Replace the static commit threshold with a two-mechanism system:

1. **Entropy floor commit**: commit immediately when $H(b_t) < H_\text{floor}$ bits, regardless of EVPI. This captures the case where the belief is extremely concentrated even if EVPI is not near zero (as happens when multiple good actions compete).

2. **ε-annealing**: decrease $\varepsilon_t = \varepsilon_0 \cdot \lambda^t$ after each information-gathering step. This models the *diminishing returns* of further questioning — each additional clarification question costs user patience and should require a higher justification bar. After $K$ info-gathering steps, the agent is forced to commit regardless.

3. **Information-gain ratio stopping**: stop when $\frac{\text{EIG}_t}{\text{EIG}_1} < \tau$, where EIG$_t$ is the expected information gain of the best available information-gathering action. When information gathering becomes $\tau$-fraction as useful as it was initially, further questioning is not worth it.

**Paper angle**: Prove that ε-annealing with geometric decay guarantees termination in $O(\log(1/\varepsilon_0))$ steps. Show that this is asymptotically optimal in the sense that for any fixed horizon $T$, the annealing schedule minimises expected regret subject to the constraint that the agent always terminates.

---

## Problem 5 — Continuous / Combinatorial Argument Space

### The Problem

Even a single tool like `write_file(path, content)` has an argument space that is effectively infinite — `path` is a string over a filesystem, `content` is arbitrary text. The MCTS treats each (tool, argument) pair as a distinct action, so the branching factor is not just the number of tools but the number of *instantiated argument combinations*, which is uncountable.

In v3, we discretized arguments into fixed variants (`del_temp::current`, `del_temp::recursive`). In reality, you cannot enumerate all possible values of a path argument.

### Creative Solution: **Latent Action Clustering via Tool Embedding Space**

Learn an embedding $\phi: \mathcal{A} \to \mathbb{R}^d$ of the argument space using contrastive training on tool call traces: two argument instantiations are close in embedding space if they produce similar environment transitions and similar downstream rewards. At planning time:

1. Project the current belief $b_t$ through a *belief-conditioned query encoder* to produce a query vector $q(b_t)$.
2. Retrieve the $K$ nearest neighbors in the tool embedding space as candidate actions.
3. MCTS only expands these $K$ candidates, reducing the effective branching factor from infinite to $K$ (typically 5–20).

This is closely related to retrieval-augmented planning, but applied to the *action space* rather than the *knowledge space*.

**Paper angle**: Show that embedding-space action retrieval achieves near-oracle performance (performance using the full action space) with only $K=10$ candidates per node, using a benchmark with 500+ tool variants. The key insight is that reward-contrastive embeddings capture *functional equivalence* — two argument instantiations that differ syntactically but are semantically similar (different filenames in the same directory) cluster together.

---

## Problem 6 — Non-Stationarity of User Intent

### The Problem

All current POMDP formulations (including ours) assume the user's intent $s$ is fixed and hidden — a static latent variable. In reality, user intent drifts during a long conversation. A user who starts with "clean up the folder" may, after seeing the file listing, change their mind and decide to archive instead of delete. The belief update formula $b_{t+1}(s) \propto Z(o|s,a) \cdot b_t(s)$ assumes stationarity: once an intent has low probability, it stays low.

This creates a dangerous failure mode: the agent becomes overconfident in a stale intent and executes an irreversible action based on a belief that reflects the user's old, superseded goal.

### Creative Solution: **Drift-Aware Belief with Intent Transition Model**

Extend the POMDP with an explicit *intent transition model* $T_I(s'|s)$ that models the probability of the user's intent changing between turns. This can be learned from conversation datasets:

$$b_{t+1}(s') \propto Z(o|s',a) \cdot \sum_{s} T_I(s'|s) \cdot b_t(s)$$

The transition matrix $T_I$ encodes domain knowledge: e.g., `list_only → delete_temp` is plausible (user looks, then decides to delete) but `delete_all → list_only` is unlikely. This is structurally equivalent to a *forgetting factor* in Bayesian filtering.

Additionally, track a *belief freshness score* $\tau_t$ that decays as the conversation lengthens without new observations confirming the prior belief. When $\tau_t < \tau_\text{min}$, trigger a mandatory re-confirmation action before any irreversible commit.

**Paper angle**: Collect a dataset of multi-turn agentic conversations where user intent is annotated at each turn. Fit $T_I$ from this data. Show that drift-aware POMDP achieves significantly lower catastrophic-action rate on long conversations compared to the static belief model, with the improvement scaling with conversation length.

---

## Problem 7 — Credit Assignment Over Multi-Tool Trajectories

### The Problem

In a 10-step trajectory ending in a catastrophic deletion, which intermediate actions were responsible? The MCTS assigns cumulative discounted reward to each trajectory, but this does not tell us *which specific tool call* introduced the critical error. Without this, the value function cannot learn to avoid the sequence `[list_files → ask_clarification → delete_all]` when the clarification answer was ambiguous.

This is the agentic analogue of the temporal credit assignment problem, but harder: the side effects of tool calls are *real-world* (not simulated), so you cannot run the trajectory again with a different action at step 3 to see what would have happened.

### Creative Solution: **Counterfactual Side-Effect Traces (CoSET)**

After a trajectory completes (success or failure), generate *counterfactual traces*: for each action $a_t$ in the trajectory, use the surrogate world model to simulate what would have happened if a different action $a'_t$ had been taken at step $t$, holding all other steps constant. This produces a set of counterfactual trajectories with known outcomes.

Use these counterfactuals as training signal: the *counterfactual regret* at step $t$ is $R_t^{\text{CF}} = V(\text{real}_{t:T}) - V(\text{counterfactual}_{t:T})$, and this is used as a step-specific value target to train the MCTS value estimator.

This is related to hindsight relabeling (HER) but applied to irreversible side-effect modeling rather than goal-conditioned RL.

**Paper angle**: This is highly novel. No existing work applies counterfactual reasoning to the credit assignment problem in multi-tool agentic trajectories. Show that CoSET training significantly improves the MCTS value function's accuracy at identifying the "point of no return" in long tool-call trajectories, measured on a synthetic benchmark with known ground-truth causal graphs of side effects.

---

## Problem 8 — Observation Aliasing

### The Problem

Different true intents can produce identical observations from a given tool. If `list_files` returns `{"files": ["data.csv", "notes.txt"]}`, this observation provides almost no information about whether the user's intent is `delete_all`, `compress`, or `rename`. The Bayesian update barely moves the belief. The agent is stuck in a region of observation space that is highly ambiguous, and no amount of calling `list_files` will help.

### Creative Solution: **Multi-Modal Observation Fusion**

Instead of a single observation per tool call, design tools that return *multi-modal* observations combining syntactic content (file list), metadata (file sizes, modification dates), and contextual signals (previous tool call history, user's conversation tone). The observation function becomes:

$$Z(o_{\text{multi}}|s,a) = Z_\text{content}(o_c|s,a) \cdot Z_\text{meta}(o_m|s,a) \cdot Z_\text{context}(o_x|s,a)$$

Under conditional independence, this factored observation can be evaluated efficiently. More importantly, observation modalities that are *aliased* on one channel (file content) may be *discriminative* on another (file sizes reveal whether the user is working with large media files, suggesting a compress intent).

Additionally, introduce *active observation design*: rather than calling a fixed tool, let the MCTS select *which observation modality to request* from a tool. This is Bayesian Experimental Design (BED) applied to the observation function.

**Paper angle**: Show that multi-modal observation fusion reduces the number of tool calls needed before commit by a factor of 2–5x across a benchmark of aliasing-heavy scenarios, compared to single-modal observation.

---

## Problem 9 — Computational Cost at Inference Time

### The Problem

Running 800 MCTS simulations with depth-5 rollouts (as in v3) took ~7 seconds per scenario on a standard CPU. In real agentic settings, each rollout requires one or more LLM forward passes as the surrogate world model. At GPT-4 latency (~1s per call), 800 simulations × 5 depth = 4000 LLM calls = **~1.1 hours per decision step**. Completely intractable.

### Creative Solution: **Amortized MCTS via Rapid Policy Distillation**

Train a lightweight *amortized policy network* $\pi_\phi(a|b_t)$ that, given the current particle filter belief, directly predicts the MCTS winner *without running the tree search*. This network is trained from MCTS traces collected offline:

1. Run full MCTS on a large corpus of synthetic planning problems to generate $(b_t, a^*_t)$ pairs.
2. Distill into a small feedforward net $\pi_\phi$ mapping belief encodings to action distributions.
3. At inference, use $\pi_\phi$ directly (< 1ms), falling back to full MCTS only when $\pi_\phi$'s confidence is low (entropy of its output distribution > threshold).

This mirrors AlphaZero's use of a fast policy network to guide tree search, but in the belief-MDP setting.

**Paper angle**: Show that the amortized policy achieves 95%+ of the MCTS-optimal action rate while reducing inference latency by 3–4 orders of magnitude. The key contribution is the *confidence-based fallback*: demonstrating that the regime where $\pi_\phi$ is uncertain (and MCTS is needed) is precisely the regime where EVPI is high — meaning the expensive MCTS is only invoked when it matters most.

---

## Problem 10 — Mixed Reversibility / Undo Semantics

### The Problem

The binary irreversible/reversible distinction is a massive oversimplification. In practice:
- `rename_file` is reversible if the original name is stored — but only for a limited time before a downstream process overwrites the backup.
- `send_email` is irreversible to the recipient, but a "recall" API may exist with ~60% success rate.
- `delete_file` is irreversible on some filesystems, but recoverable via trash-bin or snapshot for the next 30 days.

The prototype models reversibility as a static scalar, but real reversibility is *time-dependent*, *probabilistic*, and *context-dependent*.

### Creative Solution: **Reversibility Certificates + Undo-Tree Planning**

Extend each tool with a *reversibility certificate*: a machine-readable declaration of the form:

```json
{
  "undo_action": "restore_file",
  "undo_success_prob": 0.95,
  "undo_window_seconds": 2592000,
  "undo_cost": 0.1
}
```

The POMDP planner uses these certificates to construct an *undo-tree*: a parallel data structure tracking, for each committed action, the available undo operations, their success probabilities, and their expiry times. This converts the irreversible action into a *conditionally reversible* action with known rollback probability.

The value function is then computed as:

$$V^\pi(b) = \mathbb{E}\left[R + \gamma \cdot \left[(1-\rho) \cdot V^\pi(b') + \rho \cdot V^\pi(b^\text{undo})\right]\right]$$

where $\rho$ is the undo success probability. This forces the planner to value having a rollback path available, naturally causing it to prefer actions with high-probability undo certificates over structurally identical actions without them.

**Paper angle**: This is highly practical and publishable. Define a formal schema for reversibility certificates, build a registry of 100+ common tools annotated with certificates, and show that undo-tree planning significantly reduces the frequency of non-recoverable failures compared to binary irreversibility models, with only a constant overhead per planning step.

---

## Cross-Cutting Verdict

The problems above can be grouped into two fundamental research threads:

**Thread A — Tractability**: Problems 2, 5, 9 are all facets of the same core challenge: the action space is too large for standard planning. The unified solution is a hierarchy of amortization: *embedding-space action clustering* compresses the argument space, *irreversibility-stratified pruning* focuses the MCTS budget, and *policy distillation* removes the tree search altogether at inference time. A single paper unifying these three contributions around a formal *regret-complexity tradeoff theorem* would be publishable at NeurIPS or ICML.

**Thread B — Safety under Uncertainty**: Problems 3, 4, 6, 7, 10 are all about the agent committing to irreversible actions prematurely, under a wrong or stale belief. The unified solution is a *safety layer* that sits above the MCTS: reversibility certificates (Problem 10) + drift-aware belief (Problem 6) + hallucination-discounted simulation (Problem 3) + counterfactual credit assignment (Problem 7) together constitute a comprehensive *safe agentic planning* framework. This is highly publishable at ICLR or ICRA, where the emphasis on real-world deployability is strong.

> [!TIP]
> If forced to pick **one idea** for a single paper: **Problem 7 (CoSET — Counterfactual Side-Effect Traces)** is the most novel, the most directly tied to the unique challenges of irreversible tool use, and has no direct predecessor in the literature. It can be framed as a general method applicable to any agentic POMDP and benchmarked on both synthetic and real tool-execution environments.
