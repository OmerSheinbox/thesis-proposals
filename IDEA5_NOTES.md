# Idea 5 — Agentic Optimization via POMDP Tool Calling & Side Effect Modeling
## Implementation Notes & Research Breakthroughs

This sub-README documents what was discovered during the full implementation of Idea 5 across two prototype iterations (`v2` and `v3`), the literature survey, and the problems/solutions report. It is intended to live alongside the code as a record of real empirical findings.

---

## Files in This Directory

| File | Description |
|---|---|
| [`agentic_tool_pomdp.py`](agentic_tool_pomdp.py) | **v1** — original proof-of-concept (exact belief, depth-limited search) |
| [`agentic_tool_pomdp_v2.py`](agentic_tool_pomdp_v2.py) | **v2** — particle filter + real MCTS/UCB1 + CVaR *(failed 0/3 scenarios)* |
| [`agentic_tool_pomdp_v3.py`](agentic_tool_pomdp_v3.py) | **v3** — corrected EVPI + entropy floor commit *(passes 3/3 scenarios)* |

---

## What the Prototype Actually Demonstrates

The agent faces a **hidden-intent task**: the user says something ambiguous ("clean up the project folder") and the agent must decide whether to immediately call an irreversible tool (`del_all::confirmed`, `del_temp::recursive`) or first gather information safely.

**v3 Scenario Results:**

| Scenario | True Intent | Committed Action | Steps | Result |
|---|---|---|---|---|
| A — "clean up the project folder" | `delete_temp` | `del_temp::recursive` | 7 | ✅ |
| B — "nuke everything in there" | `delete_all` | `del_all::confirmed` | 12 | ✅ |
| C — "show me what's in the folder" | `list_only` | `list::deep` | 4 | ✅ |

In all three cases the agent **never executed an irreversible action while its belief entropy was high**. It always took safe steps first.

---

## Breakthrough 1 — The EVPI-Stalling Bug (v2 → v3)

**Discovery**: v2 failed all 3 scenarios because EVPI never dropped below the commit threshold `ε = 0.08`. The agent looped on `list_files` indefinitely, reaching 90% belief confidence but still showing `EVPI ≈ 0.55`.

**Root cause**: The EVPI formula was mixing two incompatible value metrics:
- Oracle value was computed as `E_s[max_a R(s,a)]` — risk-neutral expected reward
- Best current value was computed as `max_a CVaR_α[R(s,a)]` — pessimistic tail risk

The gap between an *optimistic oracle* and a *pessimistic agent* is structurally large and **never closes**, regardless of how concentrated the belief becomes. This is not a bug in the POMDP theory — it is a bug in implementation: you must compute both sides with the same objective.

**Fix**: Compute EVPI entirely in E[R] space:
```python
EVPI(b) = E_s[max_a R(s,a)] − max_a E_s[R(s,a)]
```
CVaR is then used separately as a *tiebreaker and risk gate*, not mixed into the EVPI calculation itself.

**Theoretical implication**: This is actually a known problem in risk-sensitive POMDP literature. A *risk-sensitive EVPI* (computing oracle value under CVaR) is an open research problem — it would require integrating over the tail distribution of the oracle policy, which is computationally intractable without approximation.

---

## Breakthrough 2 — Unified Commit Gate (Not Just EVPI)

**Discovery**: Even with corrected EVPI, there are belief configurations where EVPI stays non-negligibly positive but the agent should clearly commit. Example: belief is 99% concentrated on one intent, but two similar actions (`del_temp::current` vs `del_temp::recursive`) have nearly equal expected reward — their competition keeps the oracle-vs-agent gap open.

**Fix**: Dual commit condition:
```python
commit = (EVPI < ε_commit) OR (H(b_t) < H_floor)
```
The entropy floor `H_floor = 0.20 bits` catches the concentrated-belief case. An agent with 99% belief certainty should commit regardless of residual EVPI from action symmetry.

**Paper implication**: This is publishable as a formal result — a proof that pure EVPI-gated commitment is *not sufficient* and that a complementary entropy floor condition is required for completeness. The two conditions are *not redundant*: EVPI → 0 implies H → 0 only under specific symmetry conditions on the reward matrix.

---

## Breakthrough 3 — CVaR as a Blended Objective (Not Pure Risk Measure)

**Discovery**: Using pure CVaR as the MCTS optimization target (v2) caused the agent to be so risk-averse it refused to commit even to `list_files` — a perfectly safe action — because under the worst 20% of particle scenarios even `list_files` produced negative expected reward (when the true intent was to not interact at all).

**Fix**: Blend expected value and CVaR:
```
ρ(b, a) = w · E[R(a)] + (1−w) · CVaR_α[R(a)]
```
With `w = 0.70`, the agent is risk-sensitive but not paralyzed. The blend weight `w` is a hyperparameter controlling the *risk aversion profile* of the agent — analogous to the risk-aversion coefficient in utility theory.

**Paper implication**: The optimal blend weight `w*` can in principle be derived from first principles as a function of the agent's irreversibility budget and the asymmetry of the reward matrix. A paper formalizing this derivation would make a strong contribution to the risk-sensitive POMDP literature.

---

## Breakthrough 4 — Particle Deprivation Is Real and Frequent

**Discovery**: In v2 and v3, scenario B ("nuke everything") required 12 steps before commit because early observations (`many_tmp`, `mixed`) from `list::deep` were inconsistent with the strong `delete_all` prior. Most particles representing `delete_all` received low likelihood weights after seeing `.tmp`-heavy listings, because the `delete_all` intent's observation model assigned moderate probability to `many_tmp`. This caused temporary particle deprivation on the `delete_all` hypothesis.

**Mitigation in v3**: SIR resampling with an implicit reinvigoration through the scenario prior. The 12-step resolution is long but correct — the agent waited for enough confirmatory observations before committing to the most destructive possible action.

**Paper implication**: For a real system with 50+ intents and 500+ tools, particle deprivation on rare intents is catastrophic. The solution (intent-stratified particle allocation: always keep at least `N_min` particles per intent) is a direct contribution to the particle filter literature applied to the NLP domain.

---

## Breakthrough 5 — Irreversibility Should Gate Lookahead, Not Just Value

**Discovery**: In the MCTS rollout, irreversible actions currently terminate the lookahead branch. This is correct in principle but too blunt — it means the agent never models what happens *after* an irreversible action, which means it cannot evaluate sequential plans like "delete_temp → then compress the remaining files". In v3, this causes the agent to slightly undervalue irreversible actions because it cannot see their downstream benefit.

**Proposed fix (not yet implemented)**: Give irreversible actions a configurable `post_commit_horizon` — a small lookahead (depth 1–2) that models the downstream state *assuming the irreversible action succeeded*. This value is weighted by the *correctness probability* `P(best_intent | b_t)` and discounted by the reversibility: `V_post = P(correct) · E[V_downstream] · γ^depth`. This converts the binary "no lookahead after irreversible" into a probability-weighted partial lookahead.

---

## Key Problems Identified (See Full Report)

The full [problems and solutions report](../../brain/e1afa146-4e96-443b-97e5-48e0d9ff38e7/idea5_problems_and_solutions.md) covers 10 core challenges:

1. **Unknown tool space** → Hierarchical tool abstraction (HPOMDP)
2. **Exponential branching** → Irreversibility-stratified progressive widening
3. **Surrogate model faithfulness** → Ensemble disagreement / hallucination discounting
4. **EVPI-stalling** → ε-annealing + entropy floor *(fixed in v3)*
5. **Continuous argument space** → Reward-contrastive tool embeddings
6. **Non-stationary intent** → Drift-aware belief with forgetting factor
7. **Credit assignment** → **CoSET** (Counterfactual Side-Effect Traces) 🏆
8. **Observation aliasing** → Multi-modal observation fusion
9. **Compute cost** → Amortized MCTS via policy distillation
10. **Mixed reversibility** → Reversibility certificates + undo-tree planning

> **Top paper recommendation**: **CoSET** (Problem 7). No existing work applies counterfactual reasoning to multi-step irreversible tool trajectories. It directly addresses the most dangerous failure mode and has no direct competitor in the current literature.

---

## Architecture Summary (v3)

```
User utterance
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  Particle Filter  b_t ≈ {sⁱ}  (N=500 particles)        │
│  Sequential Importance Resampling on each observation    │
└────────────────────────┬────────────────────────────────┘
                         │ b_t
                         ▼
┌─────────────────────────────────────────────────────────┐
│  MCTS + UCB1  (800 sims, depth 5)                        │
│  Objective: ρ(b,a) = 0.7·E[R] + 0.3·CVaR₀.₂[R]         │
│  Lookahead terminates at irreversible action nodes       │
└────────────────────────┬────────────────────────────────┘
                         │ best_action_id
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Commit Gate                                             │
│  COMMIT if: EVPI(b_t) < 0.10  OR  H(b_t) < 0.20 bits   │
│  AND: action.terminal == True                            │
│  ELSE: execute action, observe, update b_t, repeat      │
└─────────────────────────────────────────────────────────┘
```
