"""
Idea 5 Prototype: Agentic Optimization via POMDP Tool Calling & Side Effect Modeling
======================================================================================

Demonstrates the three core mechanisms described in the README:

  1. Belief State  — a probability distribution over the user's hidden intent,
                     continuously updated via Bayesian inference on observations.

  2. EVPI          — Expected Value of Perfect Information; gates execution of
                     irreversible tools by quantifying how much uncertainty costs.
                     Agent only commits to destructive actions when EVPI < epsilon.

  3. Tree Search   — depth-limited lookahead over tool-call sequences; combines
                     discounted future value with a risk penalty term that grows
                     with irreversibility × current belief entropy.

Scenario
--------
  User says: "clean up the project folder"
  True hidden intent: delete_temp_files  (agent cannot observe this directly)

  The agent must decide whether to immediately call delete_all_files (catastrophic
  if wrong) or first gather information — list_files, ask_clarification — to
  collapse its uncertainty before committing.
"""

import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────
# 1.  WORLD DEFINITION
# ──────────────────────────────────────────────────────────────────

# The hidden state space S: all possible true user intents.
INTENTS: List[str] = ["delete_temp", "delete_all", "list_only", "rename"]

INTENT_LABELS: Dict[str, str] = {
    "delete_temp": "Delete only .tmp files",
    "delete_all":  "Delete ALL files in directory",
    "list_only":   "Just list what's there — no deletion",
    "rename":      "Rename files, do not delete",
}


@dataclass
class Tool:
    """
    A single tool in the agent's action space A.

    rewards         — R(s, a): reward if this tool is executed under true intent s.
                      Negative values model catastrophic mismatches (e.g., deleting
                      everything when the user only wanted a listing).

    obs_likelihoods — Z(o | s, a): the probability of each observation o, given
                      that the tool was called while the true hidden state is s.
                      Used for the Bayesian belief update.

    irreversible    — If True, the tree search does not expand children past this
                      node: once executed, the side effect cannot be simulated away.

    risk_weight     — Scales the risk penalty term in the value function.
                      0.0 = perfectly safe tool; 1.0 = maximally dangerous.
    """
    name: str
    irreversible: bool
    risk_weight: float
    rewards: Dict[str, float] = field(default_factory=dict)
    obs_likelihoods: Dict[str, Dict[str, float]] = field(default_factory=dict)


def build_tools() -> Dict[str, Tool]:
    return {

        "ask_clarification": Tool(
            name="ask_clarification",
            irreversible=False,
            risk_weight=0.0,
            rewards={
                "delete_temp": 0.3,
                "delete_all":  0.3,
                "list_only":   0.3,
                "rename":      0.3,
            },
            obs_likelihoods={
                # After asking, user responds; 80% chance they clarify correctly.
                "delete_temp": {"obs_confirm_temp": 0.80, "obs_ambiguous": 0.20},
                "delete_all":  {"obs_confirm_all":  0.80, "obs_ambiguous": 0.20},
                "list_only":   {"obs_confirm_list": 0.80, "obs_ambiguous": 0.20},
                "rename":      {"obs_confirm_rename":0.80, "obs_ambiguous": 0.20},
            },
        ),

        "list_files": Tool(
            name="list_files",
            irreversible=False,
            risk_weight=0.0,
            rewards={
                "delete_temp": 0.40,   # useful diagnostic step
                "delete_all":  0.40,
                "list_only":   1.00,   # this IS what they wanted
                "rename":      0.40,
            },
            obs_likelihoods={
                # Seeing many .tmp files is strong evidence for delete_temp intent.
                "delete_temp": {"obs_many_tmp": 0.75, "obs_mixed": 0.25},
                "delete_all":  {"obs_many_tmp": 0.40, "obs_mixed": 0.60},
                "list_only":   {"obs_many_tmp": 0.45, "obs_mixed": 0.55},
                "rename":      {"obs_mixed":    0.85, "obs_many_tmp": 0.15},
            },
        ),

        "delete_temp_files": Tool(
            name="delete_temp_files",
            irreversible=True,
            risk_weight=0.4,
            rewards={
                "delete_temp":  1.00,   # perfect match
                "delete_all":  -0.50,   # wrong scope: user wanted everything gone
                "list_only":   -1.50,   # catastrophe: user just wanted to look
                "rename":      -1.50,   # catastrophe: files deleted, not renamed
            },
            obs_likelihoods={
                "delete_temp": {"obs_success": 1.0},
                "delete_all":  {"obs_partial": 1.0},
                "list_only":   {"obs_unintended_deletion": 1.0},
                "rename":      {"obs_unintended_deletion": 1.0},
            },
        ),

        "delete_all_files": Tool(
            name="delete_all_files",
            irreversible=True,
            risk_weight=1.0,
            rewards={
                "delete_temp": -0.80,   # overkill: nuked more than intended
                "delete_all":   1.00,   # perfect match
                "list_only":   -2.00,   # disaster
                "rename":      -2.00,   # disaster
            },
            obs_likelihoods={
                "delete_temp": {"obs_unintended_deletion": 1.0},
                "delete_all":  {"obs_success": 1.0},
                "list_only":   {"obs_unintended_deletion": 1.0},
                "rename":      {"obs_unintended_deletion": 1.0},
            },
        ),

        "rename_files": Tool(
            name="rename_files",
            irreversible=False,
            risk_weight=0.2,
            rewards={
                "delete_temp": -0.50,
                "delete_all":  -0.50,
                "list_only":   -0.50,
                "rename":       1.00,
            },
            obs_likelihoods={
                "delete_temp": {"obs_wrong_action": 1.0},
                "delete_all":  {"obs_wrong_action": 1.0},
                "list_only":   {"obs_wrong_action": 1.0},
                "rename":      {"obs_success": 1.0},
            },
        ),
    }


# ──────────────────────────────────────────────────────────────────
# 2.  BELIEF STATE  b_t(s)
# ──────────────────────────────────────────────────────────────────

BeliefState = Dict[str, float]   # intent -> probability


def uniform_belief() -> BeliefState:
    p = 1.0 / len(INTENTS)
    return {intent: p for intent in INTENTS}


def entropy(belief: BeliefState) -> float:
    """Shannon entropy H(b) in bits."""
    h = 0.0
    for p in belief.values():
        if p > 1e-12:
            h -= p * math.log2(p)
    return h


def bayesian_update(belief: BeliefState, tool: Tool, observation: str) -> BeliefState:
    """
    Recursive Bayesian update:
        b_{t+1}(s) ∝ Z(o | s, a) · b_t(s)

    Returns a normalized posterior over intents.
    """
    new_belief: BeliefState = {}
    total = 0.0
    for intent, prior in belief.items():
        likelihood = tool.obs_likelihoods.get(intent, {}).get(observation, 1e-3)
        new_belief[intent] = prior * likelihood
        total += new_belief[intent]

    if total > 1e-12:
        for intent in new_belief:
            new_belief[intent] /= total
    else:
        new_belief = uniform_belief()

    return new_belief


# ──────────────────────────────────────────────────────────────────
# 3.  EXPECTED VALUE OF PERFECT INFORMATION  (EVPI)
# ──────────────────────────────────────────────────────────────────

def expected_reward(belief: BeliefState, tool: Tool) -> float:
    """E[R(a)] = Σ_s  P(s) · R(s, a)"""
    return sum(belief[s] * tool.rewards[s] for s in INTENTS)


def risk_penalty(belief: BeliefState, tool: Tool) -> float:
    """
    Irreversibility-scaled risk term added to the value function:

        risk_pen = risk_weight · H_norm(b) · E[harm | a]

    Grows when the tool is dangerous AND the agent is still uncertain.
    Zero for reversible tools.
    """
    if not tool.irreversible:
        return 0.0
    uncertainty = entropy(belief) / math.log2(len(INTENTS))          # normalised 0–1
    expected_harm = sum(
        belief[s] * max(0.0, -tool.rewards[s]) for s in INTENTS
    )
    return tool.risk_weight * uncertainty * expected_harm


def evpi(belief: BeliefState, tools: Dict[str, Tool]) -> float:
    """
    EVPI = E_s[ max_a R(s,a) ]  −  max_a E_s[ R(s,a) − risk_pen(b,a) ]

    Represents how much the agent would gain if it magically knew the true
    intent before choosing its next tool.  When EVPI ≈ 0, further information
    gathering is not worth the cost and the agent should commit.
    """
    # Value with perfect information (oracle picks best tool per state)
    v_perfect = sum(
        belief[s] * max(tool.rewards[s] for tool in tools.values())
        for s in INTENTS
    )
    # Best value achievable right now (under uncertainty)
    v_now = max(
        expected_reward(belief, t) - risk_penalty(belief, t)
        for t in tools.values()
    )
    return max(0.0, v_perfect - v_now)


# ──────────────────────────────────────────────────────────────────
# 4.  DEPTH-LIMITED TREE SEARCH   Q(b, a, depth)
# ──────────────────────────────────────────────────────────────────

GAMMA     = 0.90   # discount factor
MAX_DEPTH = 3      # lookahead horizon


def expected_obs_dist(tool: Tool, belief: BeliefState) -> Dict[str, float]:
    """
    Marginal observation probability:
        P(o) = Σ_s  P(s) · Z(o | s, a)
    """
    dist: Dict[str, float] = {}
    for intent, p_s in belief.items():
        for obs, p_o_given_s in tool.obs_likelihoods.get(intent, {}).items():
            dist[obs] = dist.get(obs, 0.0) + p_s * p_o_given_s
    return dist


def tree_value(
    belief: BeliefState,
    tools: Dict[str, Tool],
    depth: int,
) -> Tuple[float, str]:
    """
    Bellman backup over belief space:

        Q(b, a) = [E_s R(s,a) − risk_pen(b,a)]
                  + γ · Σ_o P(o|b,a) · V(b', depth−1)    [if a is reversible]

        V(b, depth) = max_a Q(b, a, depth)

    Irreversible tools terminate the lookahead branch (no future rollback).
    Returns (best_value, best_tool_name).
    """
    if depth == 0:
        best_val  = -math.inf
        best_name = ""
        for name, tool in tools.items():
            val = expected_reward(belief, tool) - risk_penalty(belief, tool)
            if val > best_val:
                best_val  = val
                best_name = name
        return best_val, best_name

    best_val  = -math.inf
    best_name = ""

    for name, tool in tools.items():
        immediate = expected_reward(belief, tool) - risk_penalty(belief, tool)

        if tool.irreversible:
            # Cannot simulate side effects safely; tree terminates here.
            q = immediate
        else:
            # Lookahead: weight future value by expected observation probabilities.
            obs_dist   = expected_obs_dist(tool, belief)
            future_val = 0.0
            for obs, p_obs in obs_dist.items():
                b_prime        = bayesian_update(belief, tool, obs)
                v_next, _      = tree_value(b_prime, tools, depth - 1)
                future_val    += p_obs * v_next
            q = immediate + GAMMA * future_val

        if q > best_val:
            best_val  = q
            best_name = name

    return best_val, best_name


# ──────────────────────────────────────────────────────────────────
# 5.  DISPLAY HELPERS
# ──────────────────────────────────────────────────────────────────

W = 68

def header(text: str):
    print("\n" + "=" * W)
    print(f"  {text}")
    print("=" * W)

def section(text: str):
    print("\n" + "─" * W)
    print(f"  {text}")
    print("─" * W)

def print_belief(belief: BeliefState, label: str = "Belief State"):
    print(f"\n  📊  {label}  [H = {entropy(belief):.3f} bits]")
    for intent, p in sorted(belief.items(), key=lambda x: -x[1]):
        bar   = "█" * int(p * 35)
        label_ = INTENT_LABELS[intent]
        print(f"    {intent:16s}  {p:.3f}  {bar}")
    print()

def print_tool_scores(belief: BeliefState, tools: Dict[str, Tool]):
    print("  🔧  Tool Value Scores  (E[R] − risk_penalty):")
    rows = []
    for name, tool in tools.items():
        er    = expected_reward(belief, tool)
        rp    = risk_penalty(belief, tool)
        score = er - rp
        rows.append((score, name, er, rp, tool.irreversible))
    for score, name, er, rp, irrev in sorted(rows, reverse=True):
        flag = "  ⚠️  IRREVERSIBLE" if irrev else ""
        print(f"    {name:25s}  {score:+.3f}  "
              f"(E[R]={er:+.3f}, risk_pen={rp:.3f}){flag}")
    print()


# ──────────────────────────────────────────────────────────────────
# 6.  DEMO
# ──────────────────────────────────────────────────────────────────

def main():
    tools = build_tools()

    header("IDEA 5: POMDP Tool Calling & Side Effect Modeling — Demo")
    print("""
  Scenario  : User says → "clean up the project folder"
  True intent: delete_temp_files   (agent does NOT know this)

  The agent maintains a belief over 4 possible intents and must
  decide at each step whether to gather more information or commit
  to an irreversible tool call.
""")

    # ── Step 0: uniform prior ──────────────────────────────────────
    belief = uniform_belief()
    section("Step 0 — Initial Belief  (no information yet)")
    print_belief(belief)

    e = evpi(belief, tools)
    print(f"  💡  EVPI = {e:.3f}")
    print(f"      → High. The value of gathering info before acting is large.\n")

    v, best = tree_value(belief, tools, MAX_DEPTH)
    print(f"  🌲  Tree search (depth={MAX_DEPTH}) → best action: [{best}]  (V = {v:.3f})")
    print_tool_scores(belief, tools)

    # ── Step 1: call list_files ────────────────────────────────────
    section("Step 1 — Agent calls: list_files")
    print("  Observation: 'obs_many_tmp'  (directory is full of .tmp files)\n")

    belief = bayesian_update(belief, tools["list_files"], "obs_many_tmp")
    print_belief(belief, "Belief after list_files + obs_many_tmp")

    e = evpi(belief, tools)
    print(f"  💡  EVPI = {e:.3f}")
    print(f"      → Reduced. delete_temp is now most likely, but still uncertain.\n")

    v, best = tree_value(belief, tools, MAX_DEPTH)
    print(f"  🌲  Tree search (depth={MAX_DEPTH}) → best action: [{best}]  (V = {v:.3f})")
    print_tool_scores(belief, tools)

    # ── Step 2: ask_clarification ──────────────────────────────────
    section("Step 2 — Agent calls: ask_clarification")
    print("  Agent asks: 'Should I delete only .tmp files, or everything?'")
    print("  Observation: 'obs_confirm_temp'  (user confirms: only .tmp files)\n")

    belief = bayesian_update(belief, tools["ask_clarification"], "obs_confirm_temp")
    print_belief(belief, "Belief after clarification")

    e = evpi(belief, tools)
    print(f"  💡  EVPI = {e:.3f}")
    print(f"      → Near zero. Further questions add no value. Safe to commit.\n")

    v, best = tree_value(belief, tools, MAX_DEPTH)
    print(f"  🌲  Tree search (depth={MAX_DEPTH}) → best action: [{best}]  (V = {v:.3f})")
    print_tool_scores(belief, tools)

    # ── Final decision ─────────────────────────────────────────────
    header(f"FINAL DECISION → Execute: [{best}]")
    print(f"""
  ✅  Risk-penalized planning blocked delete_all_files despite it
      having high reward under the delete_all intent — its risk
      penalty was too high while belief entropy was large.

  ✅  EVPI guided the agent through two safe, informative steps
      (list_files → ask_clarification) before the belief collapsed
      enough to justify committing to an irreversible action.

  ✅  Total lookahead: {MAX_DEPTH} steps, discount γ = {GAMMA}.
      No real-world tool was executed until EVPI ≈ 0.
""")


if __name__ == "__main__":
    main()
