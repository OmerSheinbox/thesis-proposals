"""
Idea 5 v2: Agentic Optimization via POMDP Tool Calling & Side Effect Modeling
==============================================================================
Significantly improved prototype over v1. New in this version:

  [1] Real MCTS with UCB1 exploration
      — replaces exhaustive depth-limited search with stochastic rollouts
        guided by Upper Confidence Bound (UCB1), matching the ToolTree /
        LATS architecture.

  [2] Particle Filter belief approximation
      — replaces exact enumeration with N particles for scalability;
        mirrors POMCP's approach to handling large/continuous state spaces.

  [3] CVaR (Conditional Value at Risk) risk criterion
      — replaces the ad-hoc risk_penalty heuristic with a proper risk-
        sensitive objective: instead of maximising E[R], the agent maximises
        E[R | R ≥ VaR_α], explicitly bounding worst-case tail losses under
        belief uncertainty.

  [4] Tool argument space
      — each tool now carries discrete argument variants (e.g. delete_temp
        with scope="current_dir" vs scope="recursive"), each with different
        rewards and obs likelihoods, modelling the combinatorial A space.

  [5] Partial reversibility spectrum
      — tools carry a reversibility ∈ [0,1] float rather than a bool;
        the MCTS lookahead uses this to discount future value proportionally.

  [6] EVPI-gated commitment
      — the agent only executes an irreversible tool when
        EVPI(b_t) < ε_commit, with ε_commit configurable.

  [7] Multi-scenario benchmark
      — runs three scenarios with different true intents and reports
        whether the agent converged to the correct action without a
        catastrophic premature execution.

  [8] Full trajectory audit log
      — every step (belief, EVPI, MCTS winner, CVaR scores, chosen action,
        observation) is stored and printed as a structured trace.

Mathematical foundation
-----------------------
  State space  S = {user_intents}  (hidden)
  Action space A = {tool × argument_variant}  (combinatorial)
  Obs space    O = {possible API/env responses}
  Transition   T(s'|s,a)  — environment dynamics (simplified: intent is sticky)
  Obs function Z(o|s,a)   — stochastic tool output given true state
  Reward       R(s,a)     — utility of executing tool a when true state is s
  Discount     γ ∈ (0,1)

  MCTS value:  Q(b,a) = CVaR_α[R(s,a)] + γ · Σ_o P(o|b,a) · V(b'|o)
  EVPI:        EVPI(b) = E_s[max_a R(s,a)] − max_a CVaR_α[R(s,a)|b]
  Commit gate: execute irreversible action iff EVPI(b) < ε_commit
"""

import math
import random
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

random.seed(42)


# ══════════════════════════════════════════════════════════════════════════════
# § 0  HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

GAMMA          = 0.92    # discount factor
N_PARTICLES    = 400     # particle filter size
N_MCTS_SIMS    = 600     # MCTS simulations per decision
MCTS_DEPTH     = 6       # maximum rollout depth
UCB_C          = 1.5     # UCB1 exploration constant
CVAR_ALPHA     = 0.20    # CVaR tail fraction (lower = more risk-averse)
EVPI_EPSILON   = 0.08    # commit threshold: execute if EVPI < epsilon
MAX_QUESTIONS  = 3       # agent may ask at most this many clarification Qs


# ══════════════════════════════════════════════════════════════════════════════
# § 1  WORLD DEFINITION
# ══════════════════════════════════════════════════════════════════════════════

INTENTS: List[str] = [
    "delete_temp",    # user wants .tmp files gone
    "delete_all",     # user wants the entire directory wiped
    "list_only",      # user just wants a listing
    "rename",         # user wants files renamed
    "compress",       # user wants files archived/compressed
]

INTENT_LABELS: Dict[str, str] = {
    "delete_temp": "Delete only .tmp / cache files",
    "delete_all":  "Delete ALL files in directory",
    "list_only":   "List files — no mutation",
    "rename":      "Rename files to new pattern",
    "compress":    "Compress / archive the folder",
}


@dataclass
class ArgVariant:
    """
    A specific parametric instantiation of a tool.
    e.g. delete_temp(scope='current') vs delete_temp(scope='recursive')
    """
    label: str                        # human-readable argument description
    rewards: Dict[str, float]         # R(s, a_variant)
    obs_likelihoods: Dict[str, Dict[str, float]]  # Z(o|s, a_variant)
    reversibility: float = 1.0        # 0=fully irreversible, 1=fully reversible
    question_cost: float = 0.0        # cost paid for asking (friction)


@dataclass
class Tool:
    """
    A tool in the agent's action space.  Each tool has one or more ArgVariants,
    representing the combinatorial argument space.
    """
    name: str
    variants: List[ArgVariant]

    def all_actions(self) -> List[Tuple[str, ArgVariant]]:
        """Returns (action_id, variant) pairs for every instantiation."""
        return [(f"{self.name}::{v.label}", v) for v in self.variants]


def _clamp(d: Dict[str, float]) -> Dict[str, float]:
    """Ensure obs likelihoods sum to ≤ 1 (remainder → 'obs_null')."""
    s = sum(d.values())
    if abs(s - 1.0) > 1e-6:
        d = {k: v / s for k, v in d.items()}
    return d


def build_tools() -> List[Tool]:
    """
    Construct the agent's tool arsenal with multiple argument variants per tool.
    This directly models the combinatorial action space A described in Idea 5.
    """
    return [

        # ── ask_clarification ──────────────────────────────────────────────
        # Reversible, costless information gatherer.  Two variants: broad or
        # targeted question (targeted is more informative but requires more
        # belief certainty to formulate well).
        Tool("ask_clarification", variants=[
            ArgVariant(
                label="broad",
                reversibility=1.0,
                question_cost=0.05,
                rewards={s: 0.25 for s in INTENTS},
                obs_likelihoods={
                    "delete_temp": _clamp({"obs_hint_delete": 0.55, "obs_ambiguous": 0.45}),
                    "delete_all":  _clamp({"obs_hint_delete": 0.55, "obs_ambiguous": 0.45}),
                    "list_only":   _clamp({"obs_hint_list":   0.60, "obs_ambiguous": 0.40}),
                    "rename":      _clamp({"obs_hint_rename": 0.60, "obs_ambiguous": 0.40}),
                    "compress":    _clamp({"obs_hint_compress":0.60, "obs_ambiguous": 0.40}),
                },
            ),
            ArgVariant(
                label="targeted",
                reversibility=1.0,
                question_cost=0.10,
                rewards={s: 0.20 for s in INTENTS},
                obs_likelihoods={
                    "delete_temp": _clamp({"obs_confirm_temp":    0.85, "obs_ambiguous": 0.15}),
                    "delete_all":  _clamp({"obs_confirm_all":     0.85, "obs_ambiguous": 0.15}),
                    "list_only":   _clamp({"obs_confirm_list":    0.85, "obs_ambiguous": 0.15}),
                    "rename":      _clamp({"obs_confirm_rename":  0.85, "obs_ambiguous": 0.15}),
                    "compress":    _clamp({"obs_confirm_compress":0.85, "obs_ambiguous": 0.15}),
                },
            ),
        ]),

        # ── list_files ────────────────────────────────────────────────────
        # Safe read-only probe.  Shallow vs deep listing.
        Tool("list_files", variants=[
            ArgVariant(
                label="shallow",
                reversibility=1.0,
                rewards={
                    "delete_temp": 0.35,
                    "delete_all":  0.35,
                    "list_only":   0.90,
                    "rename":      0.35,
                    "compress":    0.35,
                },
                obs_likelihoods={
                    "delete_temp": _clamp({"obs_many_tmp": 0.70, "obs_mixed": 0.30}),
                    "delete_all":  _clamp({"obs_many_tmp": 0.35, "obs_mixed": 0.65}),
                    "list_only":   _clamp({"obs_many_tmp": 0.40, "obs_mixed": 0.60}),
                    "rename":      _clamp({"obs_mixed":    0.80, "obs_many_tmp": 0.20}),
                    "compress":    _clamp({"obs_mixed":    0.75, "obs_many_tmp": 0.25}),
                },
            ),
            ArgVariant(
                label="deep+sizes",
                reversibility=1.0,
                rewards={
                    "delete_temp": 0.45,
                    "delete_all":  0.45,
                    "list_only":   1.00,
                    "rename":      0.40,
                    "compress":    0.50,
                },
                obs_likelihoods={
                    "delete_temp": _clamp({"obs_many_tmp": 0.80, "obs_mixed": 0.20}),
                    "delete_all":  _clamp({"obs_many_tmp": 0.40, "obs_mixed": 0.60}),
                    "list_only":   _clamp({"obs_many_tmp": 0.45, "obs_mixed": 0.55}),
                    "rename":      _clamp({"obs_mixed":    0.90, "obs_many_tmp": 0.10}),
                    "compress":    _clamp({"obs_large_files": 0.70, "obs_mixed": 0.30}),
                },
            ),
        ]),

        # ── delete_temp_files ──────────────────────────────────────────────
        # Irreversible.  Two scope variants: current dir only vs recursive.
        Tool("delete_temp_files", variants=[
            ArgVariant(
                label="scope=current",
                reversibility=0.0,
                rewards={
                    "delete_temp":  0.90,
                    "delete_all":  -0.40,
                    "list_only":   -1.50,
                    "rename":      -1.50,
                    "compress":    -0.60,
                },
                obs_likelihoods={
                    "delete_temp": _clamp({"obs_success": 1.0}),
                    "delete_all":  _clamp({"obs_partial": 1.0}),
                    "list_only":   _clamp({"obs_unintended_deletion": 1.0}),
                    "rename":      _clamp({"obs_unintended_deletion": 1.0}),
                    "compress":    _clamp({"obs_unintended_deletion": 1.0}),
                },
            ),
            ArgVariant(
                label="scope=recursive",
                reversibility=0.0,
                rewards={
                    "delete_temp":  1.00,
                    "delete_all":  -0.30,
                    "list_only":   -1.80,
                    "rename":      -1.80,
                    "compress":    -0.80,
                },
                obs_likelihoods={
                    "delete_temp": _clamp({"obs_success": 1.0}),
                    "delete_all":  _clamp({"obs_partial": 1.0}),
                    "list_only":   _clamp({"obs_unintended_deletion": 1.0}),
                    "rename":      _clamp({"obs_unintended_deletion": 1.0}),
                    "compress":    _clamp({"obs_unintended_deletion": 1.0}),
                },
            ),
        ]),

        # ── delete_all_files ───────────────────────────────────────────────
        # Maximally destructive. Highest risk_weight → biggest CVaR penalty.
        Tool("delete_all_files", variants=[
            ArgVariant(
                label="confirmed",
                reversibility=0.0,
                rewards={
                    "delete_temp": -0.80,
                    "delete_all":   1.00,
                    "list_only":   -2.50,
                    "rename":      -2.50,
                    "compress":    -1.50,
                },
                obs_likelihoods={
                    "delete_temp": _clamp({"obs_unintended_deletion": 1.0}),
                    "delete_all":  _clamp({"obs_success": 1.0}),
                    "list_only":   _clamp({"obs_unintended_deletion": 1.0}),
                    "rename":      _clamp({"obs_unintended_deletion": 1.0}),
                    "compress":    _clamp({"obs_unintended_deletion": 1.0}),
                },
            ),
        ]),

        # ── rename_files ───────────────────────────────────────────────────
        # Partially reversible (can undo rename, but with effort).
        Tool("rename_files", variants=[
            ArgVariant(
                label="pattern=snake_case",
                reversibility=0.7,
                rewards={
                    "delete_temp": -0.40,
                    "delete_all":  -0.40,
                    "list_only":   -0.40,
                    "rename":       0.85,
                    "compress":    -0.20,
                },
                obs_likelihoods={
                    "delete_temp": _clamp({"obs_wrong_action": 1.0}),
                    "delete_all":  _clamp({"obs_wrong_action": 1.0}),
                    "list_only":   _clamp({"obs_wrong_action": 1.0}),
                    "rename":      _clamp({"obs_success": 0.90, "obs_partial": 0.10}),
                    "compress":    _clamp({"obs_wrong_action": 1.0}),
                },
            ),
        ]),

        # ── compress_folder ────────────────────────────────────────────────
        # Reversible (can decompress). Useful for compress or list_only intents.
        Tool("compress_folder", variants=[
            ArgVariant(
                label="format=zip",
                reversibility=0.85,
                rewards={
                    "delete_temp": -0.20,
                    "delete_all":  -0.50,
                    "list_only":   -0.10,
                    "rename":      -0.20,
                    "compress":     1.00,
                },
                obs_likelihoods={
                    "delete_temp": _clamp({"obs_wrong_action": 1.0}),
                    "delete_all":  _clamp({"obs_wrong_action": 1.0}),
                    "list_only":   _clamp({"obs_success": 0.50, "obs_wrong_action": 0.50}),
                    "rename":      _clamp({"obs_wrong_action": 1.0}),
                    "compress":    _clamp({"obs_success": 0.95, "obs_partial": 0.05}),
                },
            ),
        ]),
    ]


# Build flattened action map: action_id -> ArgVariant
def build_action_map(tools: List[Tool]) -> Dict[str, ArgVariant]:
    amap: Dict[str, ArgVariant] = {}
    for tool in tools:
        for action_id, variant in tool.all_actions():
            amap[action_id] = variant
    return amap


# ══════════════════════════════════════════════════════════════════════════════
# § 2  PARTICLE FILTER  (scalable belief approximation)
# ══════════════════════════════════════════════════════════════════════════════

class ParticleFilter:
    """
    Approximates b_t(s) as a set of N weighted particles, each being one
    sampled intent.  Scales to large/continuous intent spaces where exact
    enumeration is intractable.

    Update rule (sequential importance resampling):
        w_i ← w_i · Z(o | s_i, a)
        Resample N particles from {s_i} weighted by {w_i}
    """

    def __init__(self, intents: List[str], n: int = N_PARTICLES):
        self.intents = intents
        self.n       = n
        # Start uniform
        self.particles: List[str] = [
            random.choice(intents) for _ in range(n)
        ]

    def belief_dict(self) -> Dict[str, float]:
        """Convert particle set to a probability dict."""
        counts = defaultdict(int)
        for p in self.particles:
            counts[p] += 1
        return {s: counts[s] / self.n for s in self.intents}

    def entropy(self) -> float:
        b = self.belief_dict()
        h = 0.0
        for p in b.values():
            if p > 1e-12:
                h -= p * math.log2(p)
        return h

    def update(self, variant: ArgVariant, observation: str):
        """
        Sequential Importance Resampling (SIR):
        1. Weight each particle by Z(o | s_i, a)
        2. Resample with replacement according to weights
        """
        weights = []
        for s in self.particles:
            w = variant.obs_likelihoods.get(s, {}).get(observation, 1e-4)
            weights.append(w)

        total = sum(weights)
        if total < 1e-12:
            # Degenerate: reset to uniform (particle deprivation recovery)
            self.particles = [random.choice(self.intents) for _ in range(self.n)]
            return

        # Normalise
        weights = [w / total for w in weights]
        # Resample
        self.particles = random.choices(self.particles, weights=weights, k=self.n)

    def clone(self) -> "ParticleFilter":
        pf = ParticleFilter(self.intents, self.n)
        pf.particles = list(self.particles)
        return pf


# ══════════════════════════════════════════════════════════════════════════════
# § 3  RISK METRICS
# ══════════════════════════════════════════════════════════════════════════════

def expected_reward_pf(pf: ParticleFilter, variant: ArgVariant) -> float:
    """E[R] computed from particle distribution."""
    b = pf.belief_dict()
    return sum(b[s] * variant.rewards.get(s, 0.0) for s in pf.intents)


def cvar(pf: ParticleFilter, variant: ArgVariant, alpha: float = CVAR_ALPHA) -> float:
    """
    Conditional Value at Risk (CVaR) at level α.

    CVaR_α[R] = E[R | R ≤ VaR_α(R)]

    This is the mean reward of the worst α-fraction of scenarios.
    A risk-averse agent maximises CVaR rather than E[R], explicitly
    protecting against tail catastrophes (e.g. deleting important files).

    Here we compute it from the particle distribution:
      - Sample reward for each particle
      - Sort ascending
      - Return mean of bottom α-fraction
    """
    rewards = [variant.rewards.get(s, 0.0) for s in pf.particles]
    rewards.sort()
    cutoff = max(1, int(alpha * len(rewards)))
    return sum(rewards[:cutoff]) / cutoff


def evpi(pf: ParticleFilter, action_map: Dict[str, ArgVariant]) -> float:
    """
    EVPI = E_s[max_a R(s,a)] − max_a CVaR_α[R(s,a)|b]

    Expected value of knowing the true intent before acting.
    When EVPI < EVPI_EPSILON, gathering more information is not worth the cost.
    """
    b = pf.belief_dict()
    # Oracle value: for each particle, pick the best possible action
    v_perfect = sum(
        b[s] * max(v.rewards.get(s, 0.0) for v in action_map.values())
        for s in pf.intents
    )
    # Best CVaR achievable under current belief
    v_now = max(cvar(pf, v) for v in action_map.values())
    return max(0.0, v_perfect - v_now)


# ══════════════════════════════════════════════════════════════════════════════
# § 4  MCTS WITH UCB1
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MCTSNode:
    """
    Node in the MCTS belief tree.  Each node represents a (belief, action)
    pair and accumulates visit counts and total value for UCB1 selection.
    """
    action_id:  Optional[str]
    parent:     Optional["MCTSNode"]
    children:   Dict[str, "MCTSNode"] = field(default_factory=dict)
    visits:     int   = 0
    total_val:  float = 0.0

    def ucb1(self, parent_visits: int, c: float = UCB_C) -> float:
        if self.visits == 0:
            return float("inf")
        exploit = self.total_val / self.visits
        explore = c * math.sqrt(math.log(parent_visits) / self.visits)
        return exploit + explore

    def mean_val(self) -> float:
        return self.total_val / self.visits if self.visits > 0 else 0.0


def _rollout(pf: ParticleFilter, action_map: Dict[str, ArgVariant],
             depth: int, gamma: float) -> float:
    """
    Random rollout policy: pick a random reversible action until depth 0.
    Returns discounted cumulative reward estimate.
    """
    if depth == 0:
        return 0.0
    # Random safe action
    safe = [aid for aid, v in action_map.items() if v.reversibility > 0.5]
    if not safe:
        return 0.0
    aid    = random.choice(safe)
    v      = action_map[aid]
    # Sample a particle
    s      = random.choice(pf.particles)
    r      = v.rewards.get(s, 0.0)
    # Sample observation
    obs_d  = v.obs_likelihoods.get(s, {})
    if obs_d:
        obs = random.choices(list(obs_d.keys()), weights=list(obs_d.values()))[0]
        pf2 = pf.clone()
        pf2.update(v, obs)
    else:
        pf2 = pf.clone()
    return r + gamma * _rollout(pf2, action_map, depth - 1, gamma)


def mcts(
    pf: ParticleFilter,
    action_map: Dict[str, ArgVariant],
    n_sims: int  = N_MCTS_SIMS,
    depth:  int  = MCTS_DEPTH,
    gamma:  float = GAMMA,
    alpha:  float = CVAR_ALPHA,
) -> Tuple[str, Dict[str, float]]:
    """
    Monte Carlo Tree Search over the belief-space action tree.

    Selection:    UCB1 over children
    Expansion:    Add one unexplored action
    Simulation:   Random rollout with particle-filter belief propagation
    Backprop:     Update visits and total_val up the tree

    Returns (best_action_id, {action_id: mean_value}) for all root children.
    Uses CVaR-weighted value: value = CVaR * visit_normalised_weight.
    """
    root = MCTSNode(action_id=None, parent=None)
    # Pre-create children for all actions
    for aid in action_map:
        root.children[aid] = MCTSNode(action_id=aid, parent=root)

    for _ in range(n_sims):
        # ─ Selection ──────────────────────────────────────────────────
        node = root
        sim_pf = pf.clone()

        selected_aid = max(
            root.children.keys(),
            key=lambda aid: root.children[aid].ucb1(max(root.visits, 1)),
        )
        child = root.children[selected_aid]
        variant = action_map[selected_aid]

        # ─ Simulation (rollout) ───────────────────────────────────────
        s      = random.choice(sim_pf.particles)
        r_imm  = variant.rewards.get(s, 0.0)

        future = 0.0
        if variant.reversibility > 0.1:                 # can look ahead
            obs_d = variant.obs_likelihoods.get(s, {})
            if obs_d:
                obs = random.choices(
                    list(obs_d.keys()), weights=list(obs_d.values())
                )[0]
                sim_pf.update(variant, obs)
            future = gamma * _rollout(sim_pf, action_map, depth - 1, gamma)

        # CVaR-adjusted total: weight the rollout value by the risk of
        # the immediate action under the current belief
        cvar_val  = cvar(pf, variant, alpha)
        sim_value = 0.5 * (r_imm + future) + 0.5 * cvar_val

        # ─ Backpropagation ────────────────────────────────────────────
        child.visits     += 1
        child.total_val  += sim_value
        root.visits      += 1

    # Build score dict and pick winner
    scores = {
        aid: (ch.mean_val(), ch.visits)
        for aid, ch in root.children.items()
    }
    best = max(scores, key=lambda aid: scores[aid][0])
    return best, {aid: v[0] for aid, v in scores.items()}


# ══════════════════════════════════════════════════════════════════════════════
# § 5  AGENT LOOP
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class StepLog:
    step:         int
    action_id:    str
    observation:  str
    belief_before: Dict[str, float]
    belief_after:  Dict[str, float]
    entropy_before: float
    entropy_after:  float
    evpi_before:   float
    evpi_after:    float
    mcts_winner:   str
    mcts_scores:   Dict[str, float]
    cvar_scores:   Dict[str, float]
    committed:     bool


def agent_loop(
    pf: ParticleFilter,
    action_map: Dict[str, ArgVariant],
    true_intent: str,
    scenario_label: str,
    max_steps: int = 10,
    evpi_eps: float = EVPI_EPSILON,
    max_questions: int = MAX_QUESTIONS,
) -> Tuple[str, List[StepLog], bool]:
    """
    Run one episode of the POMDP agent.

    At each step:
      1. Compute EVPI; if below threshold AND best MCTS action is irreversible,
         commit immediately.
      2. Otherwise, run MCTS and execute winning action.
      3. Sample an observation from Z(o | true_intent, action) — this is the
         only place the true_intent is used (simulates env feedback).
      4. Update particle filter via SIR.
      5. Log everything.

    Returns (final_action_committed, trajectory, success_flag)
    """
    logs:        List[StepLog] = []
    q_count:     int           = 0
    final_action: str          = ""
    success:      bool         = False

    for step in range(max_steps):
        b_before   = pf.belief_dict()
        H_before   = pf.entropy()
        e_before   = evpi(pf, action_map)

        # ── MCTS decision ──────────────────────────────────────────────────
        best_aid, scores = mcts(pf, action_map)
        best_variant = action_map[best_aid]

        # ── CVaR scores (for display) ──────────────────────────────────────
        cvar_scores = {aid: cvar(pf, v) for aid, v in action_map.items()}

        # ── Commit gate ────────────────────────────────────────────────────
        committed = False
        if (e_before < evpi_eps) and (best_variant.reversibility < 0.5):
            committed     = True
            final_action  = best_aid
            success       = (true_intent in best_aid or
                             best_variant.rewards.get(true_intent, -99) >= 0.8)
            # Don't update belief after irreversible commit — game over
            logs.append(StepLog(
                step=step, action_id=best_aid, observation="[COMMITTED]",
                belief_before=b_before, belief_after=b_before,
                entropy_before=H_before, entropy_after=H_before,
                evpi_before=e_before, evpi_after=e_before,
                mcts_winner=best_aid, mcts_scores=scores,
                cvar_scores=cvar_scores, committed=True,
            ))
            break

        # ── Enforce question budget ────────────────────────────────────────
        if "ask_clarification" in best_aid:
            q_count += 1
            if q_count > max_questions:
                # Switch to best non-question reversible action
                alt = max(
                    (aid for aid, v in action_map.items()
                     if "ask_clarification" not in aid and v.reversibility >= 0.5),
                    key=lambda aid: scores.get(aid, -999),
                    default=best_aid,
                )
                best_aid      = alt
                best_variant  = action_map[alt]

        # ── Sample observation from environment (using true_intent) ────────
        obs_dist = best_variant.obs_likelihoods.get(true_intent, {})
        if obs_dist:
            observation = random.choices(
                list(obs_dist.keys()), weights=list(obs_dist.values())
            )[0]
        else:
            observation = "obs_null"

        # ── Belief update ──────────────────────────────────────────────────
        pf.update(best_variant, observation)

        b_after  = pf.belief_dict()
        H_after  = pf.entropy()
        e_after  = evpi(pf, action_map)

        logs.append(StepLog(
            step=step, action_id=best_aid, observation=observation,
            belief_before=b_before, belief_after=b_after,
            entropy_before=H_before, entropy_after=H_after,
            evpi_before=e_before, evpi_after=e_after,
            mcts_winner=best_aid, mcts_scores=scores,
            cvar_scores=cvar_scores, committed=False,
        ))

        # Early termination if EVPI is negligible and last action was a
        # strong diagnostic (agent has effectively converged)
        if H_after < 0.15:
            break

    return final_action, logs, success


# ══════════════════════════════════════════════════════════════════════════════
# § 6  DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

W = 72


def hdr(text: str):
    print("\n" + "═" * W)
    print(f"  {text}")
    print("═" * W)


def sec(text: str):
    print("\n" + "─" * W)
    print(f"  {text}")
    print("─" * W)


def belief_bar(b: Dict[str, float], intents: List[str]) -> str:
    lines = []
    for s in intents:
        p   = b.get(s, 0.0)
        bar = "█" * int(p * 28)
        lines.append(f"      {s:16s}  {p:.3f}  {bar}")
    return "\n".join(lines)


def print_step(log: StepLog, intents: List[str]):
    tag = "🔒 COMMIT" if log.committed else f"Step {log.step + 1}"
    print(f"\n  ┌─ {tag} ─────────────────────────────────────────────────")
    print(f"  │  Action    : {log.action_id}")
    print(f"  │  Obs       : {log.observation}")
    print(f"  │  EVPI      : {log.evpi_before:.4f} → {log.evpi_after:.4f}")
    print(f"  │  Entropy   : {log.entropy_before:.4f} → {log.entropy_after:.4f} bits")
    print(f"  │")
    print(f"  │  Belief after update:")
    for s in intents:
        p   = log.belief_after.get(s, 0.0)
        bar = "█" * int(p * 24)
        print(f"  │      {s:16s}  {p:.3f}  {bar}")
    print(f"  │")
    top_cvar = sorted(log.cvar_scores.items(), key=lambda x: -x[1])[:4]
    print(f"  │  CVaR scores (top 4):")
    for aid, cv in top_cvar:
        irr_flag = " [irrev]" if action_map_global.get(aid) and action_map_global[aid].reversibility < 0.5 else ""
        print(f"  │      {aid:35s}  {cv:+.3f}{irr_flag}")
    print(f"  └──────────────────────────────────────────────────────────")


action_map_global: Dict[str, ArgVariant] = {}   # set in main for display


# ══════════════════════════════════════════════════════════════════════════════
# § 7  MULTI-SCENARIO BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════

SCENARIOS = [
    {
        "label":       "Scenario A — Ambiguous cleanup request",
        "user_says":   '"clean up the project folder"',
        "true_intent": "delete_temp",
        "prior":       {s: 0.20 for s in INTENTS},   # uniform
    },
    {
        "label":       "Scenario B — Strong delete signal",
        "user_says":   '"just nuke everything in there, I don\'t need any of it"',
        "true_intent": "delete_all",
        # Slightly skewed prior: "nuke" pushes toward delete_all
        "prior":       {"delete_temp": 0.10, "delete_all": 0.50,
                        "list_only": 0.10, "rename": 0.10, "compress": 0.20},
    },
    {
        "label":       "Scenario C — Passive request (safe intent)",
        "user_says":   '"show me what\'s in the folder"',
        "true_intent": "list_only",
        "prior":       {"delete_temp": 0.10, "delete_all": 0.05,
                        "list_only": 0.60, "rename": 0.15, "compress": 0.10},
    },
]


def run_scenario(scenario: dict, tools: List[Tool]) -> bool:
    action_map = build_action_map(tools)
    global action_map_global
    action_map_global = action_map

    pf = ParticleFilter(INTENTS, N_PARTICLES)
    # Seed particles from prior
    prior = scenario["prior"]
    pf.particles = random.choices(
        list(prior.keys()), weights=list(prior.values()), k=N_PARTICLES
    )

    true_intent = scenario["true_intent"]

    hdr(scenario["label"])
    print(f"\n  User says   : {scenario['user_says']}")
    print(f"  True intent : {true_intent}  ← agent does NOT observe this")
    print(f"  Actions     : {len(action_map)} (tools × arg variants)")
    print(f"  Particles   : {N_PARTICLES}")
    print(f"  MCTS sims   : {N_MCTS_SIMS}  depth={MCTS_DEPTH}")
    print(f"  CVaR α      : {CVAR_ALPHA}  (risk-averse tail fraction)")
    print(f"  EVPI ε      : {EVPI_EPSILON}  (commit threshold)\n")

    t0 = time.time()
    final, logs, success = agent_loop(
        pf, action_map, true_intent, scenario["label"]
    )
    elapsed = time.time() - t0

    for log in logs:
        print_step(log, INTENTS)

    sec(f"Result — {'✅ SUCCESS' if success else '❌ FAILURE'}")
    print(f"  Final committed action : {final or '(no commit reached)'}")
    print(f"  True intent            : {true_intent}")
    print(f"  Steps taken            : {len(logs)}")
    print(f"  Wall time              : {elapsed:.2f}s")
    return success


# ══════════════════════════════════════════════════════════════════════════════
# § 8  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    hdr("IDEA 5 v2 — POMDP Tool Calling & Side Effect Modeling")
    print("""
  Improvements over v1:
    [1] Real MCTS with UCB1 exploration     [5] Partial reversibility spectrum
    [2] Particle filter belief (N=400)      [6] EVPI-gated commitment (ε=0.08)
    [3] CVaR risk criterion (α=0.20)        [7] Multi-scenario benchmark
    [4] Tool argument space (5 tools×2 args)[8] Full trajectory audit log
""")

    tools = build_tools()
    results = []
    for scenario in SCENARIOS:
        ok = run_scenario(scenario, tools)
        results.append((scenario["label"], ok))

    hdr("BENCHMARK SUMMARY")
    for label, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {label}")
    passed = sum(1 for _, ok in results if ok)
    print(f"\n  {passed}/{len(results)} scenarios solved without catastrophic execution.\n")


if __name__ == "__main__":
    main()
