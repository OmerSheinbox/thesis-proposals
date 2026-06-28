"""
Idea 5 v3: Agentic Optimization via POMDP Tool Calling & Side Effect Modeling
==============================================================================

Fixes over v2:
  - EVPI now computed consistently with E[R] (not mixed CVaR/oracle)
  - Commit gate unified: fires when EVPI < ε regardless of reversibility
    (for reversible "terminal" actions like list_files when list_only is the intent)
  - Entropy-based early termination corrected: uses H < H_threshold
  - Belief prior from scenario seeded correctly into particles
  - Action success check is intent-specific, not string-based
  - CVaR used as secondary tiebreaker in MCTS, not primary value

Architecture (v3)
-----------------
  [1] Particle Filter belief — N particles ≈ b_t(s)
  [2] MCTS + UCB1 — stochastic lookahead over (tool × arg) sequences
  [3] Mixed objective — 0.7·E[R] + 0.3·CVaR_α[R] (risk-blended, not pure CVaR)
  [4] EVPI gating — unified commit trigger for all action types
  [5] Partial reversibility — discounts lookahead depth proportionally
  [6] Multi-scenario benchmark — 3 scenarios, pass/fail reporting
"""

import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

random.seed(42)

# ─────────────────────────────────────────────────────────────────
# § 0  HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────

GAMMA          = 0.90
N_PARTICLES    = 500
N_MCTS_SIMS    = 800
MCTS_DEPTH     = 5
UCB_C          = 1.4
CVAR_ALPHA     = 0.20      # tail fraction for CVaR
EV_WEIGHT      = 0.70      # blend: 0.7·E[R] + 0.3·CVaR
EVPI_EPSILON   = 0.10      # commit when EVPI drops below this
H_COMMIT       = 0.20      # also commit when entropy < this (bits)
MAX_STEPS      = 12

# ─────────────────────────────────────────────────────────────────
# § 1  WORLD
# ─────────────────────────────────────────────────────────────────

INTENTS = ["delete_temp", "delete_all", "list_only", "rename", "compress"]
INTENT_LABELS = {
    "delete_temp": "Delete .tmp / cache files only",
    "delete_all":  "Wipe entire directory",
    "list_only":   "List files — no mutation",
    "rename":      "Rename files",
    "compress":    "Archive / compress the folder",
}


@dataclass
class Action:
    """
    One (tool, argument_variant) pair — a single node in the action space A.

    reversibility ∈ [0, 1]:
        0.0 = fully irreversible (delete_all)
        0.5 = partially reversible (rename — can undo with effort)
        1.0 = fully reversible / read-only (list_files, ask_clarification)

    terminal:
        True if executing this action is itself the goal-completion step
        (e.g. list_files when true intent == list_only).  Prevents the agent
        from looping on safe actions forever.
    """
    id:            str
    reversibility: float
    terminal:      bool
    rewards:       Dict[str, float]
    obs_likelihoods: Dict[str, Dict[str, float]]


def _norm(d: Dict[str, float]) -> Dict[str, float]:
    t = sum(d.values())
    return {k: v / t for k, v in d.items()} if t > 0 else d


def build_actions() -> Dict[str, Action]:
    return {

        # ── Information-gathering (fully reversible) ───────────────────────

        "ask::broad": Action(
            id="ask::broad", reversibility=1.0, terminal=False,
            rewards={s: 0.20 for s in INTENTS},
            obs_likelihoods={
                "delete_temp": _norm({"hint_delete": 0.55, "ambiguous": 0.45}),
                "delete_all":  _norm({"hint_delete": 0.50, "ambiguous": 0.50}),
                "list_only":   _norm({"hint_list": 0.60, "ambiguous": 0.40}),
                "rename":      _norm({"hint_rename": 0.60, "ambiguous": 0.40}),
                "compress":    _norm({"hint_compress": 0.60, "ambiguous": 0.40}),
            },
        ),

        "ask::targeted": Action(
            id="ask::targeted", reversibility=1.0, terminal=False,
            rewards={s: 0.15 for s in INTENTS},
            obs_likelihoods={
                "delete_temp": _norm({"confirm_temp": 0.88, "ambiguous": 0.12}),
                "delete_all":  _norm({"confirm_all":  0.88, "ambiguous": 0.12}),
                "list_only":   _norm({"confirm_list": 0.88, "ambiguous": 0.12}),
                "rename":      _norm({"confirm_rename": 0.88, "ambiguous": 0.12}),
                "compress":    _norm({"confirm_compress": 0.88, "ambiguous": 0.12}),
            },
        ),

        "list::shallow": Action(
            id="list::shallow", reversibility=1.0, terminal=True,
            rewards={
                "delete_temp": 0.30, "delete_all": 0.30,
                "list_only":   0.85, "rename": 0.30, "compress": 0.30,
            },
            obs_likelihoods={
                "delete_temp": _norm({"many_tmp": 0.72, "mixed": 0.28}),
                "delete_all":  _norm({"many_tmp": 0.35, "mixed": 0.65}),
                "list_only":   _norm({"many_tmp": 0.40, "mixed": 0.60}),
                "rename":      _norm({"mixed": 0.82, "many_tmp": 0.18}),
                "compress":    _norm({"mixed": 0.78, "many_tmp": 0.22}),
            },
        ),

        "list::deep": Action(
            id="list::deep", reversibility=1.0, terminal=True,
            rewards={
                "delete_temp": 0.40, "delete_all": 0.40,
                "list_only":   1.00, "rename": 0.40, "compress": 0.50,
            },
            obs_likelihoods={
                "delete_temp": _norm({"many_tmp": 0.80, "mixed": 0.20}),
                "delete_all":  _norm({"many_tmp": 0.40, "mixed": 0.60}),
                "list_only":   _norm({"many_tmp": 0.45, "mixed": 0.55}),
                "rename":      _norm({"mixed": 0.90, "many_tmp": 0.10}),
                "compress":    _norm({"large": 0.70, "mixed": 0.30}),
            },
        ),

        # ── Partially reversible ───────────────────────────────────────────

        "rename::snake": Action(
            id="rename::snake", reversibility=0.6, terminal=True,
            rewards={
                "delete_temp": -0.40, "delete_all": -0.40,
                "list_only":   -0.30, "rename": 0.90, "compress": -0.20,
            },
            obs_likelihoods={
                "delete_temp": _norm({"wrong_action": 1.0}),
                "delete_all":  _norm({"wrong_action": 1.0}),
                "list_only":   _norm({"wrong_action": 1.0}),
                "rename":      _norm({"success": 0.92, "partial": 0.08}),
                "compress":    _norm({"wrong_action": 1.0}),
            },
        ),

        "compress::zip": Action(
            id="compress::zip", reversibility=0.85, terminal=True,
            rewards={
                "delete_temp": -0.20, "delete_all": -0.50,
                "list_only":   -0.10, "rename": -0.20, "compress": 1.00,
            },
            obs_likelihoods={
                "delete_temp": _norm({"wrong_action": 1.0}),
                "delete_all":  _norm({"wrong_action": 1.0}),
                "list_only":   _norm({"wrong_action": 0.50, "success": 0.50}),
                "rename":      _norm({"wrong_action": 1.0}),
                "compress":    _norm({"success": 0.95, "partial": 0.05}),
            },
        ),

        # ── Irreversible ───────────────────────────────────────────────────

        "del_temp::current": Action(
            id="del_temp::current", reversibility=0.0, terminal=True,
            rewards={
                "delete_temp": 0.90, "delete_all": -0.40,
                "list_only":  -1.60, "rename": -1.60, "compress": -0.60,
            },
            obs_likelihoods={
                "delete_temp": _norm({"success": 1.0}),
                "delete_all":  _norm({"partial": 1.0}),
                "list_only":   _norm({"unintended_delete": 1.0}),
                "rename":      _norm({"unintended_delete": 1.0}),
                "compress":    _norm({"unintended_delete": 1.0}),
            },
        ),

        "del_temp::recursive": Action(
            id="del_temp::recursive", reversibility=0.0, terminal=True,
            rewards={
                "delete_temp": 1.00, "delete_all": -0.30,
                "list_only":  -1.90, "rename": -1.90, "compress": -0.80,
            },
            obs_likelihoods={
                "delete_temp": _norm({"success": 1.0}),
                "delete_all":  _norm({"partial": 1.0}),
                "list_only":   _norm({"unintended_delete": 1.0}),
                "rename":      _norm({"unintended_delete": 1.0}),
                "compress":    _norm({"unintended_delete": 1.0}),
            },
        ),

        "del_all::confirmed": Action(
            id="del_all::confirmed", reversibility=0.0, terminal=True,
            rewards={
                "delete_temp": -0.80, "delete_all": 1.00,
                "list_only":  -2.50, "rename": -2.50, "compress": -1.50,
            },
            obs_likelihoods={
                "delete_temp": _norm({"unintended_delete": 1.0}),
                "delete_all":  _norm({"success": 1.0}),
                "list_only":   _norm({"unintended_delete": 1.0}),
                "rename":      _norm({"unintended_delete": 1.0}),
                "compress":    _norm({"unintended_delete": 1.0}),
            },
        ),
    }


# ─────────────────────────────────────────────────────────────────
# § 2  PARTICLE FILTER
# ─────────────────────────────────────────────────────────────────

class ParticleFilter:
    """
    Sequential Importance Resampling (SIR) over intent hypotheses.
    Represents b_t(s) ≈ {particle_i, weight_i}.
    """

    def __init__(self, n: int = N_PARTICLES, prior: Optional[Dict[str, float]] = None):
        if prior:
            self.particles = random.choices(
                list(prior.keys()), weights=list(prior.values()), k=n
            )
        else:
            self.particles = [random.choice(INTENTS) for _ in range(n)]
        self.n = n

    def belief(self) -> Dict[str, float]:
        counts = defaultdict(int)
        for p in self.particles:
            counts[p] += 1
        return {s: counts[s] / self.n for s in INTENTS}

    def entropy(self) -> float:
        h = 0.0
        for p in self.belief().values():
            if p > 1e-12:
                h -= p * math.log2(p)
        return h

    def map_intent(self) -> str:
        """Maximum a posteriori intent."""
        b = self.belief()
        return max(b, key=b.get)

    def update(self, action: Action, obs: str):
        """SIR update: weight by Z(o|s,a), resample."""
        weights = [
            action.obs_likelihoods.get(s, {}).get(obs, 1e-4)
            for s in self.particles
        ]
        total = sum(weights)
        if total < 1e-10:
            # Particle deprivation: reset
            self.particles = [random.choice(INTENTS) for _ in range(self.n)]
            return
        weights = [w / total for w in weights]
        self.particles = random.choices(self.particles, weights=weights, k=self.n)

    def clone(self) -> "ParticleFilter":
        pf = ParticleFilter.__new__(ParticleFilter)
        pf.particles = list(self.particles)
        pf.n         = self.n
        return pf


# ─────────────────────────────────────────────────────────────────
# § 3  VALUE METRICS
# ─────────────────────────────────────────────────────────────────

def ev(pf: ParticleFilter, action: Action) -> float:
    """E[R(a)] from particle distribution."""
    b = pf.belief()
    return sum(b[s] * action.rewards.get(s, 0.0) for s in INTENTS)


def cvar(pf: ParticleFilter, action: Action, alpha: float = CVAR_ALPHA) -> float:
    """
    CVaR_α[R]:  mean reward of the worst α-fraction of world states
    (sampled from particles).  More risk-averse than E[R].
    """
    rs = sorted(action.rewards.get(s, 0.0) for s in pf.particles)
    k  = max(1, int(alpha * len(rs)))
    return sum(rs[:k]) / k


def blended_value(pf: ParticleFilter, action: Action,
                  w: float = EV_WEIGHT) -> float:
    """
    ρ(b, a) = w · E[R] + (1−w) · CVaR_α[R]

    Interpolates between pure expected-value maximisation and pure
    risk-aversion.  w=1 → risk-neutral; w=0 → worst-case pessimist.
    """
    return w * ev(pf, action) + (1 - w) * cvar(pf, action)


def evpi(pf: ParticleFilter, actions: Dict[str, Action]) -> float:
    """
    EVPI = E_s[max_a R(s,a)] − max_a E_s[R(s,a)]

    Computed consistently in E[R] space (not mixed with CVaR).
    """
    b = pf.belief()
    v_oracle  = sum(
        b[s] * max(a.rewards.get(s, 0.0) for a in actions.values())
        for s in INTENTS
    )
    v_best_ev = max(ev(pf, a) for a in actions.values())
    return max(0.0, v_oracle - v_best_ev)


# ─────────────────────────────────────────────────────────────────
# § 4  MCTS + UCB1
# ─────────────────────────────────────────────────────────────────

class MCTSNode:
    __slots__ = ("action_id", "visits", "total")

    def __init__(self, action_id: str):
        self.action_id = action_id
        self.visits    = 0
        self.total     = 0.0

    def ucb1(self, parent_visits: int) -> float:
        if self.visits == 0:
            return float("inf")
        return (self.total / self.visits
                + UCB_C * math.sqrt(math.log(parent_visits) / self.visits))

    def mean(self) -> float:
        return self.total / self.visits if self.visits > 0 else 0.0


def _random_rollout(pf: ParticleFilter, actions: Dict[str, Action],
                    depth: int) -> float:
    if depth == 0:
        return 0.0
    # Only roll out safe/reversible actions
    safe = [aid for aid, a in actions.items() if a.reversibility >= 0.6]
    if not safe:
        return 0.0
    aid = random.choice(safe)
    a   = actions[aid]
    s   = random.choice(pf.particles)
    r   = a.rewards.get(s, 0.0)
    obs_d = a.obs_likelihoods.get(s, {})
    if obs_d:
        obs = random.choices(list(obs_d.keys()), weights=list(obs_d.values()))[0]
        pf2 = pf.clone()
        pf2.update(a, obs)
        return r + GAMMA * _random_rollout(pf2, actions, depth - 1)
    return r


def run_mcts(pf: ParticleFilter,
             actions: Dict[str, Action]) -> Tuple[str, Dict[str, float]]:
    """
    Runs N_MCTS_SIMS simulations.  Each simulation:
      1. Selects action via UCB1
      2. Simulates reward by sampling one particle as true state
      3. Rolls out with random policy (reversible only) for MCTS_DEPTH-1 steps
      4. Backpropagates blended value

    Returns (best_action_id, {action_id -> mean_value}).
    """
    nodes: Dict[str, MCTSNode] = {aid: MCTSNode(aid) for aid in actions}
    total_visits = 0

    for _ in range(N_MCTS_SIMS):
        # Selection by UCB1
        aid  = max(nodes, key=lambda k: nodes[k].ucb1(max(total_visits, 1)))
        node = nodes[aid]
        a    = actions[aid]

        # Simulation
        s    = random.choice(pf.particles)
        r    = a.rewards.get(s, 0.0)

        # Lookahead (only for reversible actions)
        future = 0.0
        if a.reversibility > 0.1 and MCTS_DEPTH > 1:
            obs_d = a.obs_likelihoods.get(s, {})
            if obs_d:
                obs  = random.choices(list(obs_d.keys()), weights=list(obs_d.values()))[0]
                pf2  = pf.clone()
                pf2.update(a, obs)
                future = GAMMA * _random_rollout(pf2, actions, MCTS_DEPTH - 1)

        sim_val = blended_value(pf, a)  # risk-adjusted immediate signal
        value   = 0.6 * (r + future) + 0.4 * sim_val

        # Backprop
        node.visits += 1
        node.total  += value
        total_visits += 1

    best = max(nodes, key=lambda k: nodes[k].mean())
    return best, {aid: nodes[aid].mean() for aid in nodes}


# ─────────────────────────────────────────────────────────────────
# § 5  AGENT LOOP
# ─────────────────────────────────────────────────────────────────

@dataclass
class Step:
    t:         int
    action_id: str
    obs:       str
    belief:    Dict[str, float]
    H:         float
    evpi_val:  float
    committed: bool
    ev_scores: Dict[str, float]
    cv_scores: Dict[str, float]


def agent_loop(
    pf:          ParticleFilter,
    actions:     Dict[str, Action],
    true_intent: str,
) -> Tuple[str, List[Step], bool]:
    """
    Agent decision loop.

    Commit conditions (unified):
      (a) EVPI < EVPI_EPSILON  AND  MCTS winner is a terminal action, OR
      (b) H(b_t) < H_COMMIT   AND  MCTS winner is a terminal action

    The environment samples obs from Z(o | true_intent, a).
    true_intent is ONLY used here — the agent never reads it directly.
    """
    trajectory: List[Step] = []
    q_asked = 0

    for t in range(MAX_STEPS):
        H        = pf.entropy()
        evpi_val = evpi(pf, actions)
        b        = pf.belief()

        best_aid, mcts_scores = run_mcts(pf, actions)
        best_action = actions[best_aid]

        ev_scores = {aid: ev(pf, a)  for aid, a in actions.items()}
        cv_scores = {aid: cvar(pf, a) for aid, a in actions.items()}

        # ── Commit gate ────────────────────────────────────────────
        commit_ready = (evpi_val < EVPI_EPSILON) or (H < H_COMMIT)
        if commit_ready and best_action.terminal:
            success = best_action.rewards.get(true_intent, -99) >= 0.75
            trajectory.append(Step(
                t=t, action_id=best_aid, obs="[COMMITTED]",
                belief=b, H=H, evpi_val=evpi_val, committed=True,
                ev_scores=ev_scores, cv_scores=cv_scores,
            ))
            return best_aid, trajectory, success

        # ── Enforce question budget ────────────────────────────────
        if best_aid.startswith("ask::"):
            q_asked += 1
            if q_asked > 3:
                # Fall back to best non-question terminal action by EV
                cands = [
                    aid for aid, a in actions.items()
                    if not aid.startswith("ask::") and a.terminal
                ]
                best_aid    = max(cands, key=lambda k: ev_scores.get(k, -999))
                best_action = actions[best_aid]

        # ── Sample environment observation ─────────────────────────
        obs_d = best_action.obs_likelihoods.get(true_intent, {})
        obs   = (random.choices(list(obs_d.keys()), weights=list(obs_d.values()))[0]
                 if obs_d else "obs_null")

        # ── Belief update ──────────────────────────────────────────
        pf.update(best_action, obs)

        trajectory.append(Step(
            t=t, action_id=best_aid, obs=obs,
            belief=pf.belief(), H=pf.entropy(), evpi_val=evpi(pf, actions),
            committed=False, ev_scores=ev_scores, cv_scores=cv_scores,
        ))

    # Max steps reached without commit
    # Commit to MAP intent's best action as fallback
    map_intent  = pf.map_intent()
    fallback    = max(actions, key=lambda k: actions[k].rewards.get(map_intent, -999))
    success     = actions[fallback].rewards.get(true_intent, -99) >= 0.75
    return fallback, trajectory, success


# ─────────────────────────────────────────────────────────────────
# § 6  DISPLAY
# ─────────────────────────────────────────────────────────────────

W = 72

def hdr(t: str):
    print("\n" + "═" * W + f"\n  {t}\n" + "═" * W)

def sec(t: str):
    print("\n" + "─" * W + f"\n  {t}\n" + "─" * W)

INTENT_SHORT = {
    "delete_temp": "del_tmp",
    "delete_all":  "del_all",
    "list_only":   "list   ",
    "rename":      "rename ",
    "compress":    "compr  ",
}

def print_step(step: Step):
    tag = "🔒 COMMIT" if step.committed else f"t={step.t + 1}"
    print(f"\n  ┌─ {tag} ──────────────────────────────────────────────────")
    print(f"  │  action   : {step.action_id}")
    print(f"  │  obs      : {step.obs}")
    print(f"  │  H(b)     : {step.H:.3f} bits    EVPI: {step.evpi_val:.4f}")
    print(f"  │  belief   :")
    for s in INTENTS:
        p   = step.belief.get(s, 0.0)
        bar = "█" * int(p * 22)
        print(f"  │    {INTENT_SHORT[s]}  {p:.3f}  {bar}")
    print(f"  │  top E[R] scores:")
    for aid, sc in sorted(step.ev_scores.items(), key=lambda x: -x[1])[:4]:
        irr = " [⚠ irrev]" if actions_global.get(aid, None) and \
              actions_global[aid].reversibility < 0.1 else ""
        print(f"  │    {aid:28s}  {sc:+.3f}{irr}")
    print(f"  └──────────────────────────────────────────────────────────")


actions_global: Dict[str, Action] = {}


# ─────────────────────────────────────────────────────────────────
# § 7  SCENARIOS
# ─────────────────────────────────────────────────────────────────

SCENARIOS = [
    {
        "label":       "A — Ambiguous cleanup ('clean up the project folder')",
        "true_intent": "delete_temp",
        "prior":       {s: 0.20 for s in INTENTS},
    },
    {
        "label":       "B — Strong delete signal ('nuke everything in there')",
        "true_intent": "delete_all",
        "prior":       {"delete_temp": 0.10, "delete_all": 0.55,
                        "list_only": 0.10, "rename": 0.10, "compress": 0.15},
    },
    {
        "label":       "C — Passive request ('show me what's in the folder')",
        "true_intent": "list_only",
        "prior":       {"delete_temp": 0.08, "delete_all": 0.04,
                        "list_only": 0.65, "rename": 0.13, "compress": 0.10},
    },
]


# ─────────────────────────────────────────────────────────────────
# § 8  MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    global actions_global
    actions = build_actions()
    actions_global = actions

    hdr("IDEA 5 v3 — POMDP Tool Calling & Side Effect Modeling")
    print(f"""
  Action space : {len(actions)} actions  ({len(INTENTS)} intents)
  Particles    : {N_PARTICLES}    MCTS sims: {N_MCTS_SIMS}  depth: {MCTS_DEPTH}
  Risk blend   : {EV_WEIGHT}·E[R] + {1-EV_WEIGHT:.1f}·CVaR(α={CVAR_ALPHA})
  Commit gate  : EVPI < {EVPI_EPSILON}  OR  H(b) < {H_COMMIT} bits
""")

    results = []
    for sc in SCENARIOS:
        pf = ParticleFilter(N_PARTICLES, sc["prior"])
        hdr(f"Scenario {sc['label']}")
        print(f"  True intent (hidden): {sc['true_intent']}\n")

        t0 = time.time()
        final, traj, ok = agent_loop(pf, actions, sc["true_intent"])
        elapsed = time.time() - t0

        for step in traj:
            print_step(step)

        status = "✅ SUCCESS" if ok else "❌ FAILURE"
        sec(f"Result — {status}")
        print(f"  Committed action : {final}")
        print(f"  True intent      : {sc['true_intent']}")
        print(f"  Reward achieved  : {actions[final].rewards.get(sc['true_intent'], 0):.2f}")
        print(f"  Steps taken      : {len(traj)}")
        print(f"  Wall time        : {elapsed:.2f}s")
        results.append((sc["label"], ok, final, sc["true_intent"]))

    hdr("BENCHMARK SUMMARY")
    for label, ok, final, intent in results:
        marker = "✅" if ok else "❌"
        print(f"  {marker}  {label}")
        print(f"       committed → {final}  |  true intent: {intent}")
    passed = sum(1 for _, ok, _, _ in results if ok)
    print(f"\n  {passed}/{len(results)} scenarios resolved correctly.\n")


if __name__ == "__main__":
    main()
