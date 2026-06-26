"""
Idea 4: Multi-Agent Dec-POMDP for Cognitive/Affective Alignment
================================================================

Full implementation of the Decentralized POMDP swarm described in the README.

Architecture
------------
The global hidden state S encompasses three orthogonal latent variables:

  s_task   — the semantic complexity and status of the user's actual request
  s_affect — the user's emotional state (calm → frustrated → distressed)
  s_urgency— the temporal pressure of the request (low → medium → critical)

No single agent observes S directly. Instead:

  Agent 1 — TaskAgent      sees: s_task-correlated signals (keywords, specificity)
  Agent 2 — AffectAgent    sees: s_affect-correlated signals (punctuation, tone)
  Agent 3 — UrgencyAgent   sees: s_urgency-correlated signals (time words, exclamations)

Each agent maintains its own local belief b_i(s_i) updated via Bayesian inference
on its private observation stream. Agents communicate ONLY when their local belief
crosses a critical divergence threshold — enforced by a message budget token system.

The system is defined by the Dec-POMDP tuple:
  ⟨I, S, {A_i}, {O_i}, T, Z, R, γ⟩

  I        = {TaskAgent, AffectAgent, UrgencyAgent}
  S        = s_task × s_affect × s_urgency   (global hidden state)
  A_i      = local action space per agent (response fragments + inter-agent messages)
  O_i      = private observation space per agent
  T(s'|s)  = state transition (user state evolves across turns)
  Z(o_i|s) = observation likelihood per agent
  R(s, a)  = shared global reward (correctness + empathy + timeliness)
  γ        = 0.90  discount factor

Training paradigm (not simulated here but architecturally modeled):
  CTDE — Centralized Training with Decentralized Execution
  MAGRPO — advantage computed against group baseline, penalises chatter loops

Key mechanisms implemented:
  [1] Per-agent local belief with private Bayesian updates
  [2] Message budget tokens — agents pay a cost to send inter-agent messages
  [3] Soft handoff clocks — each agent has a deadline; exceeding it triggers escalation
  [4] MAGRPO-style advantage computation from group rollouts
  [5] Global reward function with communication penalty
  [6] Multi-turn conversation simulation over 3 scenarios

Dec-POMDP formalization reference:
  Bernstein et al. (2002), "The Complexity of Decentralized Control of Markov Decision Processes"
  Oliehoek & Amato (2016), "A Concise Introduction to Decentralized POMDPs"
"""

import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

random.seed(7)

# ══════════════════════════════════════════════════════════════════════════════
# § 0  HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

GAMMA               = 0.90
MESSAGE_BUDGET      = 3        # max inter-agent messages per turn
MESSAGE_COST        = 0.15     # reward penalty per inter-agent message sent
CLOCK_TICKS         = 5        # max processing steps before escalation warning
DIVERGENCE_THRESHOLD= 0.35     # KL divergence above which agent must broadcast
MAGRPO_GROUPS       = 4        # parallel rollouts for advantage estimation


# ══════════════════════════════════════════════════════════════════════════════
# § 1  GLOBAL STATE SPACE
# ══════════════════════════════════════════════════════════════════════════════

# Each latent variable is discrete.
TASK_STATES    = ["simple", "moderate", "complex", "impossible"]
AFFECT_STATES  = ["calm", "mildly_frustrated", "frustrated", "distressed"]
URGENCY_STATES = ["low", "medium", "high", "critical"]

# Global state is a tuple (task, affect, urgency)
GlobalState = Tuple[str, str, str]

# Canonical global states used in the demo
DEMO_STATES: Dict[str, GlobalState] = {
    "routine_query":    ("simple",   "calm",              "low"),
    "stressful_tech":   ("complex",  "frustrated",        "high"),
    "urgent_crisis":    ("moderate", "distressed",        "critical"),
    "confused_passive": ("moderate", "mildly_frustrated", "medium"),
}


# ══════════════════════════════════════════════════════════════════════════════
# § 2  OBSERVATION FUNCTIONS  Z_i(o | s)
# ══════════════════════════════════════════════════════════════════════════════

# Each agent receives a private observation slice of the global state.
# These are noisy projections: an agent never sees the full state.

def z_task(global_state: GlobalState) -> Dict[str, float]:
    """
    TaskAgent observation likelihood over task complexity signals.
    Correlates with s_task but is noisy due to linguistic ambiguity.
    """
    task = global_state[0]
    if task == "simple":
        return {"obs_clear_request": 0.75, "obs_vague_request": 0.20, "obs_no_request": 0.05}
    elif task == "moderate":
        return {"obs_clear_request": 0.40, "obs_vague_request": 0.50, "obs_no_request": 0.10}
    elif task == "complex":
        return {"obs_clear_request": 0.20, "obs_vague_request": 0.65, "obs_no_request": 0.15}
    else:  # impossible
        return {"obs_clear_request": 0.05, "obs_vague_request": 0.40, "obs_no_request": 0.55}


def z_affect(global_state: GlobalState) -> Dict[str, float]:
    """
    AffectAgent observation likelihood over emotional tone signals.
    """
    affect = global_state[1]
    if affect == "calm":
        return {"obs_neutral_tone": 0.80, "obs_mild_frustration": 0.15, "obs_strong_emotion": 0.05}
    elif affect == "mildly_frustrated":
        return {"obs_neutral_tone": 0.35, "obs_mild_frustration": 0.55, "obs_strong_emotion": 0.10}
    elif affect == "frustrated":
        return {"obs_neutral_tone": 0.10, "obs_mild_frustration": 0.45, "obs_strong_emotion": 0.45}
    else:  # distressed
        return {"obs_neutral_tone": 0.03, "obs_mild_frustration": 0.12, "obs_strong_emotion": 0.85}


def z_urgency(global_state: GlobalState) -> Dict[str, float]:
    """
    UrgencyAgent observation likelihood over time-pressure signals.
    """
    urgency = global_state[2]
    if urgency == "low":
        return {"obs_no_time_pressure": 0.80, "obs_mild_deadline": 0.15, "obs_urgent_language": 0.05}
    elif urgency == "medium":
        return {"obs_no_time_pressure": 0.35, "obs_mild_deadline": 0.50, "obs_urgent_language": 0.15}
    elif urgency == "high":
        return {"obs_no_time_pressure": 0.10, "obs_mild_deadline": 0.35, "obs_urgent_language": 0.55}
    else:  # critical
        return {"obs_no_time_pressure": 0.02, "obs_mild_deadline": 0.08, "obs_urgent_language": 0.90}


# ══════════════════════════════════════════════════════════════════════════════
# § 3  LOCAL BELIEF  b_i(s_i)
# ══════════════════════════════════════════════════════════════════════════════

def _entropy(dist: Dict[str, float]) -> float:
    h = 0.0
    for p in dist.values():
        if p > 1e-12:
            h -= p * math.log2(p)
    return h


def _kl_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    """KL(P||Q) — measures how much P diverges from Q."""
    kl = 0.0
    for k, pk in p.items():
        qk = q.get(k, 1e-9)
        if pk > 1e-12:
            kl += pk * math.log2(pk / qk)
    return kl


class LocalBelief:
    """
    A per-agent Bayesian belief over its private latent dimension.

    b_i(s_i) is updated each turn via:
        b_{i,t+1}(s_i) ∝ Z_i(o_i | s_i) · b_{i,t}(s_i)

    The agent also tracks divergence from a broadcast_prior — when the
    KL divergence between the current belief and the last broadcast exceeds
    DIVERGENCE_THRESHOLD, the agent sends an inter-agent message.
    """

    def __init__(self, states: List[str], name: str):
        self.states      = states
        self.name        = name
        n                = len(states)
        self.belief      = {s: 1.0 / n for s in states}   # uniform prior
        self.last_broadcast = dict(self.belief)

    def update(self, obs_likelihoods: Dict[str, float]) -> None:
        """Bayesian update: b' ∝ Z(o|s) · b."""
        new_b = {}
        total = 0.0
        for s in self.states:
            likelihood       = obs_likelihoods.get(s, 1e-4)
            new_b[s]         = self.belief[s] * likelihood
            total           += new_b[s]
        if total > 1e-12:
            self.belief = {s: v / total for s, v in new_b.items()}
        # else: keep old belief (degenerate observation)

    def map_state(self) -> str:
        """Maximum a posteriori estimate."""
        return max(self.belief, key=self.belief.get)

    def entropy(self) -> float:
        return _entropy(self.belief)

    def divergence_from_last_broadcast(self) -> float:
        """KL(current || last_broadcast) — triggers message if > threshold."""
        return _kl_divergence(self.belief, self.last_broadcast)

    def mark_broadcast(self) -> None:
        self.last_broadcast = dict(self.belief)

    def __repr__(self) -> str:
        top = sorted(self.belief.items(), key=lambda x: -x[1])
        parts = [f"{s}:{p:.2f}" for s, p in top]
        return f"Belief[{self.name}]({', '.join(parts)}, H={self.entropy():.2f})"


# ══════════════════════════════════════════════════════════════════════════════
# § 4  INTER-AGENT MESSAGE SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Message:
    sender:    str
    recipient: str          # "ALL" for broadcast
    content:   Dict[str, Any]
    cost:      float = MESSAGE_COST

    def __str__(self) -> str:
        return (f"  📨  {self.sender} → {self.recipient}: "
                f"{self.content}")


class MessageBus:
    """
    Shared message bus with budget enforcement.
    Agents may send at most MESSAGE_BUDGET messages per turn.
    Exceeding the budget triggers a reward penalty (modelled as cost accumulation).
    This is the architectural enforcement of the MAGRPO chatter-loop penalty.
    """

    def __init__(self):
        self.inbox:      Dict[str, List[Message]] = {}
        self.sent_count: int   = 0
        self.total_cost: float = 0.0
        self.log:        List[Message] = []

    def reset_turn(self) -> None:
        self.inbox      = {}
        self.sent_count = 0

    def send(self, msg: Message) -> bool:
        """Returns False if budget exceeded (message dropped)."""
        if self.sent_count >= MESSAGE_BUDGET:
            return False   # chatter penalty: message silently dropped
        self.sent_count  += 1
        self.total_cost  += msg.cost

        if msg.recipient == "ALL":
            for agent_id in ["TaskAgent", "AffectAgent", "UrgencyAgent"]:
                if agent_id != msg.sender:
                    self.inbox.setdefault(agent_id, []).append(msg)
        else:
            self.inbox.setdefault(msg.recipient, []).append(msg)

        self.log.append(msg)
        return True

    def receive(self, agent_id: str) -> List[Message]:
        return self.inbox.get(agent_id, [])


# ══════════════════════════════════════════════════════════════════════════════
# § 5  GLOBAL REWARD FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def global_reward(
    true_state:      GlobalState,
    response_plan:   Dict[str, Any],
    message_cost:    float,
    clock_ticks:     int,
) -> float:
    """
    Shared global reward R(s, a_joint) evaluated centrally.

    Decomposed into three components:
      R_task    — did the response correctly address the task complexity?
      R_affect  — did the tone match the user's emotional state?
      R_urgency — was the response delivered within the urgency budget?

    Plus penalties:
      - message_cost: total cost of inter-agent communication this turn
      - latency_penalty: each clock tick beyond 1 costs 0.05

    This reward is only available centrally (CTDE) — agents cannot observe it
    individually during execution.
    """
    s_task, s_affect, s_urgency = true_state

    # ── Task alignment ────────────────────────────────────────────────────
    task_map    = response_plan.get("task_response_type", "generic")
    task_reward = {
        "simple":     {"direct_answer": 1.0, "detailed_walkthrough": 0.4,
                       "empathetic_hold": 0.1, "escalate": 0.0},
        "moderate":   {"direct_answer": 0.6, "detailed_walkthrough": 0.9,
                       "empathetic_hold": 0.3, "escalate": 0.1},
        "complex":    {"direct_answer": 0.2, "detailed_walkthrough": 1.0,
                       "empathetic_hold": 0.3, "escalate": 0.5},
        "impossible": {"direct_answer": 0.0, "detailed_walkthrough": 0.3,
                       "empathetic_hold": 0.4, "escalate": 1.0},
    }
    R_task = task_reward.get(s_task, {}).get(task_map, 0.2)

    # ── Affect alignment ──────────────────────────────────────────────────
    tone_map     = response_plan.get("tone", "neutral")
    affect_reward = {
        "calm":             {"neutral": 1.0, "warm": 0.7, "empathetic": 0.4, "urgent": 0.2},
        "mildly_frustrated":{"neutral": 0.6, "warm": 1.0, "empathetic": 0.8, "urgent": 0.3},
        "frustrated":       {"neutral": 0.2, "warm": 0.7, "empathetic": 1.0, "urgent": 0.4},
        "distressed":       {"neutral": 0.0, "warm": 0.5, "empathetic": 1.0, "urgent": 0.6},
    }
    R_affect = affect_reward.get(s_affect, {}).get(tone_map, 0.2)

    # ── Urgency alignment ─────────────────────────────────────────────────
    speed_map     = response_plan.get("response_speed", "standard")
    urgency_reward = {
        "low":      {"brief": 0.7, "standard": 1.0, "thorough": 0.9, "immediate": 0.5},
        "medium":   {"brief": 0.6, "standard": 0.9, "thorough": 0.8, "immediate": 0.7},
        "high":     {"brief": 0.8, "standard": 0.7, "thorough": 0.4, "immediate": 1.0},
        "critical": {"brief": 0.6, "standard": 0.4, "thorough": 0.1, "immediate": 1.0},
    }
    R_urgency = urgency_reward.get(s_urgency, {}).get(speed_map, 0.3)

    # ── Penalties ─────────────────────────────────────────────────────────
    latency_penalty = max(0.0, (clock_ticks - 1) * 0.05)

    R_total = (0.40 * R_task + 0.35 * R_affect + 0.25 * R_urgency
               - message_cost - latency_penalty)
    return round(max(0.0, R_total), 4)


# ══════════════════════════════════════════════════════════════════════════════
# § 6  SPECIALIZED SUB-AGENTS
# ══════════════════════════════════════════════════════════════════════════════

class DecPOMDPAgent:
    """
    Base class for a specialized agent in the Dec-POMDP swarm.

    Each agent:
      - Maintains its own LocalBelief over its private latent dimension
      - Receives a restricted observation slice from the global state
      - Processes incoming inter-agent messages to update its belief
      - Proposes a partial action (response plan fragment)
      - Broadcasts its belief update if KL > DIVERGENCE_THRESHOLD

    This implements strict decentralized execution: no agent sees the full
    global state. The only information flow between agents is through the
    MessageBus, which is budget-constrained.
    """

    def __init__(self, agent_id: str, states: List[str],
                 obs_fn, bus: MessageBus):
        self.id      = agent_id
        self.belief  = LocalBelief(states, agent_id)
        self.obs_fn  = obs_fn       # Z_i(o_i | s): observation function
        self.bus     = bus
        self.clock   = 0            # steps taken this turn (soft handoff clock)

    def observe(self, global_state: GlobalState) -> str:
        """
        Sample a private observation from Z_i(·|s).
        In production, this would be a feature extractor on raw text.
        """
        obs_dist = self.obs_fn(global_state)
        keys     = list(obs_dist.keys())
        weights  = list(obs_dist.values())
        return random.choices(keys, weights=weights)[0]

    def update_belief(self, obs: str, global_state: GlobalState) -> None:
        """Update local belief from private observation."""
        obs_dist       = self.obs_fn(global_state)
        # obs_likelihoods: for each latent state, what is P(this obs | state)?
        # We reverse the observation function: P(obs | s_i)
        likelihoods = {
            s: self.obs_fn((s, global_state[1], global_state[2])
                           if "task" in self.id.lower()
                           else (global_state[0], s, global_state[2])
                           if "affect" in self.id.lower()
                           else (global_state[0], global_state[1], s)
                          ).get(obs, 1e-4)
            for s in self.belief.states
        }
        self.belief.update(likelihoods)
        self.clock += 1

    def process_messages(self, messages: List[Message]) -> None:
        """
        Incorporate peer beliefs received from the message bus.
        Each incoming belief update nudges the agent's own belief via
        a soft fusion: b_i ← (1-α) · b_i + α · b_peer
        """
        for msg in messages:
            if "belief_update" in msg.content:
                peer_belief = msg.content["belief_update"]
                # Soft fusion — weight peer belief with 0.3
                for s in self.belief.states:
                    if s in peer_belief:
                        self.belief.belief[s] = (
                            0.70 * self.belief.belief[s]
                            + 0.30 * peer_belief[s]
                        )
                # Re-normalise
                total = sum(self.belief.belief.values())
                for s in self.belief.states:
                    self.belief.belief[s] /= total

    def maybe_broadcast(self) -> bool:
        """
        Soft handoff clock + KL-divergence trigger.
        The agent broadcasts its current belief ONLY IF:
          (a) KL(current || last_broadcast) > DIVERGENCE_THRESHOLD, OR
          (b) clock > CLOCK_TICKS (deadline exceeded, must report)
        """
        kl      = self.belief.divergence_from_last_broadcast()
        trigger = kl > DIVERGENCE_THRESHOLD or self.clock >= CLOCK_TICKS

        if trigger:
            msg = Message(
                sender    = self.id,
                recipient = "ALL",
                content   = {
                    "belief_update": dict(self.belief.belief),
                    "map_state":     self.belief.map_state(),
                    "entropy":       round(self.belief.entropy(), 3),
                    "kl_trigger":    round(kl, 3),
                },
            )
            sent = self.bus.send(msg)
            if sent:
                self.belief.mark_broadcast()
            return sent
        return False

    def propose_action(self) -> Dict[str, Any]:
        """
        Each agent proposes a fragment of the joint action plan based on
        its local MAP state estimate. The coordinator synthesizes these.
        """
        raise NotImplementedError


class TaskAgent(DecPOMDPAgent):
    """
    Tracks: semantic complexity of the user's actual request.
    Local obs: task-correlated linguistic features (specificity, request clarity).
    Proposes: the appropriate response type and depth.
    """

    RESPONSE_MAP = {
        "simple":     "direct_answer",
        "moderate":   "detailed_walkthrough",
        "complex":    "detailed_walkthrough",
        "impossible": "escalate",
    }
    SPEED_MAP = {
        "simple":     "brief",
        "moderate":   "standard",
        "complex":    "thorough",
        "impossible": "standard",
    }

    def __init__(self, bus: MessageBus):
        super().__init__("TaskAgent", TASK_STATES, z_task, bus)

    def update_belief(self, obs: str, global_state: GlobalState) -> None:
        likelihoods = {s: z_task((s, global_state[1], global_state[2])).get(obs, 1e-4)
                       for s in self.belief.states}
        self.belief.update(likelihoods)
        self.clock += 1

    def propose_action(self) -> Dict[str, Any]:
        map_s = self.belief.map_state()
        return {
            "task_response_type": self.RESPONSE_MAP[map_s],
            "response_speed":     self.SPEED_MAP[map_s],
            "confidence":         round(self.belief.belief[map_s], 3),
        }


class AffectAgent(DecPOMDPAgent):
    """
    Tracks: user's emotional state (affect / frustration level).
    Local obs: tone markers (punctuation, sentiment words, all-caps).
    Proposes: appropriate response tone.
    """

    TONE_MAP = {
        "calm":              "neutral",
        "mildly_frustrated": "warm",
        "frustrated":        "empathetic",
        "distressed":        "empathetic",
    }

    def __init__(self, bus: MessageBus):
        super().__init__("AffectAgent", AFFECT_STATES, z_affect, bus)

    def update_belief(self, obs: str, global_state: GlobalState) -> None:
        likelihoods = {s: z_affect((global_state[0], s, global_state[2])).get(obs, 1e-4)
                       for s in self.belief.states}
        self.belief.update(likelihoods)
        self.clock += 1

    def propose_action(self) -> Dict[str, Any]:
        map_s = self.belief.map_state()
        return {
            "tone":       self.TONE_MAP[map_s],
            "affect_map": map_s,
            "confidence": round(self.belief.belief[map_s], 3),
        }


class UrgencyAgent(DecPOMDPAgent):
    """
    Tracks: temporal urgency of the user's request.
    Local obs: time-pressure language markers (deadline words, exclamations).
    Proposes: appropriate response speed tier.
    """

    SPEED_OVERRIDE = {
        "low":      None,            # defer to TaskAgent's speed estimate
        "medium":   None,
        "high":     "immediate",     # override TaskAgent to respond fast
        "critical": "immediate",
    }

    def __init__(self, bus: MessageBus):
        super().__init__("UrgencyAgent", URGENCY_STATES, z_urgency, bus)

    def update_belief(self, obs: str, global_state: GlobalState) -> None:
        likelihoods = {s: z_urgency((global_state[0], global_state[1], s)).get(obs, 1e-4)
                       for s in self.belief.states}
        self.belief.update(likelihoods)
        self.clock += 1

    def propose_action(self) -> Dict[str, Any]:
        map_s    = self.belief.map_state()
        override = self.SPEED_OVERRIDE[map_s]
        return {
            "urgency_override":    override,
            "urgency_map":         map_s,
            "confidence":          round(self.belief.belief[map_s], 3),
            "escalation_required": map_s == "critical",
        }


# ══════════════════════════════════════════════════════════════════════════════
# § 7  COORDINATOR (CTDE EXECUTION PHASE)
# ══════════════════════════════════════════════════════════════════════════════

class DecPOMDPCoordinator:
    """
    Implements the Centralized Training / Decentralized Execution (CTDE) pattern.

    During execution (simulated here):
      - No agent sees the global state
      - The coordinator receives fragmented proposals from each agent
      - It synthesizes them into a final joint response plan
      - Global reward is computed (only available to the central critic in training)

    MAGRPO advantage computation:
      During training, K parallel rollouts are generated for the same input.
      The advantage for agent i in rollout k is:
        A_{i,k} = R_k − (1/K) ∑_{j≠k} R_j
      This computes the agent's contribution relative to its peers without a
      learned critic network, preventing the instability of centralised critics
      in large multi-agent systems.
    """

    def __init__(self, agents: List[DecPOMDPAgent], bus: MessageBus):
        self.agents = {a.id: a for a in agents}
        self.bus    = bus
        self.turn   = 0

    def run_turn(
        self,
        global_state:  GlobalState,
        scenario_label: str,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Execute one dialogue turn under decentralised execution.

        Steps:
          1. Each agent observes its private slice of the global state
          2. Each agent updates its local belief
          3. Agents check their KL divergence → broadcast if triggered
          4. Agents process incoming peer messages (soft belief fusion)
          5. Each agent proposes a local action fragment
          6. Coordinator synthesizes the joint response plan
          7. Global reward computed centrally
        """
        self.bus.reset_turn()
        self.turn += 1

        print(f"\n  {'─'*66}")
        print(f"  Turn {self.turn}  |  Scenario: {scenario_label}")
        print(f"  Hidden global state: task={global_state[0]}, "
              f"affect={global_state[1]}, urgency={global_state[2]}")
        print(f"  {'─'*66}")

        # ── Step 1 & 2: Observe + belief update ───────────────────────────
        clock_ticks = 0
        for agent in self.agents.values():
            obs = agent.observe(global_state)
            agent.update_belief(obs, global_state)
            clock_ticks += agent.clock
            print(f"  [{agent.id:15s}]  obs={obs:35s}  {agent.belief}")

        # ── Step 3: Maybe broadcast (KL or clock trigger) ─────────────────
        print()
        for agent in self.agents.values():
            sent = agent.maybe_broadcast()
            kl   = agent.belief.divergence_from_last_broadcast()
            if sent:
                print(f"  [{agent.id:15s}]  📡 BROADCAST triggered "
                      f"(KL={kl:.3f} > {DIVERGENCE_THRESHOLD})")
            else:
                print(f"  [{agent.id:15s}]  🔇 silent          "
                      f"(KL={kl:.3f} ≤ {DIVERGENCE_THRESHOLD})")

        # ── Step 4: Process peer messages ────────────────────────────────
        print()
        for agent in self.agents.values():
            msgs = self.bus.receive(agent.id)
            if msgs:
                agent.process_messages(msgs)
                for m in msgs:
                    print(f"  [{agent.id:15s}]  received {m}")

        # ── Step 5: Collect proposals ─────────────────────────────────────
        proposals = {aid: a.propose_action() for aid, a in self.agents.items()}

        # ── Step 6: Synthesize joint plan ─────────────────────────────────
        task_prop    = proposals["TaskAgent"]
        affect_prop  = proposals["AffectAgent"]
        urgency_prop = proposals["UrgencyAgent"]

        # UrgencyAgent can override TaskAgent's speed estimate
        final_speed = (urgency_prop["urgency_override"]
                       or task_prop["response_speed"])

        joint_plan = {
            "task_response_type": task_prop["task_response_type"],
            "tone":               affect_prop["tone"],
            "response_speed":     final_speed,
            "escalate":           urgency_prop["escalation_required"],
        }

        # ── Step 7: Global reward (CTDE — only available centrally) ───────
        reward = global_reward(
            global_state  = global_state,
            response_plan = joint_plan,
            message_cost  = self.bus.total_cost,
            clock_ticks   = max(a.clock for a in self.agents.values()),
        )

        # ── Display ────────────────────────────────────────────────────────
        print(f"\n  Agent proposals:")
        for aid, prop in proposals.items():
            print(f"    [{aid:15s}]  {prop}")

        print(f"\n  🤝  Joint response plan: {joint_plan}")
        print(f"  📬  Messages sent this turn: {self.bus.sent_count}"
              f"  (cost={self.bus.total_cost:.2f})")
        print(f"  🏆  Global reward (CTDE):   {reward:.4f}")

        # Reset agent clocks for next turn
        for agent in self.agents.values():
            agent.clock = 0

        return joint_plan, reward


# ══════════════════════════════════════════════════════════════════════════════
# § 8  MAGRPO ADVANTAGE ESTIMATION
# ══════════════════════════════════════════════════════════════════════════════

def magrpo_advantage(
    rewards: List[float],
    agent_id: str,
    agents: List[str],
) -> Dict[str, float]:
    """
    Multi-Agent Group Relative Policy Optimization (MAGRPO) advantage.

    For K parallel rollouts of the same scenario:
      A_{i,k} = R_k − mean({R_j : j ≠ k})

    This computes each rollout's advantage relative to the group baseline,
    without requiring a centralised critic network.  The advantage signal
    is used to update each agent's policy: positive advantage → reinforce,
    negative advantage → suppress.

    Critically: agents are penalised when R is low due to excessive
    communication cost (built into the global reward).  This automatically
    discourages chatter loops without explicit chatter penalties in training.
    """
    K    = len(rewards)
    adv  = {}
    mean = sum(rewards) / K

    for k, r in enumerate(rewards):
        # Group mean excluding rollout k
        others_mean = (sum(rewards) - r) / (K - 1) if K > 1 else 0.0
        adv[f"rollout_{k}"] = round(r - others_mean, 4)

    # Normalise advantages (std-normalisation, as in GRPO)
    vals  = list(adv.values())
    mu    = sum(vals) / len(vals)
    sigma = math.sqrt(sum((v - mu)**2 for v in vals) / len(vals)) + 1e-8
    adv_norm = {k: round((v - mu) / sigma, 4) for k, v in adv.items()}

    print(f"\n  📊  MAGRPO Advantage (normalised) for {agent_id}:")
    print(f"      Rewards:   {[round(r, 4) for r in rewards]}")
    print(f"      Mean:      {round(mean, 4)}")
    print(f"      Advantage: {adv_norm}")
    return adv_norm


# ══════════════════════════════════════════════════════════════════════════════
# § 9  MULTI-TURN SCENARIOS
# ══════════════════════════════════════════════════════════════════════════════

SCENARIOS = [
    {
        "label":   "Scenario 1 — Routine query (simple, calm, low-urgency)",
        "user_msg": '"How do I reset my password?"',
        "state":   DEMO_STATES["routine_query"],
    },
    {
        "label":   "Scenario 2 — Stressful tech problem (complex, frustrated, high-urgency)",
        "user_msg": '"My entire database is down and I can\'t figure out why. '
                    'This is a production system!!!"',
        "state":   DEMO_STATES["stressful_tech"],
    },
    {
        "label":   "Scenario 3 — Urgent crisis (moderate, distressed, critical)",
        "user_msg": '"URGENT: I accidentally deleted our entire customer list '
                    'and the board meeting is in 20 minutes!!!"',
        "state":   DEMO_STATES["urgent_crisis"],
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# § 10  MAIN
# ══════════════════════════════════════════════════════════════════════════════

W = 70

def hdr(text: str):
    print("\n" + "═" * W + f"\n  {text}\n" + "═" * W)

def sec(text: str):
    print("\n" + "─" * W + f"\n  {text}\n" + "─" * W)


def main():
    hdr("IDEA 4: Multi-Agent Dec-POMDP — Cognitive/Affective Alignment")
    print(f"""
  Dec-POMDP tuple: ⟨I, S, {{A_i}}, {{O_i}}, T, Z, R, γ⟩
  Agents (I)  : TaskAgent | AffectAgent | UrgencyAgent
  Global state: s_task × s_affect × s_urgency   (all HIDDEN)
  γ           : {GAMMA}
  Msg budget  : {MESSAGE_BUDGET} msgs/turn  (cost={MESSAGE_COST}/msg)
  Divergence θ: {DIVERGENCE_THRESHOLD} KL  (broadcast trigger)
  Clock ticks : {CLOCK_TICKS}  (soft handoff deadline)

  Each agent sees ONLY its private observation slice.
  Agents broadcast only when belief divergence > θ.
  Global reward is only available to the central critic (CTDE).
""")

    all_rewards: Dict[str, List[float]] = {sc["label"]: [] for sc in SCENARIOS}

    # ── Run each scenario ──────────────────────────────────────────────────
    for sc in SCENARIOS:
        hdr(sc["label"])
        print(f'  User message: {sc["user_msg"]}\n')

        # Simulate MAGRPO_GROUPS parallel rollouts of the same scenario
        rollout_rewards = []
        for rollout_idx in range(MAGRPO_GROUPS):
            bus    = MessageBus()
            agents = [TaskAgent(bus), AffectAgent(bus), UrgencyAgent(bus)]
            coord  = DecPOMDPCoordinator(agents, bus)

            sec(f"Rollout {rollout_idx + 1}/{MAGRPO_GROUPS}")
            _, reward = coord.run_turn(sc["state"], sc["label"])
            rollout_rewards.append(reward)

        # MAGRPO advantage across rollouts
        sec("MAGRPO Advantage Estimation (across parallel rollouts)")
        magrpo_advantage(rollout_rewards, "all_agents", list(coord.agents.keys()))
        all_rewards[sc["label"]] = rollout_rewards

    # ── Summary ────────────────────────────────────────────────────────────
    hdr("BENCHMARK SUMMARY")
    for label, rewards in all_rewards.items():
        mean_r = sum(rewards) / len(rewards)
        max_r  = max(rewards)
        min_r  = min(rewards)
        print(f"\n  {label}")
        print(f"    Mean reward : {mean_r:.4f}")
        print(f"    Best rollout: {max_r:.4f}")
        print(f"    Worst rollout:{min_r:.4f}")
        print(f"    Variance    : {sum((r-mean_r)**2 for r in rewards)/len(rewards):.4f}")

    print(f"""
  Key observations:
  ─────────────────
  • Agents with high KL divergence broadcast — agents in agreement stay silent.
    This naturally implements the MAGRPO chatter penalty without explicit rules.

  • UrgencyAgent overrides TaskAgent's speed estimate in high/critical scenarios.
    This demonstrates horizontal delegation: each agent controls only its variable.

  • MAGRPO advantage shows variance across rollouts due to stochastic observations.
    In training, positive-advantage rollouts reinforce the responsible agents.

  • Global reward is NEVER available to individual agents during execution.
    They coordinate blindly through budget-constrained messages only.
""")


if __name__ == "__main__":
    main()
