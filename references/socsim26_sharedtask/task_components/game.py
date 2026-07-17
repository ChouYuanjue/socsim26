"""Repeated-game backends and GM components for S3 and S8."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from silisocs.environments.backends.base import BackendApp, app_action
from silisocs.runtime.types import ActionOutput, ToolCall


class GameChoiceStore:
    """Per-simulation store for choices and payoffs across rounds."""

    def __init__(self) -> None:
        self._choices: dict[int, dict[str, str]] = {}
        self._payoffs: dict[int, dict[str, float]] = {}

    def record_choice(self, step: int, agent: str, choice: str) -> None:
        self._choices.setdefault(step, {})[agent] = str(choice)

    def record_payoffs(self, step: int, payoffs: dict[str, float]) -> None:
        self._payoffs[step] = dict(payoffs)

    def choices_at(self, step: int) -> dict[str, str]:
        return dict(self._choices.get(step, {}))

    def payoffs_at(self, step: int) -> dict[str, float]:
        return dict(self._payoffs.get(step, {}))

    def all_steps(self) -> list[int]:
        return sorted(set(self._choices) | set(self._payoffs))


def beauty_contest_payoff(choices: dict[str, int]) -> dict[str, float]:
    """11-20 money-game payoffs, basic version (Arad & Rubinstein 2012).

    Each agent picks N in [11, 20] and earns N points. Bonus: +20 if
    your number equals exactly one less than any other agent's number.
    """
    payoffs: dict[str, float] = {}
    for agent, n in choices.items():
        others = [v for a, v in choices.items() if a != agent]
        bonus = 20.0 if any(n == m - 1 for m in others) else 0.0
        payoffs[agent] = float(n) + bonus
    return payoffs


def money_game_cycle_payoff(choices: dict[str, int]) -> dict[str, float]:
    """Cycle version (A&R 2012, variant 1): the undercut bonus also wraps —
    requesting 20 wins the bonus against an opponent who requested 11."""
    payoffs: dict[str, float] = {}
    for agent, n in choices.items():
        others = [v for a, v in choices.items() if a != agent]
        undercut = any(n == m - 1 for m in others)
        wrap = n == 20 and any(m == 11 for m in others)
        payoffs[agent] = float(n) + (20.0 if (undercut or wrap) else 0.0)
    return payoffs


def money_game_costless_payoff(choices: dict[str, int]) -> dict[str, float]:
    """Costless-iterations version (A&R 2012, variant 2): requesting 20 pays a
    sure 20; any request 11-19 pays a flat 17, plus the +20 bonus when it is
    exactly one less than another player's request."""
    payoffs: dict[str, float] = {}
    for agent, n in choices.items():
        if n == 20:
            payoffs[agent] = 20.0
            continue
        others = [v for a, v in choices.items() if a != agent]
        bonus = 20.0 if any(n == m - 1 for m in others) else 0.0
        payoffs[agent] = 17.0 + bonus
    return payoffs


MONEY_GAME_PAYOFFS = {
    "basic": beauty_contest_payoff,
    "cycle": money_game_cycle_payoff,
    "costless": money_game_costless_payoff,
}


def pd_payoff(
    choices: dict[str, str],
    *,
    T: float = 5.0,
    R: float = 3.0,
    P: float = 1.0,
    S: float = 0.0,
) -> dict[str, float]:
    """Iterated prisoner's dilemma payoffs over all unordered player pairs."""
    agents = list(choices)
    if len(agents) < 2:
        return {a: 0.0 for a in agents}

    matrix: dict[tuple[str, str], tuple[float, float]] = {
        ("COOPERATE", "COOPERATE"): (R, R),
        ("COOPERATE", "DEFECT"): (S, T),
        ("DEFECT", "COOPERATE"): (T, S),
        ("DEFECT", "DEFECT"): (P, P),
    }

    totals: dict[str, float] = {a: 0.0 for a in agents}
    for i, a in enumerate(agents):
        for j, b in enumerate(agents):
            if i >= j:
                continue
            ca = choices[a].upper().strip()
            cb = choices[b].upper().strip()
            pa, pb = matrix[(ca, cb)]
            totals[a] += pa
            totals[b] += pb

    n_opponents = len(agents) - 1
    return {a: totals[a] / n_opponents for a in agents}


def parse_number_choice(text: str) -> int | None:
    """Extract an integer 11-20 from agent action text."""
    matches = re.findall(r"\b(1[1-9]|20)\b", str(text or ""))
    if matches:
        return int(matches[0])
    return None


def parse_pd_choice(
    text: str,
    cooperate_label: str = "COOPERATE",
    defect_label: str = "DEFECT",
) -> str | None:
    """Extract the (possibly relabelled) PD choice from agent action text.

    Returns the canonical token "COOPERATE" or "DEFECT" regardless of the
    labels shown to the agent. When both labels appear, the earliest
    occurrence wins.
    """
    upper = str(text or "").upper()
    hits: list[tuple[int, str]] = []
    for canonical, label in (("COOPERATE", cooperate_label), ("DEFECT", defect_label)):
        match = re.search(rf"\b{re.escape(label.upper())}\b", upper)
        if match:
            hits.append((match.start(), canonical))
    if not hits:
        return None
    return min(hits)[1]


class RepeatedGameBackendBase(BackendApp):
    """Common state and event logging for simultaneous repeated games."""

    app_description = "A repeated game backend."

    def __init__(self, *, action_logger: Any = None, app_description: str = "") -> None:
        super().__init__()
        self.action_logger = action_logger
        if app_description:
            self.app_description = str(app_description)
        self.agent_names: list[str] = []
        self.store = GameChoiceStore()

    def description(self) -> str:
        return self.app_description

    def initialize(self, agent_names: list[str], **kwargs: Any) -> None:
        del kwargs
        self.agent_names = list(agent_names)
        self.store = GameChoiceStore()
        self._log("system", "initialize_game", {"players": list(self.agent_names)})

    def update(self, *, step: int, agent_names: Sequence[str], context: Any | None = None) -> None:
        del context
        self.agent_names = list(agent_names)
        if step > 0:
            self.finalize_round(step - 1)

    def current_step(self) -> int:
        action_logger = getattr(self, "action_logger", None)
        return int(getattr(action_logger, "episode_idx", 0) or 0)

    def record_choice(self, agent_name: str, choice: str) -> str:
        if agent_name not in self.agent_names:
            raise ValueError(f"Unknown game player {agent_name!r}.")
        step = self.current_step()
        self.store.record_choice(step, agent_name, choice)
        self._log(agent_name, self.choice_event_label, {"round": step, "choice": choice})
        return f"{agent_name} chose {choice}. Waiting for all players."

    def finalize_round(self, step: int) -> None:
        if self.store.payoffs_at(step):
            return
        choices = self.store.choices_at(step)
        if not choices:
            return
        missing = [name for name in self.agent_names if name not in choices]
        if missing:
            raise ValueError(f"Round {step + 1} is missing choices for: {', '.join(missing)}.")
        payoffs = self.compute_payoffs(choices)
        self.store.record_payoffs(step, payoffs)
        self._log(
            "system",
            "round_payoff",
            {"round": step, "choices": choices, "payoffs": payoffs},
        )

    def _log(self, source_user: str, label: str, data: Mapping[str, Any]) -> None:
        if self.action_logger is None:
            return
        self.action_logger.log(
            {
                "source_user": source_user,
                "action_type": label,
                "label": label,
                "data": dict(data),
            }
        )

    @property
    def choice_event_label(self) -> str:
        raise NotImplementedError

    def compute_payoffs(self, choices: dict[str, str]) -> dict[str, float]:
        raise NotImplementedError

    def observe(self, actor_name: str, **kwargs: Any) -> str:
        del kwargs
        raise NotImplementedError


_MONEY_GAME_RULES = {
    "basic": (
        "=== 11-20 MONEY GAME ===\n"
        "You and the other player each request a whole number of points from 11 to 20.\n"
        "You receive the amount you request.\n"
        "BONUS: you receive an additional 20 points if you request exactly one less "
        "than the other player.\n"
        "Choices are made simultaneously; neither player sees the other's choice in advance."
    ),
    "cycle": (
        "=== 11-20 MONEY GAME ===\n"
        "You and the other player each request a whole number of points from 11 to 20.\n"
        "You receive the amount you request.\n"
        "BONUS: you receive an additional 20 points if you request exactly one less "
        "than the other player, OR if you request 20 and the other player requests 11.\n"
        "Choices are made simultaneously; neither player sees the other's choice in advance."
    ),
    "costless": (
        "=== 11-20 MONEY GAME ===\n"
        "You and the other player each request a whole number from 11 to 20.\n"
        "If you request 20, you receive 20 points for sure.\n"
        "If you request any number from 11 to 19, you receive 17 points, plus a BONUS "
        "of an additional 20 points if you request exactly one less than the other player.\n"
        "Choices are made simultaneously; neither player sees the other's choice in advance."
    ),
}


class MoneyGameBackend(RepeatedGameBackendBase):
    """Backend for the 11-20 money game (basic / cycle / costless variants)."""

    app_description = "A two-player 11-20 money game."

    def __init__(
        self,
        *,
        action_logger: Any = None,
        app_description: str = "",
        variant: str = "basic",
        rules_text: str | None = None,
    ) -> None:
        super().__init__(action_logger=action_logger, app_description=app_description)
        if variant not in MONEY_GAME_PAYOFFS:
            raise ValueError(
                f"Unknown money-game variant {variant!r}; expected one of "
                f"{sorted(MONEY_GAME_PAYOFFS)}."
            )
        self.variant = str(variant)
        self.rules_text = str(rules_text) if rules_text else None

    def name(self) -> str:
        return "money_game"

    @property
    def choice_event_label(self) -> str:
        return "choose_number"

    @app_action(selectable_name="CHOOSE_NUMBER")
    def choose_number(self, agent_name: str, number: int) -> str:
        """Choose a whole number between 11 and 20 for the current round.

        Parameters
        ----------
        agent_name:
            Runtime-supplied player name.
        number:
            Whole number from 11 to 20.
        """
        if int(number) < 11 or int(number) > 20:
            raise ValueError(f"Money-game choice must be between 11 and 20, got {number}.")
        return self.record_choice(agent_name, str(int(number)))

    def choose_from_text(self, agent_name: str, text: str) -> str:
        choice = parse_number_choice(text)
        if choice is None:
            raise ValueError(f"{agent_name} did not provide a valid 11-20 money-game choice.")
        return self.choose_number(agent_name, choice)

    def compute_payoffs(self, choices: dict[str, str]) -> dict[str, float]:
        payoff_fn = MONEY_GAME_PAYOFFS[self.variant]
        return payoff_fn({agent: int(choice) for agent, choice in choices.items()})

    def observe(self, actor_name: str, **kwargs: Any) -> str:
        del kwargs
        step = self.current_step()
        parts = [self.rules_text or _MONEY_GAME_RULES[self.variant]]
        if step == 0:
            parts.append("\nMake your choice.")
        else:
            prev = step - 1
            choices = self.store.choices_at(prev)
            payoffs = self.store.payoffs_at(prev)
            parts.append(f"\nRound {prev + 1} results:")
            for name in self.agent_names:
                marker = " <- you" if name == actor_name else ""
                parts.append(
                    f"  {name}: chose {choices.get(name, '?')}, "
                    f"earned {payoffs.get(name, 0.0):g}{marker}"
                )
            parts.append(f"\nNow choose your number for round {step + 1}.")
        return "\n".join(parts)


_PD_RULES_TEMPLATE = (
    "=== REPEATED CHOICE GAME ===\n"
    "Each round: choose {coop} or {defect}.\n"
    "You play against {opponents} simultaneously.\n"
    "Payoff per opponent (you / them):\n"
    "  {coop} vs {coop} = {R:g} / {R:g}\n"
    "  {coop} vs {defect} = {S:g} / {T:g}\n"
    "  {defect} vs {coop} = {T:g} / {S:g}\n"
    "  {defect} vs {defect} = {P:g} / {P:g}\n"
    "Your score per round = mean of pairwise payoffs. History is visible."
)

PD_HISTORY_FORMATS = ("lines", "table", "summary_counts")


class IteratedPDBackend(RepeatedGameBackendBase):
    """Backend for an all-pairs iterated prisoner's dilemma.

    Choices are stored and logged canonically (COOPERATE / DEFECT) even when
    relabelled for the agents; the labels shown in observations come from
    ``cooperate_label`` / ``defect_label``.
    """

    app_description = "An all-pairs iterated prisoner's dilemma."

    def __init__(
        self,
        *,
        action_logger: Any = None,
        app_description: str = "",
        T: float = 5.0,
        R: float = 3.0,
        P: float = 1.0,
        S: float = 0.0,
        cooperate_label: str = "COOPERATE",
        defect_label: str = "DEFECT",
        history_format: str = "lines",
        rules_text: str | None = None,
        framing_note: str = "",
    ) -> None:
        super().__init__(action_logger=action_logger, app_description=app_description)
        self.T = float(T)
        self.R = float(R)
        self.P = float(P)
        self.S = float(S)
        self.cooperate_label = str(cooperate_label)
        self.defect_label = str(defect_label)
        if self.cooperate_label.upper() == self.defect_label.upper():
            raise ValueError("cooperate_label and defect_label must differ.")
        if history_format not in PD_HISTORY_FORMATS:
            raise ValueError(
                f"Unknown history_format {history_format!r}; expected one of "
                f"{PD_HISTORY_FORMATS}."
            )
        self.history_format = str(history_format)
        self.rules_text = str(rules_text) if rules_text else None
        self.framing_note = str(framing_note or "")

    def name(self) -> str:
        return "iterated_pd"

    @property
    def choice_event_label(self) -> str:
        return "choose_pd_action"

    def _label(self, canonical: str) -> str:
        if canonical == "COOPERATE":
            return self.cooperate_label
        if canonical == "DEFECT":
            return self.defect_label
        return str(canonical)

    def _rules(self) -> str:
        if self.rules_text:
            return self.rules_text
        n_opponents = max(len(self.agent_names) - 1, 1)
        opponents = "the other player" if n_opponents == 1 else "all other players"
        return _PD_RULES_TEMPLATE.format(
            T=self.T,
            R=self.R,
            P=self.P,
            S=self.S,
            coop=self.cooperate_label,
            defect=self.defect_label,
            opponents=opponents,
        )

    @app_action(selectable_name="CHOOSE_PD_ACTION")
    def choose_pd_action(self, agent_name: str, choice: str) -> str:
        """Choose one of the two game actions for the current round.

        Parameters
        ----------
        agent_name:
            Runtime-supplied player name.
        choice:
            One of the two action labels shown in the game rules.
        """
        parsed = parse_pd_choice(choice, self.cooperate_label, self.defect_label)
        if parsed is None:
            raise ValueError(
                f"PD choice must be {self.cooperate_label} or {self.defect_label}, "
                f"got {choice!r}."
            )
        return self.record_choice(agent_name, parsed)

    def choose_from_text(self, agent_name: str, text: str) -> str:
        choice = parse_pd_choice(text, self.cooperate_label, self.defect_label)
        if choice is None:
            raise ValueError(
                f"{agent_name} did not provide {self.cooperate_label} or {self.defect_label}."
            )
        return self.record_choice(agent_name, choice)

    def compute_payoffs(self, choices: dict[str, str]) -> dict[str, float]:
        return pd_payoff(choices, T=self.T, R=self.R, P=self.P, S=self.S)

    def _history_lines(self, actor_name: str, completed: list[int]) -> list[str]:
        parts = ["\nHistory:"]
        for round_idx in completed:
            choices = self.store.choices_at(round_idx)
            payoffs = self.store.payoffs_at(round_idx)
            others = [
                f"{name}:{self._label(choices.get(name, '?'))}"
                for name in self.agent_names
                if name != actor_name
            ]
            parts.append(
                f"  Round {round_idx + 1}: you={self._label(choices.get(actor_name, '?'))}"
                f"({payoffs.get(actor_name, 0.0):g})  "
                + ", ".join(others)
            )
        return parts

    def _history_table(self, actor_name: str, completed: list[int]) -> list[str]:
        others = [name for name in self.agent_names if name != actor_name]
        header = "round | you | " + " | ".join(others) + " | your payoff"
        parts = ["\nHistory:", "  " + header]
        for round_idx in completed:
            choices = self.store.choices_at(round_idx)
            payoffs = self.store.payoffs_at(round_idx)
            cells = [
                str(round_idx + 1),
                self._label(choices.get(actor_name, "?")),
                *[self._label(choices.get(name, "?")) for name in others],
                f"{payoffs.get(actor_name, 0.0):g}",
            ]
            parts.append("  " + " | ".join(cells))
        return parts

    def _history_summary(self, actor_name: str, completed: list[int]) -> list[str]:
        parts = [f"\nHistory after {len(completed)} round(s):"]
        total = sum(self.store.payoffs_at(s).get(actor_name, 0.0) for s in completed)
        for name in self.agent_names:
            coop = sum(
                1 for s in completed if self.store.choices_at(s).get(name) == "COOPERATE"
            )
            who = "you" if name == actor_name else name
            parts.append(
                f"  {who}: played {self.cooperate_label} {coop} of {len(completed)} round(s)."
            )
            if name == actor_name:
                last = self.store.choices_at(completed[-1]).get(name, "?")
                parts.append(f"  Your last choice: {self._label(last)}.")
        parts.append(f"  Your total score so far: {total:g}.")
        return parts

    def observe(self, actor_name: str, **kwargs: Any) -> str:
        del kwargs
        step = self.current_step()
        parts = [self._rules()]
        if self.framing_note:
            parts.append("\n" + self.framing_note)
        completed = [s for s in self.store.all_steps() if s < step and self.store.payoffs_at(s)]
        if not completed:
            parts.append("\nRound 1 — no previous results yet.")
        else:
            if self.history_format == "table":
                parts.extend(self._history_table(actor_name, completed))
            elif self.history_format == "summary_counts":
                parts.extend(self._history_summary(actor_name, completed))
            else:
                parts.extend(self._history_lines(actor_name, completed))
            parts.append(f"\nNow choose for round {step + 1}.")
        return "\n".join(parts)


class GameNextActingComponent:
    """Select every registered player each round.

    With ``acting_rounds`` set, no one acts once that many rounds have been
    played: scheduling one extra engine step (``num_steps = acting_rounds + 1``)
    lets the update phase finalize the LAST round's payoffs into the action
    log at zero extra LLM cost.
    """

    def __init__(
        self,
        *,
        agent_names: Sequence[str] = (),
        context: Any = None,
        acting_rounds: int | None = None,
        active_probability: float | None = None,
        activity_transition_rates: Mapping[str, Any] | None = None,
        min_active_agents: int | None = None,
    ) -> None:
        del active_probability, activity_transition_rates, min_active_agents
        self._agent_names = list(agent_names)
        self._context = context
        self._acting_rounds = int(acting_rounds) if acting_rounds is not None else None

    def _current_step(self) -> int:
        backend = getattr(self._context, "backend", None)
        if backend is None or not hasattr(backend, "current_step"):
            return 0
        return int(backend.current_step())

    def acting_agent_names(self) -> list[str]:
        if self._acting_rounds is not None and self._current_step() >= self._acting_rounds:
            return []
        return list(self._agent_names)


class GameObservationComponent:
    """Delegate observations to the repeated-game backend."""

    def __init__(
        self,
        *,
        backend: RepeatedGameBackendBase,
        observation_params: Mapping[str, Any] | None = None,
    ) -> None:
        del observation_params
        self._backend = backend

    def make_observation(self, agent_name: str) -> str:
        return self._backend.observe(agent_name)


class GameResolveComponent:
    """Resolve text or tool-call choices through the repeated-game backend."""

    def __init__(self, *, backend: RepeatedGameBackendBase) -> None:
        self._backend = backend

    def resolve_action(self, agent_name: str, action: ActionOutput) -> str:
        if action.tool_calls:
            return self._resolve_tool_call(agent_name, action.tool_calls[0])
        return self._backend.choose_from_text(agent_name, action.text)

    def _resolve_tool_call(self, agent_name: str, tool_call: ToolCall) -> str:
        args = dict(tool_call.arguments or {})
        if "agent_name" in args:
            raise ValueError("Game tool calls must not include runtime-owned agent_name.")
        name = str(tool_call.name)
        if isinstance(self._backend, MoneyGameBackend) and name in {
            "choose_number",
            "CHOOSE_NUMBER",
        }:
            if "number" not in args:
                raise ValueError("CHOOSE_NUMBER tool call requires number.")
            return self._backend.choose_number(agent_name, int(args["number"]))
        if isinstance(self._backend, IteratedPDBackend) and name in {
            "choose_pd_action",
            "CHOOSE_PD_ACTION",
        }:
            if "choice" not in args:
                raise ValueError("CHOOSE_PD_ACTION tool call requires choice.")
            return self._backend.choose_pd_action(agent_name, str(args["choice"]))
        raise ValueError(f"Unsupported game tool call {tool_call.name!r}.")


class GameUpdateComponent:
    """Finalize the previous round's payoffs through the backend."""

    def __init__(
        self,
        *,
        backend: RepeatedGameBackendBase,
        backend_type: str | None = None,
        default_recsys_type: str | None = None,
        update_every_n_steps: int | None = None,
        lazy: bool | None = None,
        max_posts: int | None = None,
        user_context_recent_posts: int | None = None,
        include_like_trace: bool | None = None,
        like_trace_window: int | None = None,
        like_trace_weight: float | None = None,
        include_like_trace_in_context: bool | None = None,
    ) -> None:
        del (
            backend_type,
            default_recsys_type,
            update_every_n_steps,
            lazy,
            max_posts,
            user_context_recent_posts,
            include_like_trace,
            like_trace_window,
            like_trace_weight,
            include_like_trace_in_context,
        )
        self._backend = backend

    def update(self, *, step: int, agents: Sequence[Any], context: Any = None) -> None:
        del context
        self._backend.update(step=step, agent_names=[agent.name for agent in agents])


BeautyContestObserveComponent = GameObservationComponent
BeautyContestResolveComponent = GameResolveComponent
BeautyContestUpdateComponent = GameUpdateComponent
PDObserveComponent = GameObservationComponent
PDResolveComponent = GameResolveComponent
PDUpdateComponent = GameUpdateComponent
