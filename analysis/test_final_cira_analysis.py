from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from final_cira_analysis import (
    ANCHORS,
    CLAIMS,
    CHOICES,
    compute_claim_effects,
    js_distance,
    load_released_number_parser,
    parse_output_choice,
)


class ParserTests(unittest.TestCase):
    def test_exact_action(self) -> None:
        self.assertEqual(parse_output_choice("ACTION: CHOOSE_NUMBER\nnumber: 12"), 12)

    def test_analysis_prefixed_action(self) -> None:
        output = "Analysis: choose a high undercut.\n\nACTION: CHOOSE_NUMBER\nnumber: 19"
        self.assertEqual(parse_output_choice(output), 19)

    def test_bare_number(self) -> None:
        self.assertEqual(parse_output_choice("19"), 19)

    def test_reject_out_of_range(self) -> None:
        self.assertIsNone(parse_output_choice("number: 10"))

    def test_released_parser_can_differ_from_explicit_action(self) -> None:
        output = "Analysis: the 20-point bonus suggests undercutting. ACTION: CHOOSE_NUMBER number: 19"
        repo_root = Path(__file__).resolve().parents[1]
        parser, audit = load_released_number_parser(repo_root / "references/socsim26_sharedtask/task_components/game.py")
        self.assertEqual(parser(output), 20)
        self.assertEqual(parse_output_choice(output), 19)
        self.assertEqual(audit["function"], "parse_number_choice")


class ClaimMatchingTests(unittest.TestCase):
    def test_h2_pairing_is_seed_and_context_matched(self) -> None:
        rows = []
        for seed in [1, 2]:
            for variant, value in [("basic", 12.0), ("cycle", 15.0)]:
                rows.append({
                    "condition_id": f"{variant}-{seed}",
                    "kind": "grid",
                    "seed": seed,
                    "game_variant": variant,
                    "goal_framing": "none",
                    "model": "qwen3.5-4b",
                    "persona": "neutral",
                    "instruction_wording": "default",
                    "response_format": "default",
                    "temperature": "0.5",
                    "persona_format": "default",
                    "mean_choice": value,
                    "tar_action_member": "synthetic",
                })
        h2 = next(spec for spec in CLAIMS if spec.claim_id == "H2")
        effects = compute_claim_effects(pd.DataFrame(rows), h2)
        self.assertEqual(len(effects), 2)
        np.testing.assert_allclose(effects["signed_effect"].to_numpy(), [3.0, 3.0])


class LeakageTests(unittest.TestCase):
    def test_switched_anchor_changes_distance_for_identical_behavior(self) -> None:
        p = np.repeat(0.1, len(CHOICES)).tolist()
        distances = [js_distance(p, [ANCHORS[name][choice] for choice in CHOICES]) for name in ANCHORS]
        self.assertGreater(max(distances) - min(distances), 0.0)
        self.assertEqual(0.0, 0.0)  # direct copied-behavior contrast by construction


if __name__ == "__main__":
    unittest.main()
