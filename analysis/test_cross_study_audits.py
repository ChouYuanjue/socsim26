from __future__ import annotations

import unittest
from pathlib import Path

from ipd_cira_analysis import load_released_parser_functions, parse_explicit_pd_parameter
from observed_norms_audit import strict_integer
from persona_expression_execution_audit import parse_create_tweet_calls
from polarization_cira_analysis import edge_alignment, parse_tool_call_names, unique_edges


class ReleasedParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        number_parser, pd_parser, audit = load_released_parser_functions(
            root / "references/socsim26_sharedtask/task_components/game.py"
        )
        cls.number_parser = staticmethod(number_parser)
        cls.pd_parser = staticmethod(pd_parser)
        cls.audit = audit

    def test_money_parser_prefers_first_legal_number(self) -> None:
        text = "The 20-point bonus makes 19 attractive.\nACTION: CHOOSE_NUMBER\nnumber: 19"
        self.assertEqual(self.number_parser(text), 20)

    def test_pd_parser_prefers_first_label(self) -> None:
        text = "I will avoid DEFECT.\nACTION: CHOOSE_PD_ACTION\nchoice: COOPERATE"
        self.assertEqual(self.pd_parser(text), "DEFECT")
        self.assertEqual(parse_explicit_pd_parameter(text, "COOPERATE", "DEFECT"), "COOPERATE")

    def test_relabelled_pd_parser(self) -> None:
        text = "Avoid BLUE.\nACTION: CHOOSE_PD_ACTION\nchoice: GREEN"
        self.assertEqual(self.pd_parser(text, "GREEN", "BLUE"), "DEFECT")
        self.assertEqual(parse_explicit_pd_parameter(text, "GREEN", "BLUE"), "COOPERATE")


class ToolCallTests(unittest.TestCase):
    def test_persona_multiple_create_tweets(self) -> None:
        output = "tool_calls:create_tweet({'status': 'a'}), create_tweet({'status': 'b'})"
        self.assertEqual(parse_create_tweet_calls(output), ["a", "b"])

    def test_polarization_call_names(self) -> None:
        output = "tool_calls:create_tweet({'status': 'a'}), like_tweet({'post_id': 2}), FINISHED()"
        self.assertEqual(parse_tool_call_names(output), ["create_tweet", "like_tweet", "FINISHED"])


class StructuredProbeTests(unittest.TestCase):
    def test_exact_integer_only(self) -> None:
        self.assertEqual(strict_integer(" 4 ", 1, 5), 4)
        self.assertIsNone(strict_integer("I choose 4", 1, 5))
        self.assertIsNone(strict_integer("7", 1, 5))


class NetworkMetricTests(unittest.TestCase):
    def test_edge_alignment(self) -> None:
        graph = {"0": ["1"], "1": ["0", "2"], "2": ["1"]}
        edges = unique_edges(graph)
        self.assertEqual(edges, [(0, 1), (1, 2)])
        self.assertAlmostEqual(edge_alignment([1.0, 1.0, 5.0], edges), 0.5)


if __name__ == "__main__":
    unittest.main()
