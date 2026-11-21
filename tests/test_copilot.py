"""Checks for the deterministic parts of copilot/ (no models, no network).  Run:  python -m unittest -v"""
import json
import unittest

from app import scrub
from copilot import config, metrics, prompts
from copilot.judge import QUESTION_TYPES, gold_type

GOLD = [json.loads(line) for line in open(config.GOLD_PATH, encoding="utf-8")]


class TestMetrics(unittest.TestCase):
    def test_every_gold_answer_matches_itself(self):
        for g in GOLD:
            if g["category"] == "math_logic":
                self.assertTrue(metrics.exact_match(g["answer"], g["answer"]), g["id"])

    def test_key_phrase_takes_the_value_after_ie(self):
        self.assertEqual(metrics.key_phrase("Within 45 days of the order, i.e. by 15 April 2024 (Section 41)."),
                         "15 April 2024")
        self.assertEqual(metrics.key_phrase("State Consumer Disputes Redressal Commission. The consideration"),
                         "State Consumer Disputes Redressal Commission")

    def test_exact_match_ignores_case_and_punctuation(self):
        self.assertTrue(metrics.exact_match("You must pay Rs. 15 lakh.", "50% of the amount, i.e. ₹15 lakh (s 51)"))
        self.assertFalse(metrics.exact_match("You must pay ₹25 lakh.", "50% of the amount, i.e. ₹15 lakh (s 51)"))

    def test_level_of(self):
        self.assertEqual(metrics.level_of("State Commission"), "State")
        self.assertEqual(metrics.level_of("NCDRC"), "National")
        self.assertEqual(metrics.level_of("District or State Commission"), "Unclear")
        self.assertEqual(metrics.level_of("RERA"), "Unclear")

    def test_prf(self):
        self.assertEqual(metrics.prf(tp=2, fp=0, fn=2), dict(precision=1.0, recall=0.5, f1=2 / 3))
        self.assertEqual(metrics.prf(tp=0, fp=0, fn=0)["f1"], 0.0)


class TestJudge(unittest.TestCase):
    def test_gold_type_covers_every_question(self):
        for g in GOLD:
            self.assertIn(gold_type(g), QUESTION_TYPES, g["id"])

    def test_judge_prompt_never_names_the_gold_label(self):
        from copilot import judge
        self.assertNotIn("subtype", judge.SYSTEM)
        self.assertNotIn("category", judge.SYSTEM)


class TestPrompts(unittest.TestCase):
    def test_zero_shot_without_rag_is_the_week1_prompt(self):
        messages = prompts.build_messages("Q?", "zero")
        self.assertEqual(messages, [{"role": "system", "content": prompts.SYSTEM_PROMPT},
                                    {"role": "user", "content": "Q?"}])

    def test_rag_puts_numbered_passages_before_the_question(self):
        user = prompts.build_messages("Q?", "cot", ["first", "second"])[-1]["content"]
        self.assertLess(user.index("[1] first"), user.index("[2] second"))
        self.assertLess(user.index("[2] second"), user.index("QUESTION\nQ?"))
        self.assertTrue(user.endswith(prompts.COT_INSTRUCTION))

    def test_few_shot_examples_are_not_gold_questions(self):
        questions = {g["question"] for g in GOLD}
        for q, _ in prompts.FEW_SHOT_EXAMPLES:
            self.assertNotIn(q, questions)


class TestBM25Words(unittest.TestCase):
    def test_case_and_punctuation_are_ignored(self):
        from copilot.rag import bm25_words
        self.assertEqual(bm25_words("Under Section 35(1), the NCDRC."), ["under", "section", "35", "1", "the", "ncdrc"])


class TestScrub(unittest.TestCase):
    def test_phone_and_email_are_removed(self):
        self.assertEqual(scrub("call 98765 43210 or +91-9876543210, mail a.b@x.com"),
                         "call [phone] or [phone], mail [email]")

    def test_amounts_are_kept(self):
        self.assertEqual(scrub("I paid ₹50,00,001 on 12/03/2024"), "I paid ₹50,00,001 on 12/03/2024")


if __name__ == "__main__":
    unittest.main()
