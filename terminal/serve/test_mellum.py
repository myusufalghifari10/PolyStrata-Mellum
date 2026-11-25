"""Mellum 2.1 behind the server: its pre-tokenizer (`mellum2`), backend/setup registration, ChatML template with
<think></think> reasoning, and JSON <tool_call> parsing.

    python -m unittest serve.test_mellum
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import setup  # noqa: E402
import strata_inspect as SI  # noqa: E402
import strata_tokenizer as ST  # noqa: E402
from serve.frontend import ChatTemplate, JsonOutputParser, parser_for_template  # noqa: E402
from serve.models import BACKENDS  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service  # noqa: E402

MELLUM_DIR = Path("/home/yusuf/AI/Model/Mellum")


def _load_mellum_tokenizer() -> ST.Tokenizer:
    raw = json.loads((MELLUM_DIR / "tokenizer.json").read_text(encoding="utf-8"))
    vocab = raw["model"]["vocab"]
    tokens = [None] * len(vocab)
    for tok, idx in vocab.items():
        tokens[idx] = tok
    merges = [" ".join(m) if isinstance(m, list) else m for m in raw["model"]["merges"]]
    token_types = [1] * len(tokens)
    for entry in raw.get("added_tokens", []):
        tid = int(entry["id"])
        token_types[tid] = 3 if entry.get("special", False) else 4
    special_ids = {
        "tokenizer.ggml.bos_token_id": 0,
        "tokenizer.ggml.eos_token_id": 28,
    }
    return ST.Tokenizer(tokens, merges, token_types=token_types, pre="mellum2", special_ids=special_ids)


class Registration(unittest.TestCase):
    def test_backend_and_inspect_and_setup_mappings(self):
        self.assertEqual(BACKENDS.get("mellum"), "mellum")
        self.assertEqual(SI.POLY_ARCHS.get("mellum"), "Mellum2.1-12B-A2.5B")
        self.assertEqual(setup.POLY_GGUF_ARCHS.get("mellum"), ("mellum",))
        self.assertEqual(ST.pre_tokenizer_kind("mellum2"), "poly")


class PreTokenizerPieces(unittest.TestCase):
    def pieces(self, text: str) -> list[str]:
        return ST.Tokenizer(["a"], [], pre="mellum2")._pieces(text)

    def test_digits_are_split_individually(self):
        self.assertEqual(self.pieces("12345 it's"), ["1", "2", "3", "4", "5", " it", "'s"])
        self.assertEqual(self.pieces("v2.10_beta"), ["v", "2", ".", "1", "0", "_", "beta"])

    def test_python_and_ascii_and_unicode(self):
        self.assertEqual(self.pieces("def f(x): return x + 42\n"),
                         ["def", " f", "(", "x", "):", " return", " x", " +", " ", "4", "2", "\n"])
        self.assertEqual(self.pieces("Halo, dunia! Apa kabar?"),
                         ["Halo", ",", " dunia", "!", " Apa", " kabar", "?"])


class MellumTokenizerRoundTrip(unittest.TestCase):
    @unittest.skipUnless((MELLUM_DIR / "tokenizer.json").exists(), "Mellum tokenizer.json not present")
    def test_round_trip_and_chatml_specials(self):
        tk = _load_mellum_tokenizer()
        samples = [
            "Hello, world!",
            "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    return arr\n",
            "Numbers: 0 123 9876543210 and 3.14159",
            "Selamat pagi! Ini uji coba Bahasa Indonesia dengan angka 2025.",
            "<|im_start|>user\nExplain MoE.<|im_end|>\n<|im_start|>assistant\n<think>\nstep 1\n</think>\n",
            '<tool_call>\n{"name": "read_file", "arguments": {"path": "main.py"}}\n</tool_call>',
        ]
        for s in samples:
            with self.subTest(sample=s[:40]):
                ids = tk.encode(s, parse_special=True)
                self.assertGreater(len(ids), 0)
                self.assertEqual(tk.decode(ids), s)
        # Verify special & user-defined token ids are preserved atomically
        self.assertEqual(tk.encode("<|im_start|>", parse_special=True), [27])
        self.assertEqual(tk.encode("<|im_end|>", parse_special=True), [28])
        self.assertEqual(tk.encode("<think>", parse_special=True), [23])
        self.assertEqual(tk.encode("</think>", parse_special=True), [24])
        self.assertEqual(tk.encode("<tool_call>", parse_special=True), [29])
        self.assertEqual(tk.encode("</tool_call>", parse_special=True), [30])


class MellumTokens(ByteTokenizer):
    SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>", "</tool_call>"]
    ALWAYS = ("<think>", "</think>", "<tool_call>", "</tool_call>")
    pre = "mellum2"
    special_ids = {"tokenizer.ggml.bos_token_id": 256, "tokenizer.ggml.eos_token_id": 258}


class ServiceAndToolCalls(unittest.TestCase):
    SCHEMA = [{"name": "get_weather", "parameters": {"properties": {"city": {"type": "string"},
                                                                    "days": {"type": "integer"}}}}]
    TEXT = ('checking weather\n</think>\n\nLet me look.\n'
            '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Jakarta", "days": 2}}\n</tool_call>')
    SELF_OPENED_TEXT = ('<think>\nchecking weather\n</think>\n\nLet me look.\n'
                        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Jakarta", "days": 2}}\n</tool_call>')

    def test_json_tool_call_and_think_parsing(self):
        for text in (self.TEXT, self.SELF_OPENED_TEXT):
            for step in (1, 7, 10_000):
                with self.subTest(self_opened=text.startswith("<think>"), step=step):
                    p = JsonOutputParser(thinking=True, tools=self.SCHEMA, call_format="function")
                    evs = [e for i in range(0, len(text), step) for e in p.feed(text[i:i + step])] + p.finish()
                    calls = [e.call for e in evs if e.kind == "tool_call"]
                    self.assertEqual([(c.name, c.arguments) for c in calls],
                                     [("get_weather", {"city": "Jakarta", "days": 2})])
                    reasoning = "".join(e.text for e in evs if e.kind == "reasoning")
                    self.assertFalse(reasoning.startswith("<think>"), repr(reasoning))
                    self.assertEqual(reasoning, "checking weather\n")
                    self.assertEqual("".join(e.text for e in evs if e.kind == "content").strip(), "Let me look.")

    def test_json_parser_non_thinking_content(self):
        text = ('Let me look.\n'
                '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Jakarta", "days": 2}}\n</tool_call>')
        for step in (1, 7, 10_000):
            with self.subTest(step=step):
                p = JsonOutputParser(thinking=False, tools=self.SCHEMA, call_format="function")
                evs = [e for i in range(0, len(text), step) for e in p.feed(text[i:i + step])] + p.finish()
                self.assertEqual([e.text for e in evs if e.kind == "reasoning"], [])
                self.assertEqual("".join(e.text for e in evs if e.kind == "content").strip(), "Let me look.")
                calls = [e.call for e in evs if e.kind == "tool_call"]
                self.assertEqual([(c.name, c.arguments) for c in calls],
                                 [("get_weather", {"city": "Jakarta", "days": 2})])

    @unittest.skipUnless((MELLUM_DIR / "chat_template.jinja").exists(), "Mellum chat_template.jinja not present")
    def test_service_uses_json_parser_and_chatml_stops(self):
        tpl_src = (MELLUM_DIR / "chat_template.jinja").read_text(encoding="utf-8")
        self.assertIs(parser_for_template(tpl_src), JsonOutputParser)
        tok = MellumTokens()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "chat_template.jinja"
            p.write_text(tpl_src, encoding="utf-8")
            svc = Service(MockEngine(tok, "hi"), tok, ChatTemplate(p))
        self.assertIs(svc.parser_cls, JsonOutputParser)
        self.assertEqual(svc.call_format, "function")
        self.assertEqual(svc.stop_ids, {256, 258})
        self.assertEqual(svc.turn_ids, frozenset({257}))
        prompt_think = svc.render_prompt([{"role": "user", "content": "hi"}], None, {})
        self.assertTrue(prompt_think.endswith("<|im_start|>assistant\n"))
        prompt_nothink = svc.render_prompt([{"role": "user", "content": "hi"}], None, {"enable_thinking": False})
        self.assertTrue(prompt_nothink.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))


if __name__ == "__main__":
    unittest.main()
