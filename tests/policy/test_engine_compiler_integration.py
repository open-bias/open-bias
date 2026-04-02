"""
Tests for engine.get_compiler() integration.

Verifies that each engine returns the correct compiler type (or None)
via the new get_compiler() method on PolicyEngine.
"""

import pytest

from openbias.policy.engines.fsm.engine import FSMPolicyEngine
from openbias.policy.engines.judge.engine import JudgePolicyEngine
from openbias.policy.engines.llm.engine import LLMPolicyEngine
from openbias.policy.engines.nemo.engine import NemoGuardrailsPolicyEngine
from openbias.policy.compiler.protocol import PolicyCompiler


class TestFSMGetCompiler:
    """FSMPolicyEngine.get_compiler() returns an FSMCompiler."""

    def test_returns_compiler(self):
        engine = FSMPolicyEngine()
        compiler = engine.get_compiler()
        assert compiler is not None
        assert isinstance(compiler, PolicyCompiler)
        assert compiler.engine_type == "fsm"

    def test_returns_fsm_compiler_type(self):
        from openbias.policy.engines.fsm.compiler import FSMCompiler

        engine = FSMPolicyEngine()
        compiler = engine.get_compiler()
        assert isinstance(compiler, FSMCompiler)

    def test_returns_new_instance_each_call(self):
        engine = FSMPolicyEngine()
        c1 = engine.get_compiler()
        c2 = engine.get_compiler()
        assert c1 is not c2


class TestJudgeGetCompiler:
    """JudgePolicyEngine.get_compiler() returns a JudgeCompiler."""

    def test_returns_compiler(self):
        engine = JudgePolicyEngine()
        compiler = engine.get_compiler()
        assert compiler is not None
        assert isinstance(compiler, PolicyCompiler)
        assert compiler.engine_type == "judge"

    def test_returns_judge_compiler_type(self):
        from openbias.policy.engines.judge.compiler import JudgeCompiler

        engine = JudgePolicyEngine()
        compiler = engine.get_compiler()
        assert isinstance(compiler, JudgeCompiler)


class TestLLMGetCompiler:
    """LLMPolicyEngine.get_compiler() returns an LLMCompiler."""

    def test_returns_compiler(self):
        engine = LLMPolicyEngine()
        compiler = engine.get_compiler()
        assert compiler is not None
        assert isinstance(compiler, PolicyCompiler)
        assert compiler.engine_type == "llm"

    def test_returns_llm_compiler_type(self):
        from openbias.policy.engines.llm.compiler import LLMCompiler

        engine = LLMPolicyEngine()
        compiler = engine.get_compiler()
        assert isinstance(compiler, LLMCompiler)

    def test_returns_new_instance_each_call(self):
        engine = LLMPolicyEngine()
        c1 = engine.get_compiler()
        c2 = engine.get_compiler()
        assert c1 is not c2


class TestNemoGetCompiler:
    """NemoGuardrailsPolicyEngine.get_compiler() returns a NemoCompiler."""

    def test_returns_compiler(self):
        engine = NemoGuardrailsPolicyEngine()
        compiler = engine.get_compiler()
        assert compiler is not None
        assert isinstance(compiler, PolicyCompiler)
        assert compiler.engine_type == "nemo"

    def test_returns_nemo_compiler_type(self):
        from openbias.policy.engines.nemo.compiler import NemoCompiler

        engine = NemoGuardrailsPolicyEngine()
        compiler = engine.get_compiler()
        assert isinstance(compiler, NemoCompiler)


