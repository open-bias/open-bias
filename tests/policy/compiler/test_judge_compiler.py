import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from opensentinel.policy.compiler.protocol import CompilationResult
from opensentinel.policy.engines.judge.compiler import JudgeCompiler

@pytest.fixture
def judge_compiler():
    return JudgeCompiler(model="gpt-4o-mini", api_key="test-key")

async def test_judge_compiler_registration():
    from opensentinel.policy.engines.judge.engine import JudgePolicyEngine
    engine = JudgePolicyEngine()
    compiler = engine.get_compiler()
    assert isinstance(compiler, JudgeCompiler)

def test_get_compiler_via_engine():
    """get_compiler() via engine instance should return a JudgeCompiler."""
    from opensentinel.policy.engines.judge.engine import JudgePolicyEngine
    engine = JudgePolicyEngine()
    compiler = engine.get_compiler()
    assert isinstance(compiler, JudgeCompiler)

def test_get_compiler_forwards_kwargs():
    """get_compiler(model=..., api_key=...) should forward to JudgeCompiler."""
    from opensentinel.policy.engines.judge.engine import JudgePolicyEngine
    engine = JudgePolicyEngine()
    compiler = engine.get_compiler(model="gpt-4o", api_key="sk-test", base_url="http://localhost:8080")
    assert isinstance(compiler, JudgeCompiler)
    assert compiler.model == "gpt-4o"
    assert compiler._api_key == "sk-test"
    assert compiler._base_url == "http://localhost:8080"

async def test_build_compilation_prompt(judge_compiler):
    policy = "Be professional and never share PII."
    prompt = judge_compiler._build_compilation_prompt(policy, {"domain": "customer support"})
    
    assert "Be professional and never share PII." in prompt
    assert "customer support" in prompt
    assert "Generate a JSON object" in prompt

async def test_parse_valid_response(judge_compiler):
    response = {
        "rubrics": [
            {
                "name": "safety",
                "description": "Safety checks",
                "scope": "turn",
                "evaluation_type": "pointwise",
                "pass_threshold": 0.8,
                "fail_action": "block",
                "criteria": [
                    {
                        "name": "no_pii",
                        "description": "No PII",
                        "scale": "binary",
                        "weight": 1.0,
                        "fail_threshold": 0.5
                    }
                ]
            }
        ]
    }
    
    result = judge_compiler._parse_compilation_response(response, "test policy")
    
    assert result.success
    assert len(result.config["rubrics"]) == 1
    assert result.config["rubrics"][0]["name"] == "safety"
    assert result.metadata["rubric_count"] == 1
    assert result.metadata["criteria_count"] == 1

async def test_parse_response_with_invalid_scale(judge_compiler):
    response = {
        "rubrics": [
            {
                "name": "test",
                "criteria": [
                    {
                        "name": "bad_scale",
                        "scale": "invalid_scale_name"
                    }
                ]
            }
        ]
    }
    
    result = judge_compiler._parse_compilation_response(response, "test policy")
    
    assert result.success
    # Should fall back to likert_5
    assert result.config["rubrics"][0]["criteria"][0]["scale"] == "likert_5"
    assert any("invalid_scale_name" in w for w in result.warnings)

async def test_parse_empty_response_fails(judge_compiler):
    response = {"rubrics": []}
    result = judge_compiler._parse_compilation_response(response, "test policy")
    assert not result.success
    assert "No rubrics generated" in result.errors[0]

@patch("opensentinel.policy.compiler.base.LLMPolicyCompiler._call_llm")
async def test_compile_integration(mock_call, judge_compiler):
    mock_call.return_value = '{"rubrics": [{"name": "test", "criteria": [{"name": "c1"}]}]}'
    
    result = await judge_compiler.compile("Be nice")
    
    assert result.success
    assert result.config["rubrics"][0]["name"] == "test"
    mock_call.assert_called_once()

def test_export_creates_yaml_file(judge_compiler, tmp_path):
    output_path = tmp_path / "policy.yaml"
    result = CompilationResult(
        success=True,
        config={
            "rubrics": [
                {
                    "name": "test",
                    "criteria": [{"name": "c1", "scale": "binary", "fail_threshold": None}]
                }
            ]
        }
    )
    
    judge_compiler.export(result, output_path)
    
    assert output_path.exists()
    content = output_path.read_text()
    assert "rubrics:" in content
    assert "test" in content
    # Should have removed None fail_threshold
    assert "fail_threshold" not in content

def test_validate_valid_result(judge_compiler):
    result = CompilationResult(
        success=True,
        config={
            "rubrics": [
                {
                    "name": "safety",
                    "pass_threshold": 0.8,
                    "criteria": [
                        {"name": "c1", "scale": "binary"}
                    ]
                }
            ]
        }
    )

    errors = judge_compiler.validate_result(result)
    assert not errors

def test_validate_invalid_result(judge_compiler):
    result = CompilationResult(
        success=True,
        config={
            "rubrics": [
                {
                    "name": "safety",
                    "pass_threshold": 1.5,
                    "criteria": [
                        {"name": "c1", "scale": "wrong_scale"}
                    ]
                }
            ]
        }
    )

    errors = judge_compiler.validate_result(result)
    assert len(errors) == 2
    assert any("pass_threshold" in e for e in errors)
    assert any("wrong_scale" in e for e in errors)
