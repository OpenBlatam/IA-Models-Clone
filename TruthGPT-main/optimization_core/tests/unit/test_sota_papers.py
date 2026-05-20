import pytest
import torch
import asyncio
from typing import Dict, Any

from papers.chain_of_draft import ChainOfDraft
from papers.elastic_reasoning import ElasticReasoning
from papers.fp16_stability import FP16Stability
from papers.self_consistency import SelfConsistency
from papers.speculative_decoding import SpeculativeDecoder
from papers.mcts_reasoning import MCTSReasoner, ThoughtNode, RewardEstimator

# 1. Chain of Draft Tests
def test_chain_of_draft():
    # Test variants retrieval
    for variant in ChainOfDraft.VARIANTS:
        template = ChainOfDraft.get_template(variant)
        assert isinstance(template, str)
        assert len(template) > 0
        
    # Test unknown variant fallback
    fallback = ChainOfDraft.get_template("unknown_variant")
    assert fallback == ChainOfDraft.get_template("baseline")
    
    # Test validation
    valid_draft = "Drafting steps:\n\u2022 1. [Start reasoning]\n\u2022 2. [Next reasoning]\nSolution:\n42"
    invalid_draft = "Drafting steps:\n\u2022 1. This is a very long draft text that exceeds the word limit set by the validation function in order to verify that it correctly flags long sentences\nSolution:\n42"
    
    assert ChainOfDraft.validate_draft(valid_draft) is True
    assert ChainOfDraft.validate_draft(invalid_draft) is False

# 2. Elastic Reasoning Tests
def test_elastic_reasoning():
    er = ElasticReasoning(t_budget=5, s_budget=10)
    assert er.t_budget == 5
    assert er.s_budget == 10
    assert er.total_budget == 15
    
    # Test simulate_generation thinking limit not reached
    tokens = ["<think>", "step", "one"]
    res = er.simulate_generation(tokens)
    assert res == "continue"
    
    # Test simulate_generation thinking limit reached
    tokens = ["<think>", "step", "one", "two", "three", "four", "five"]
    res = er.simulate_generation(tokens)
    assert res == "</think>"
    
    # Test simulate_generation already finished thinking
    tokens = ["<think>", "step", "one", "</think>", "solution"]
    res = er.simulate_generation(tokens)
    assert res == "continue"
    
    # Test metrics calculation
    text = "<think>reasoning steps go here</think> solution value"
    metrics = ElasticReasoning.calculate_metrics(text)
    assert metrics["has_thinking"] is True
    assert metrics["think_tokens"] == 4
    assert metrics["total_tokens"] == 6
    assert abs(metrics["ratio"] - 4/6) < 1e-4

# 3. FP16 Stability Tests
def test_fp16_stability():
    # Test stability boundaries
    stable_tensor = torch.tensor([1.0, 2.5, 100.0])
    overflow_tensor = torch.tensor([10.0, 70000.0]) # Exceeds 65504.0
    underflow_tensor = torch.tensor([1e-6, 0.5]) # Between 0 and 6.1e-5
    
    res_stable = FP16Stability.check_stability_metrics(stable_tensor)
    assert res_stable["stable"] is True
    assert res_stable["is_overflow"] is False
    assert res_stable["is_underflow"] is False
    
    res_overflow = FP16Stability.check_stability_metrics(overflow_tensor)
    assert res_overflow["stable"] is False
    assert res_overflow["is_overflow"] is True
    
    res_underflow = FP16Stability.check_stability_metrics(underflow_tensor)
    assert res_underflow["stable"] is False
    assert res_underflow["is_underflow"] is True
    
    # Test mathematical utility functions
    policy = torch.tensor([[0.8, 0.2]])
    rewards = torch.tensor([[1.0, -1.0]])
    obj = FP16Stability.objective_function(policy, rewards)
    assert isinstance(obj.item(), float)
    
    policy_new = torch.tensor([[0.9, 0.1]])
    policy_old = torch.tensor([[0.8, 0.2]])
    adv = torch.tensor([[0.5, -0.5]])
    is_loss = FP16Stability.importance_sampling_correction(policy_new, policy_old, adv)
    assert isinstance(is_loss.item(), float)
    
    tis_loss = FP16Stability.truncated_is(policy_new, policy_old, adv, clip_c=1.1)
    assert isinstance(tis_loss.item(), float)

# 4. Self-Consistency Tests
def test_self_consistency_extractors():
    sc = SelfConsistency(n_samples=3)
    
    # last_line extractor
    assert sc.extract_answer_last_line("Reasoning\nAnswer is 5") == "Answer is 5"
    assert sc.extract_answer_last_line("") == ""
    
    # boxed extractor
    assert sc.extract_answer_boxed("The answer is \\boxed{42} in math mode") == "42"
    assert sc.extract_answer_boxed("No boxed answer") == "No boxed answer"
    
    # json extractor
    assert sc.extract_answer_json('{"final_answer": "100"}') == "100"
    assert sc.extract_answer_json('{"answer": "200"}') == "200"
    assert sc.extract_answer_json('Invalid JSON') == "Invalid JSON"
    
    # tagged extractor
    assert sc.extract_answer_tagged("Prefix <answer>Yes</answer> Suffix") == "Yes"
    assert sc.extract_answer_tagged("No tags") == "No tags"

def test_self_consistency_voting():
    sc = SelfConsistency()
    
    # Majority vote
    answers = ["42", "42", "24", "42", "12"]
    best, conf = sc.majority_vote(answers)
    assert best == "42"
    assert conf == 0.6
    
    # Weighted vote
    weighted_answers = ["yes", "no", "yes"]
    scores = [0.8, 0.9, 0.4] # yes total = 1.2, no total = 0.9
    best_w, conf_w = sc.weighted_vote(weighted_answers, scores)
    assert best_w == "yes"
    assert abs(conf_w - 1.2/2.1) < 1e-4
    
    # Agreement score
    assert SelfConsistency.agreement_score(answers) == 0.6

@pytest.mark.asyncio
async def test_self_consistency_sample_and_vote():
    sc = SelfConsistency(n_samples=3, answer_extraction="last_line")
    
    async def mock_llm(prompt, **kwargs):
        await asyncio.sleep(0.001)
        return "Thinking...\n42"
        
    res = await sc.sample_and_vote("What is 6 * 7?", mock_llm)
    assert res["best_answer"] == "42"
    assert res["confidence"] == 1.0
    assert len(res["all_answers"]) == 3
    assert res["fallback"] is False

# 5. Speculative Decoding Tests
def test_speculative_confidence_estimator():
    # Good response
    good_resp = "The solution to the problem is simple. We can calculate it step-by-step. First, let's look at the given parameters. Because of the structure, we can determine the solution. Solution: ```python\nprint(42)\n```"
    confidence = SpeculativeDecoder.estimate_confidence(good_resp, "Calculate print 42 parameter")
    assert confidence > 0.6
    
    # Poor response
    poor_resp = "I apology as an ai, unfortunately i'm not sure."
    poor_confidence = SpeculativeDecoder.estimate_confidence(poor_resp, "calculate print 42")
    assert poor_confidence < 0.4

@pytest.mark.asyncio
async def test_speculative_call_accepted():
    sd = SpeculativeDecoder(confidence_threshold=0.6)
    
    async def mock_draft(prompt, **kwargs):
        return "Draft answer. Because of the details, therefore we solve it step-by-step. First, done. ```code```"
        
    async def mock_target(prompt, **kwargs):
        raise RuntimeError("Target should not be called!")
        
    res = await sd.speculative_call("Solve it", mock_draft, mock_target)
    assert res["model_used"] == "draft"
    assert res["cost_tier"] == "low"
    assert sd.get_stats()["draft_accepted"] == 1

@pytest.mark.asyncio
async def test_speculative_call_escalated():
    sd = SpeculativeDecoder(confidence_threshold=0.8) # High threshold forces target call
    
    async def mock_draft(prompt, **kwargs):
        return "Draft answer with low confidence."
        
    async def mock_target(prompt, **kwargs):
        return "Target high quality answer. First, second, finally, because of parameters, therefore we have the solution."
        
    res = await sd.speculative_call("Solve complex math", mock_draft, mock_target)
    assert res["model_used"] == "target"
    assert res["cost_tier"] == "high"
    assert sd.get_stats()["target_verified"] == 1

# 6. MCTS Reasoning Tests
def test_mcts_reward_estimator():
    estimator = RewardEstimator()
    
    # Test reward estimation for coherent thoughts
    thought = "Step 1: Therefore, given the variables, we calculate first. This means the solution is 42."
    reward = estimator.estimate(thought, "variables calculation", depth=1)
    assert 0.0 <= reward <= 1.0
    
    empty_reward = estimator.estimate("", "prompt")
    assert empty_reward == 0.0

@pytest.mark.asyncio
async def test_mcts_reasoner_search():
    mcts = MCTSReasoner(max_iterations=3, max_depth=2, branching_factor=2)
    
    async def mock_llm(prompt, **kwargs):
        return "Step: Let's calculate variables. Answer: 42"
        
    res = await mcts.search("Solve simple equation", mock_llm)
    assert "best_path" in res
    assert "best_answer" in res
    assert res["best_reward"] > 0.0
