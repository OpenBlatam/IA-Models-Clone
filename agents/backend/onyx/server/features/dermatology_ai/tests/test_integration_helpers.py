"""
Integration Testing Helpers
Specialized helpers for integration testing
"""

import pytest
from typing import Any, Dict, List, Optional, Callable
from unittest.mock import Mock, AsyncMock
import asyncio
from datetime import datetime


class IntegrationTestBuilder:
    """Builder for integration test scenarios"""
    
    def __init__(self):
        self.scenario = {
            "setup": [],
            "steps": [],
            "teardown": [],
            "mocks": {},
            "assertions": []
        }
    
    def add_setup(self, func: Callable, *args, **kwargs) -> 'IntegrationTestBuilder':
        """Add setup step"""
        self.scenario["setup"].append({
            "func": func,
            "args": args,
            "kwargs": kwargs
        })
        return self
    
    def add_step(self, name: str, func: Callable, *args, **kwargs) -> 'IntegrationTestBuilder':
        """Add test step"""
        self.scenario["steps"].append({
            "name": name,
            "func": func,
            "args": args,
            "kwargs": kwargs
        })
        return self
    
    def add_teardown(self, func: Callable, *args, **kwargs) -> 'IntegrationTestBuilder':
        """Add teardown step"""
        self.scenario["teardown"].append({
            "func": func,
            "args": args,
            "kwargs": kwargs
        })
        return self
    
    def add_mock(self, name: str, mock: Any) -> 'IntegrationTestBuilder':
        """Add mock to scenario"""
        self.scenario["mocks"][name] = mock
        return self
    
    def add_assertion(self, assertion: Callable, *args, **kwargs) -> 'IntegrationTestBuilder':
        """Add assertion"""
        self.scenario["assertions"].append({
            "func": assertion,
            "args": args,
            "kwargs": kwargs
        })
        return self
    
    async def execute(self) -> Dict[str, Any]:
        """Execute integration test scenario"""
        results = {
            "setup_results": [],
            "step_results": [],
            "teardown_results": [],
            "assertion_results": []
        }
        
        # Execute setup
        for setup in self.scenario["setup"]:
            if asyncio.iscoroutinefunction(setup["func"]):
                result = await setup["func"](*setup["args"], **setup["kwargs"])
            else:
                result = setup["func"](*setup["args"], **setup["kwargs"])
            results["setup_results"].append(result)
        
        # Execute steps
        for step in self.scenario["steps"]:
            if asyncio.iscoroutinefunction(step["func"]):
                result = await step["func"](*step["args"], **step["kwargs"])
            else:
                result = step["func"](*step["args"], **step["kwargs"])
            results["step_results"].append({
                "name": step["name"],
                "result": result
            })
        
        # Execute assertions
        for assertion in self.scenario["assertions"]:
            if asyncio.iscoroutinefunction(assertion["func"]):
                result = await assertion["func"](*assertion["args"], **assertion["kwargs"])
            else:
                result = assertion["func"](*assertion["args"], **assertion["kwargs"])
            results["assertion_results"].append(result)
        
        # Execute teardown
        for teardown in self.scenario["teardown"]:
            if asyncio.iscoroutinefunction(teardown["func"]):
                result = await teardown["func"](*teardown["args"], **teardown["kwargs"])
            else:
                result = teardown["func"](*teardown["args"], **teardown["kwargs"])
            results["teardown_results"].append(result)
        
        return results


class FlowTester:
    """Tester for complete flows"""
    
    @staticmethod
    async def test_flow(
        steps: List[Dict[str, Any]],
        expected_results: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """Test a complete flow"""
        results = []
        errors = []
        
        for i, step in enumerate(steps):
            try:
                func = step["func"]
                args = step.get("args", ())
                kwargs = step.get("kwargs", {})
                
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                results.append({
                    "step": i + 1,
                    "name": step.get("name", f"step_{i+1}"),
                    "result": result,
                    "success": True
                })
                
                # Check expected result
                if expected_results and i < len(expected_results):
                    expected = expected_results[i]
                    if result != expected:
                        errors.append({
                            "step": i + 1,
                            "expected": expected,
                            "actual": result
                        })
            except Exception as e:
                results.append({
                    "step": i + 1,
                    "name": step.get("name", f"step_{i+1}"),
                    "error": str(e),
                    "success": False
                })
                errors.append({
                    "step": i + 1,
                    "error": str(e)
                })
        
        return {
            "results": results,
            "errors": errors,
            "success": len(errors) == 0
        }


class StateChecker:
    """Checker for state consistency"""
    
    @staticmethod
    def check_state_consistency(
        states: List[Dict[str, Any]],
        consistency_rules: List[Callable]
    ) -> Dict[str, Any]:
        """Check state consistency across multiple states"""
        violations = []
        
        for rule in consistency_rules:
            try:
                result = rule(states)
                if not result:
                    violations.append({
                        "rule": rule.__name__,
                        "violation": "Rule failed"
                    })
            except Exception as e:
                violations.append({
                    "rule": rule.__name__,
                    "error": str(e)
                })
        
        return {
            "consistent": len(violations) == 0,
            "violations": violations
        }
    
    @staticmethod
    def assert_state_transition(
        initial_state: Dict[str, Any],
        final_state: Dict[str, Any],
        expected_changes: List[str]
    ):
        """Assert state transition is correct"""
        changes = []
        for key in expected_changes:
            if key in initial_state and key in final_state:
                if initial_state[key] != final_state[key]:
                    changes.append(key)
            elif key not in initial_state and key in final_state:
                changes.append(key)
        
        assert set(changes) == set(expected_changes), \
            f"State changes {changes} do not match expected {expected_changes}"


class DataConsistencyChecker:
    """Checker for data consistency"""
    
    @staticmethod
    def check_data_consistency(
        data_sources: List[Dict[str, Any]],
        key_mapping: Dict[str, str]
    ) -> Dict[str, Any]:
        """Check data consistency across sources"""
        inconsistencies = []
        
        # Get all unique keys
        all_keys = set()
        for source in data_sources:
            all_keys.update(source.keys())
        
        # Check each key
        for key in all_keys:
            values = []
            for source in data_sources:
                mapped_key = key_mapping.get(key, key)
                if mapped_key in source:
                    values.append(source[mapped_key])
            
            # Check if all values are the same
            if len(set(values)) > 1:
                inconsistencies.append({
                    "key": key,
                    "values": values
                })
        
        return {
            "consistent": len(inconsistencies) == 0,
            "inconsistencies": inconsistencies
        }


# Convenience exports
IntegrationTestBuilder = IntegrationTestBuilder
FlowTester = FlowTester
StateChecker = StateChecker
DataConsistencyChecker = DataConsistencyChecker



