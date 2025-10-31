"""
Advanced Mutation Testing Framework for HeyGen AI Testing System.
Comprehensive mutation testing including code mutation, test quality assessment,
and mutation score analysis.
"""

import ast
import astor
import subprocess
import tempfile
import shutil
import time
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import logging
import concurrent.futures
from collections import defaultdict
import difflib
import re

@dataclass
class Mutation:
    """Represents a code mutation."""
    mutation_id: str
    original_code: str
    mutated_code: str
    mutation_type: str
    line_number: int
    column_number: int
    description: str
    file_path: str
    status: str = "pending"  # pending, killed, survived, error
    test_results: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0

@dataclass
class MutationTestResult:
    """Result of mutation testing."""
    total_mutations: int
    killed_mutations: int
    survived_mutations: int
    error_mutations: int
    mutation_score: float
    execution_time: float
    test_coverage: float
    mutations: List[Mutation] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

class CodeMutator:
    """Mutates Python code to create test cases."""
    
    def __init__(self):
        self.mutation_operators = {
            'arithmetic_replacement': self._arithmetic_replacement,
            'comparison_replacement': self._comparison_replacement,
            'logical_replacement': self._logical_replacement,
            'conditional_replacement': self._conditional_replacement,
            'loop_replacement': self._loop_replacement,
            'function_call_replacement': self._function_call_replacement,
            'variable_replacement': self._variable_replacement,
            'constant_replacement': self._constant_replacement,
            'operator_replacement': self._operator_replacement,
            'statement_deletion': self._statement_deletion
        }
    
    def mutate_code(self, code: str, file_path: str) -> List[Mutation]:
        """Generate mutations for given code."""
        mutations = []
        
        try:
            # Parse the code
            tree = ast.parse(code)
            
            # Apply mutation operators
            for operator_name, operator_func in self.mutation_operators.items():
                operator_mutations = operator_func(tree, file_path)
                mutations.extend(operator_mutations)
            
        except SyntaxError as e:
            logging.error(f"Syntax error in code: {e}")
        except Exception as e:
            logging.error(f"Error mutating code: {e}")
        
        return mutations
    
    def _arithmetic_replacement(self, tree: ast.AST, file_path: str) -> List[Mutation]:
        """Replace arithmetic operators."""
        mutations = []
        
        class ArithmeticMutator(ast.NodeTransformer):
            def visit_BinOp(self, node):
                if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)):
                    # Create mutations for each arithmetic operator
                    replacements = {
                        ast.Add: [ast.Sub, ast.Mult, ast.Div],
                        ast.Sub: [ast.Add, ast.Mult, ast.Div],
                        ast.Mult: [ast.Add, ast.Sub, ast.Div],
                        ast.Div: [ast.Add, ast.Sub, ast.Mult],
                        ast.FloorDiv: [ast.Div, ast.Mod],
                        ast.Mod: [ast.FloorDiv, ast.Div],
                        ast.Pow: [ast.Mult, ast.Div]
                    }
                    
                    for replacement in replacements.get(type(node.op), []):
                        new_node = ast.BinOp(left=node.left, op=replacement(), right=node.right)
                        mutated_code = astor.to_source(new_node)
                        
                        mutation = Mutation(
                            mutation_id=f"arith_{len(mutations)}",
                            original_code=astor.to_source(node),
                            mutated_code=mutated_code,
                            mutation_type="arithmetic_replacement",
                            line_number=node.lineno,
                            column_number=node.col_offset,
                            description=f"Replace {type(node.op).__name__} with {type(replacement).__name__}",
                            file_path=file_path
                        )
                        mutations.append(mutation)
                
                return node
        
        mutator = ArithmeticMutator()
        mutator.visit(tree)
        return mutations
    
    def _comparison_replacement(self, tree: ast.AST, file_path: str) -> List[Mutation]:
        """Replace comparison operators."""
        mutations = []
        
        class ComparisonMutator(ast.NodeTransformer):
            def visit_Compare(self, node):
                for i, op in enumerate(node.ops):
                    if isinstance(op, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn)):
                        # Create mutations for each comparison operator
                        replacements = {
                            ast.Eq: [ast.NotEq, ast.Lt, ast.Gt],
                            ast.NotEq: [ast.Eq, ast.Lt, ast.Gt],
                            ast.Lt: [ast.LtE, ast.Eq, ast.Gt],
                            ast.LtE: [ast.Lt, ast.Eq, ast.Gt],
                            ast.Gt: [ast.GtE, ast.Eq, ast.Lt],
                            ast.GtE: [ast.Gt, ast.Eq, ast.Lt],
                            ast.Is: [ast.IsNot, ast.Eq],
                            ast.IsNot: [ast.Is, ast.Eq],
                            ast.In: [ast.NotIn, ast.Eq],
                            ast.NotIn: [ast.In, ast.Eq]
                        }
                        
                        for replacement in replacements.get(type(op), []):
                            new_ops = node.ops.copy()
                            new_ops[i] = replacement()
                            new_node = ast.Compare(left=node.left, ops=new_ops, comparators=node.comparators)
                            mutated_code = astor.to_source(new_node)
                            
                            mutation = Mutation(
                                mutation_id=f"comp_{len(mutations)}",
                                original_code=astor.to_source(node),
                                mutated_code=mutated_code,
                                mutation_type="comparison_replacement",
                                line_number=node.lineno,
                                column_number=node.col_offset,
                                description=f"Replace {type(op).__name__} with {type(replacement).__name__}",
                                file_path=file_path
                            )
                            mutations.append(mutation)
                
                return node
        
        mutator = ComparisonMutator()
        mutator.visit(tree)
        return mutations
    
    def _logical_replacement(self, tree: ast.AST, file_path: str) -> List[Mutation]:
        """Replace logical operators."""
        mutations = []
        
        class LogicalMutator(ast.NodeTransformer):
            def visit_BoolOp(self, node):
                if isinstance(node.op, (ast.And, ast.Or)):
                    # Replace And with Or and vice versa
                    replacement = ast.Or if isinstance(node.op, ast.And) else ast.And
                    new_node = ast.BoolOp(op=replacement(), values=node.values)
                    mutated_code = astor.to_source(new_node)
                    
                    mutation = Mutation(
                        mutation_id=f"logical_{len(mutations)}",
                        original_code=astor.to_source(node),
                        mutated_code=mutated_code,
                        mutation_type="logical_replacement",
                        line_number=node.lineno,
                        column_number=node.col_offset,
                        description=f"Replace {type(node.op).__name__} with {type(replacement).__name__}",
                        file_path=file_path
                    )
                    mutations.append(mutation)
                
                return node
        
        mutator = LogicalMutator()
        mutator.visit(tree)
        return mutations
    
    def _conditional_replacement(self, tree: ast.AST, file_path: str) -> List[Mutation]:
        """Replace conditional statements."""
        mutations = []
        
        class ConditionalMutator(ast.NodeTransformer):
            def visit_If(self, node):
                # Replace if condition with True/False
                for condition in [ast.Constant(value=True), ast.Constant(value=False)]:
                    new_node = ast.If(test=condition, body=node.body, orelse=node.orelse)
                    mutated_code = astor.to_source(new_node)
                    
                    mutation = Mutation(
                        mutation_id=f"cond_{len(mutations)}",
                        original_code=astor.to_source(node),
                        mutated_code=mutated_code,
                        mutation_type="conditional_replacement",
                        line_number=node.lineno,
                        column_number=node.col_offset,
                        description=f"Replace condition with {condition.value}",
                        file_path=file_path
                    )
                    mutations.append(mutation)
                
                return node
        
        mutator = ConditionalMutator()
        mutator.visit(tree)
        return mutations
    
    def _loop_replacement(self, tree: ast.AST, file_path: str) -> List[Mutation]:
        """Replace loop constructs."""
        mutations = []
        
        class LoopMutator(ast.NodeTransformer):
            def visit_For(self, node):
                # Replace for loop with while loop
                if node.iter and hasattr(node.iter, 'id'):  # Simple case
                    init = ast.Assign(targets=[ast.Name(id=node.target.id, ctx=ast.Store())], 
                                    value=ast.Constant(value=0))
                    condition = ast.Compare(left=ast.Name(id=node.target.id, ctx=ast.Load()),
                                          ops=[ast.Lt()],
                                          comparators=[ast.Call(func=ast.Name(id='len', ctx=ast.Load()),
                                                              args=[node.iter], keywords=[])])
                    increment = ast.AugAssign(target=ast.Name(id=node.target.id, ctx=ast.Store()),
                                            op=ast.Add(), value=ast.Constant(value=1))
                    
                    new_body = node.body + [increment]
                    new_node = ast.While(test=condition, body=new_body, orelse=node.orelse)
                    
                    # Add initialization before the loop
                    full_code = astor.to_source(init) + '\n' + astor.to_source(new_node)
                    
                    mutation = Mutation(
                        mutation_id=f"loop_{len(mutations)}",
                        original_code=astor.to_source(node),
                        mutated_code=full_code,
                        mutation_type="loop_replacement",
                        line_number=node.lineno,
                        column_number=node.col_offset,
                        description="Replace for loop with while loop",
                        file_path=file_path
                    )
                    mutations.append(mutation)
                
                return node
        
        mutator = LoopMutator()
        mutator.visit(tree)
        return mutations
    
    def _function_call_replacement(self, tree: ast.AST, file_path: str) -> List[Mutation]:
        """Replace function calls."""
        mutations = []
        
        class FunctionCallMutator(ast.NodeTransformer):
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    # Replace function name with similar name
                    original_name = node.func.id
                    similar_names = self._get_similar_names(original_name)
                    
                    for similar_name in similar_names:
                        new_func = ast.Name(id=similar_name, ctx=ast.Load())
                        new_node = ast.Call(func=new_func, args=node.args, keywords=node.keywords)
                        mutated_code = astor.to_source(new_node)
                        
                        mutation = Mutation(
                            mutation_id=f"func_{len(mutations)}",
                            original_code=astor.to_source(node),
                            mutated_code=mutated_code,
                            mutation_type="function_call_replacement",
                            line_number=node.lineno,
                            column_number=node.col_offset,
                            description=f"Replace {original_name} with {similar_name}",
                            file_path=file_path
                        )
                        mutations.append(mutation)
                
                return node
            
            def _get_similar_names(self, name: str) -> List[str]:
                """Get similar function names for mutation."""
                # Common function name variations
                variations = {
                    'len': ['length', 'size', 'count'],
                    'str': ['string', 'text'],
                    'int': ['integer', 'number'],
                    'float': ['decimal', 'real'],
                    'list': ['array', 'sequence'],
                    'dict': ['dictionary', 'map'],
                    'print': ['display', 'show'],
                    'input': ['read', 'get'],
                    'range': ['sequence', 'interval'],
                    'enumerate': ['indexed', 'numbered']
                }
                return variations.get(name, [])
        
        mutator = FunctionCallMutator()
        mutator.visit(tree)
        return mutations
    
    def _variable_replacement(self, tree: ast.AST, file_path: str) -> List[Mutation]:
        """Replace variable names."""
        mutations = []
        
        class VariableMutator(ast.NodeTransformer):
            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Load):
                    # Replace variable name with similar name
                    original_name = node.id
                    similar_name = self._get_similar_variable_name(original_name)
                    
                    if similar_name != original_name:
                        new_node = ast.Name(id=similar_name, ctx=ast.Load())
                        mutated_code = astor.to_source(new_node)
                        
                        mutation = Mutation(
                            mutation_id=f"var_{len(mutations)}",
                            original_code=astor.to_source(node),
                            mutated_code=mutated_code,
                            mutation_type="variable_replacement",
                            line_number=node.lineno,
                            column_number=node.col_offset,
                            description=f"Replace {original_name} with {similar_name}",
                            file_path=file_path
                        )
                        mutations.append(mutation)
                
                return node
            
            def _get_similar_variable_name(self, name: str) -> str:
                """Get similar variable name for mutation."""
                # Common variable name variations
                variations = {
                    'i': 'j',
                    'j': 'k',
                    'k': 'i',
                    'x': 'y',
                    'y': 'z',
                    'z': 'x',
                    'count': 'counter',
                    'counter': 'count',
                    'index': 'idx',
                    'idx': 'index',
                    'value': 'val',
                    'val': 'value',
                    'result': 'output',
                    'output': 'result',
                    'data': 'info',
                    'info': 'data'
                }
                return variations.get(name, name)
        
        mutator = VariableMutator()
        mutator.visit(tree)
        return mutations
    
    def _constant_replacement(self, tree: ast.AST, file_path: str) -> List[Mutation]:
        """Replace constants."""
        mutations = []
        
        class ConstantMutator(ast.NodeTransformer):
            def visit_Constant(self, node):
                if isinstance(node.value, (int, float)):
                    # Replace numeric constants
                    if isinstance(node.value, int):
                        new_values = [node.value + 1, node.value - 1, 0, -1, 1]
                    else:  # float
                        new_values = [node.value + 0.1, node.value - 0.1, 0.0, -1.0, 1.0]
                    
                    for new_value in new_values:
                        if new_value != node.value:
                            new_node = ast.Constant(value=new_value)
                            mutated_code = astor.to_source(new_node)
                            
                            mutation = Mutation(
                                mutation_id=f"const_{len(mutations)}",
                                original_code=astor.to_source(node),
                                mutated_code=mutated_code,
                                mutation_type="constant_replacement",
                                line_number=node.lineno,
                                column_number=node.col_offset,
                                description=f"Replace {node.value} with {new_value}",
                                file_path=file_path
                            )
                            mutations.append(mutation)
                
                return node
        
        mutator = ConstantMutator()
        mutator.visit(tree)
        return mutations
    
    def _operator_replacement(self, tree: ast.AST, file_path: str) -> List[Mutation]:
        """Replace operators."""
        mutations = []
        
        class OperatorMutator(ast.NodeTransformer):
            def visit_UnaryOp(self, node):
                if isinstance(node.op, (ast.UAdd, ast.USub, ast.Not)):
                    # Replace unary operators
                    replacements = {
                        ast.UAdd: [ast.USub, ast.Not],
                        ast.USub: [ast.UAdd, ast.Not],
                        ast.Not: [ast.UAdd, ast.USub]
                    }
                    
                    for replacement in replacements.get(type(node.op), []):
                        new_node = ast.UnaryOp(op=replacement(), operand=node.operand)
                        mutated_code = astor.to_source(new_node)
                        
                        mutation = Mutation(
                            mutation_id=f"op_{len(mutations)}",
                            original_code=astor.to_source(node),
                            mutated_code=mutated_code,
                            mutation_type="operator_replacement",
                            line_number=node.lineno,
                            column_number=node.col_offset,
                            description=f"Replace {type(node.op).__name__} with {type(replacement).__name__}",
                            file_path=file_path
                        )
                        mutations.append(mutation)
                
                return node
        
        mutator = OperatorMutator()
        mutator.visit(tree)
        return mutations
    
    def _statement_deletion(self, tree: ast.AST, file_path: str) -> List[Mutation]:
        """Delete statements."""
        mutations = []
        
        class StatementDeletionMutator(ast.NodeTransformer):
            def visit_Assign(self, node):
                # Delete assignment statement
                mutation = Mutation(
                    mutation_id=f"del_{len(mutations)}",
                    original_code=astor.to_source(node),
                    mutated_code="",
                    mutation_type="statement_deletion",
                    line_number=node.lineno,
                    column_number=node.col_offset,
                    description="Delete assignment statement",
                    file_path=file_path
                )
                mutations.append(mutation)
                
                return node
            
            def visit_Return(self, node):
                # Delete return statement
                mutation = Mutation(
                    mutation_id=f"del_{len(mutations)}",
                    original_code=astor.to_source(node),
                    mutated_code="",
                    mutation_type="statement_deletion",
                    line_number=node.lineno,
                    column_number=node.col_offset,
                    description="Delete return statement",
                    file_path=file_path
                )
                mutations.append(mutation)
                
                return node
        
        mutator = StatementDeletionMutator()
        mutator.visit(tree)
        return mutations

class MutationTestRunner:
    """Runs mutation tests and evaluates results."""
    
    def __init__(self, test_directory: str = "tests"):
        self.test_directory = Path(test_directory)
        self.mutator = CodeMutator()
        self.temp_dir = None
    
    def run_mutation_test(self, source_file: str, test_file: str) -> MutationTestResult:
        """Run mutation testing on a source file."""
        print(f"🧬 Running Mutation Testing")
        print(f"   Source: {source_file}")
        print(f"   Tests: {test_file}")
        print("=" * 50)
        
        start_time = time.time()
        
        # Read source code
        with open(source_file, 'r') as f:
            source_code = f.read()
        
        # Generate mutations
        mutations = self.mutator.mutate_code(source_code, source_file)
        print(f"📊 Generated {len(mutations)} mutations")
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        try:
            # Run tests on each mutation
            results = self._run_mutations(mutations, source_file, test_file)
            
            # Calculate mutation score
            killed = sum(1 for m in mutations if m.status == "killed")
            survived = sum(1 for m in mutations if m.status == "survived")
            errors = sum(1 for m in mutations if m.status == "error")
            
            mutation_score = (killed / len(mutations) * 100) if mutations else 0
            
            execution_time = time.time() - start_time
            
            result = MutationTestResult(
                total_mutations=len(mutations),
                killed_mutations=killed,
                survived_mutations=survived,
                error_mutations=errors,
                mutation_score=mutation_score,
                execution_time=execution_time,
                test_coverage=0.0,  # Would need coverage tool
                mutations=mutations
            )
            
            # Print results
            self._print_mutation_results(result)
            
            return result
            
        finally:
            # Cleanup
            if self.temp_dir and Path(self.temp_dir).exists():
                shutil.rmtree(self.temp_dir)
    
    def _run_mutations(self, mutations: List[Mutation], source_file: str, test_file: str) -> List[Mutation]:
        """Run tests on each mutation."""
        results = []
        
        for i, mutation in enumerate(mutations):
            print(f"🧪 Testing mutation {i+1}/{len(mutations)}: {mutation.mutation_type}")
            
            try:
                # Create mutated file
                mutated_file = self._create_mutated_file(mutation, source_file)
                
                # Run tests
                test_result = self._run_tests(mutated_file, test_file)
                
                # Determine mutation status
                if test_result['returncode'] != 0:
                    mutation.status = "killed"
                else:
                    mutation.status = "survived"
                
                mutation.test_results = test_result
                mutation.execution_time = test_result.get('execution_time', 0)
                
            except Exception as e:
                mutation.status = "error"
                mutation.test_results = {'error': str(e)}
                logging.error(f"Error testing mutation {mutation.mutation_id}: {e}")
            
            results.append(mutation)
        
        return results
    
    def _create_mutated_file(self, mutation: Mutation, source_file: str) -> str:
        """Create a file with the mutated code."""
        # Read original file
        with open(source_file, 'r') as f:
            lines = f.readlines()
        
        # Apply mutation (simplified - in practice, you'd need more sophisticated replacement)
        mutated_lines = lines.copy()
        
        # This is a simplified approach - in practice, you'd need to properly replace
        # the specific line and column with the mutated code
        if mutation.line_number <= len(mutated_lines):
            # Simple line replacement (not perfect but works for demo)
            mutated_lines[mutation.line_number - 1] = mutation.mutated_code + '\n'
        
        # Write mutated file
        mutated_file = Path(self.temp_dir) / f"mutated_{mutation.mutation_id}.py"
        with open(mutated_file, 'w') as f:
            f.writelines(mutated_lines)
        
        return str(mutated_file)
    
    def _run_tests(self, source_file: str, test_file: str) -> Dict[str, Any]:
        """Run tests on the mutated file."""
        start_time = time.time()
        
        try:
            # Run pytest on the test file
            result = subprocess.run([
                'python', '-m', 'pytest', test_file, '-v', '--tb=short'
            ], capture_output=True, text=True, timeout=30)
            
            execution_time = time.time() - start_time
            
            return {
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'execution_time': execution_time
            }
            
        except subprocess.TimeoutExpired:
            return {
                'returncode': -1,
                'stdout': '',
                'stderr': 'Test execution timed out',
                'execution_time': time.time() - start_time
            }
        except Exception as e:
            return {
                'returncode': -1,
                'stdout': '',
                'stderr': str(e),
                'execution_time': time.time() - start_time
            }
    
    def _print_mutation_results(self, result: MutationTestResult):
        """Print mutation testing results."""
        print("\n" + "=" * 60)
        print("🧬 MUTATION TESTING RESULTS")
        print("=" * 60)
        
        print(f"📊 Summary:")
        print(f"   Total Mutations: {result.total_mutations}")
        print(f"   Killed: {result.killed_mutations}")
        print(f"   Survived: {result.survived_mutations}")
        print(f"   Errors: {result.error_mutations}")
        print(f"   Mutation Score: {result.mutation_score:.1f}%")
        print(f"   Execution Time: {result.execution_time:.2f}s")
        
        # Mutation score interpretation
        if result.mutation_score >= 90:
            print(f"\n🎉 Test Quality: EXCELLENT")
        elif result.mutation_score >= 70:
            print(f"\n👍 Test Quality: GOOD")
        elif result.mutation_score >= 50:
            print(f"\n⚠️  Test Quality: FAIR")
        else:
            print(f"\n❌ Test Quality: POOR")
        
        # Show survived mutations
        survived = [m for m in result.mutations if m.status == "survived"]
        if survived:
            print(f"\n🔍 Survived Mutations ({len(survived)}):")
            for mutation in survived[:5]:  # Show first 5
                print(f"   - {mutation.mutation_type}: {mutation.description}")
            if len(survived) > 5:
                print(f"   ... and {len(survived) - 5} more")
        
        print("=" * 60)

class MutationTestingFramework:
    """Main mutation testing framework."""
    
    def __init__(self, test_directory: str = "tests"):
        self.test_directory = Path(test_directory)
        self.runner = MutationTestRunner(test_directory)
        self.results: List[MutationTestResult] = []
    
    def run_mutation_test(self, source_file: str, test_file: str) -> MutationTestResult:
        """Run mutation testing."""
        result = self.runner.run_mutation_test(source_file, test_file)
        self.results.append(result)
        return result
    
    def run_mutation_suite(self, source_files: List[str], test_files: List[str]) -> List[MutationTestResult]:
        """Run mutation testing on multiple files."""
        print("🧬 Running Mutation Testing Suite")
        print("=" * 50)
        
        results = []
        
        for source_file, test_file in zip(source_files, test_files):
            if Path(source_file).exists() and Path(test_file).exists():
                result = self.run_mutation_test(source_file, test_file)
                results.append(result)
            else:
                print(f"⚠️  Skipping {source_file} or {test_file} - file not found")
        
        # Generate suite report
        self._generate_suite_report(results)
        
        return results
    
    def _generate_suite_report(self, results: List[MutationTestResult]):
        """Generate mutation testing suite report."""
        if not results:
            return
        
        print("\n" + "=" * 60)
        print("📊 MUTATION TESTING SUITE REPORT")
        print("=" * 60)
        
        total_mutations = sum(r.total_mutations for r in results)
        total_killed = sum(r.killed_mutations for r in results)
        total_survived = sum(r.survived_mutations for r in results)
        total_errors = sum(r.error_mutations for r in results)
        
        overall_score = (total_killed / total_mutations * 100) if total_mutations > 0 else 0
        
        print(f"📈 Overall Summary:")
        print(f"   Files Tested: {len(results)}")
        print(f"   Total Mutations: {total_mutations}")
        print(f"   Total Killed: {total_killed}")
        print(f"   Total Survived: {total_survived}")
        print(f"   Total Errors: {total_errors}")
        print(f"   Overall Mutation Score: {overall_score:.1f}%")
        
        print(f"\n📋 File-by-File Results:")
        for i, result in enumerate(results, 1):
            status_icon = "✅" if result.mutation_score >= 70 else "⚠️" if result.mutation_score >= 50 else "❌"
            print(f"   {status_icon} File {i}: {result.mutation_score:.1f}% score")
        
        # Overall assessment
        if overall_score >= 90:
            print(f"\n🎉 Overall Test Quality: EXCELLENT")
        elif overall_score >= 70:
            print(f"\n👍 Overall Test Quality: GOOD")
        elif overall_score >= 50:
            print(f"\n⚠️  Overall Test Quality: FAIR")
        else:
            print(f"\n❌ Overall Test Quality: POOR")
        
        print("=" * 60)
    
    def generate_mutation_report(self, output_file: str = "mutation_report.json"):
        """Generate detailed mutation testing report."""
        if not self.results:
            return
        
        report = {
            "summary": {
                "total_files": len(self.results),
                "total_mutations": sum(r.total_mutations for r in self.results),
                "total_killed": sum(r.killed_mutations for r in self.results),
                "total_survived": sum(r.survived_mutations for r in self.results),
                "total_errors": sum(r.error_mutations for r in self.results),
                "overall_mutation_score": sum(r.mutation_score for r in self.results) / len(self.results)
            },
            "results": [
                {
                    "total_mutations": r.total_mutations,
                    "killed_mutations": r.killed_mutations,
                    "survived_mutations": r.survived_mutations,
                    "error_mutations": r.error_mutations,
                    "mutation_score": r.mutation_score,
                    "execution_time": r.execution_time,
                    "test_coverage": r.test_coverage,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in self.results
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📊 Mutation report saved to: {output_file}")

# Example usage and demo
def demo_mutation_testing():
    """Demonstrate mutation testing capabilities."""
    print("🧬 Mutation Testing Framework Demo")
    print("=" * 40)
    
    # Create mutation testing framework
    framework = MutationTestingFramework()
    
    # Example source code for testing
    source_code = '''
def add_numbers(a, b):
    """Add two numbers."""
    return a + b

def multiply_numbers(a, b):
    """Multiply two numbers."""
    return a * b

def is_even(number):
    """Check if number is even."""
    return number % 2 == 0
'''
    
    # Example test code
    test_code = '''
import pytest
from source import add_numbers, multiply_numbers, is_even

def test_add_numbers():
    assert add_numbers(2, 3) == 5
    assert add_numbers(0, 0) == 0
    assert add_numbers(-1, 1) == 0

def test_multiply_numbers():
    assert multiply_numbers(2, 3) == 6
    assert multiply_numbers(0, 5) == 0
    assert multiply_numbers(-2, 3) == -6

def test_is_even():
    assert is_even(2) == True
    assert is_even(3) == False
    assert is_even(0) == True
'''
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(source_code)
        source_file = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_code)
        test_file = f.name
    
    try:
        # Run mutation testing
        result = framework.run_mutation_test(source_file, test_file)
        
        # Generate report
        framework.generate_mutation_report()
        
    finally:
        # Cleanup
        Path(source_file).unlink()
        Path(test_file).unlink()

if __name__ == "__main__":
    # Run demo
    demo_mutation_testing()
