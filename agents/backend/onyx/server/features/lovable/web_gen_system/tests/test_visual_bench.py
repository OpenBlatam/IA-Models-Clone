import unittest
from ..agents import VisualCriticAgent
from ..evaluation.benchmarker import WebGenBench

class TestVisualAndBench(unittest.TestCase):
    
    def test_visual_critic_pass(self):
        agent = VisualCriticAgent()
        # Mock valid code: Responsive, styled image, no empty divs
        context = {"code_structure": {"page.tsx": "<div className='md:flex p-4'><img className='w-full' src='test.jpg' />Content</div>"}}
        output = agent.run(context)
        if output["status"] != "passed":
            print(f"DEBUG: {output.get('visual_feedback')}")
        self.assertEqual(output["status"], "passed")

    def test_visual_critic_fail(self):
        agent = VisualCriticAgent()
        # Mock invalid code (missing responsive classes, unstyled img)
        context = {"code_structure": {"page.tsx": "<div><img /></div>"}}
        output = agent.run(context)
        self.assertEqual(output["status"], "critique")
        self.assertTrue(len(output["visual_feedback"]) > 0)

    def test_benchmarker_score(self):
        bench = WebGenBench()
        code = {
            "package.json": "{}",
            "app/page.tsx": "<div>Hello World</div>"
        }
        requirements = {"core_features": ["Hello"]}
        
        metrics = bench.evaluate_project(code, requirements)
        
        # File structure: 100% (has package.json and app)
        # Content: 100% (has "Hello")
        # Visual: 100% (no penalties)
        self.assertGreater(metrics["total_score"], 90.0)

    def test_benchmarker_penalty(self):
        bench = WebGenBench()
        code = {
            "package.json": "{}",
            "app/page.tsx": "<div><img /></div>" # Missing alt, unstyled
        }
        requirements = {"core_features": ["Hello"]}
        
        metrics = bench.evaluate_project(code, requirements)
        
        # Should be penalized for missing alt text
        self.assertLess(metrics["visual_heuristic_score"], 100.0)

if __name__ == '__main__':
    unittest.main()
