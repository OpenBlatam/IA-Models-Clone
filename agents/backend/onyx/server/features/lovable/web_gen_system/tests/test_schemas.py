import unittest
from pydantic import ValidationError
from ..schemas import WebGenRequest, WebGenResponse, AccessibilityIssue, AgentContext

class TestSchemas(unittest.TestCase):
    def test_web_gen_request_valid(self):
        req = WebGenRequest(prompt="Create a landing page")
        self.assertEqual(req.prompt, "Create a landing page")
        self.assertEqual(req.style, "modern") # Default
        self.assertTrue(req.optimize_seo) # Default

    def test_web_gen_request_invalid(self):
        with self.assertRaises(ValidationError):
            WebGenRequest(prompt=123) # Should be str

    def test_accessibility_issue(self):
        issue = AccessibilityIssue(
            issue_type="missing_alt",
            description="Missing alt text",
            element="<img src='foo.jpg'>",
            severity="critical"
        )
        self.assertEqual(issue.severity, "critical")

    def test_accessibility_issue_invalid(self):
        with self.assertRaises(ValidationError):
            AccessibilityIssue(
                issue_type="missing_alt",
                description="Missing alt text",
                element="<img src='foo.jpg'>"
                # Missing severity
            )

    def test_agent_context(self):
        ctx = AgentContext(task_id="123", shared_memory={"foo": "bar"})
        self.assertEqual(ctx.task_id, "123")
        self.assertEqual(ctx.shared_memory["foo"], "bar")

if __name__ == '__main__':
    unittest.main()
