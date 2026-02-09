import unittest
from agents.backend.onyx.server.features.lovable.web_gen_system.accessibility import WebAccessibilityEnhancer

class TestWebAccessibilityEnhancer(unittest.TestCase):
    def setUp(self):
        self.enhancer = WebAccessibilityEnhancer()

    def test_analyze_missing_alt(self):
        html = '<div><img src="test.jpg"></div>'
        issues = self.enhancer.analyze_html(html)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].issue_type, "missing_alt")

    def test_analyze_empty_button(self):
        html = '<button></button>'
        issues = self.enhancer.analyze_html(html)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].issue_type, "empty_button")

    def test_fix_accessibility_adds_lang(self):
        html = '<html><body></body></html>'
        fixed = self.enhancer.fix_accessibility(html)
        self.assertIn('lang="en"', fixed)

    def test_fix_accessibility_adds_alt_placeholder(self):
        html = '<img src="test.jpg">'
        fixed = self.enhancer.fix_accessibility(html)
        self.assertIn('alt="Image description needed"', fixed)

if __name__ == '__main__':
    unittest.main()
