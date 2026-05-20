import unittest
from ..generator import DynamicUIGenerator

class TestDynamicUIGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = DynamicUIGenerator()

    def test_generate_page_structure(self):
        html = self.generator.generate_page("test page")
        self.assertIn('<!DOCTYPE html>', html)
        self.assertIn('<title>Test page</title>', html)
        self.assertIn('Welcome to Test page', html)

    def test_generate_page_styles(self):
        html_modern = self.generator.generate_page("test", style="modern")
        self.assertIn("font-family: 'Inter'", html_modern)
        
        html_minimal = self.generator.generate_page("test", style="minimal")
        self.assertIn("font-family: 'Helvetica'", html_minimal)

if __name__ == '__main__':
    unittest.main()
