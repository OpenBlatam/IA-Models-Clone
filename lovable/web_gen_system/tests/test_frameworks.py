import unittest
from ..frameworks.nextjs import NextJSGenerator
from ..frameworks.expo import ExpoGenerator
from ..pipeline import WebGenPipeline

class TestFrameworkGenerators(unittest.TestCase):
    def setUp(self):
        self.next_gen = NextJSGenerator()
        self.expo_gen = ExpoGenerator()
        self.pipeline = WebGenPipeline()

    def test_nextjs_project_structure(self):
        structure = self.next_gen.generate_project_structure("test-app")
        self.assertIn("test-app/package.json", structure)
        self.assertIn("test-app/next.config.js", structure)
        self.assertIn("next", structure["test-app/package.json"])

    def test_nextjs_page_generation(self):
        page = self.next_gen.generate_page("About Us", "/about")
        self.assertIn("export default function Page", page)
        self.assertIn("About Us", page)

    def test_expo_project_structure(self):
        structure = self.expo_gen.generate_project_structure("test-mobile")
        self.assertIn("test-mobile/app.json", structure)
        self.assertIn("test-mobile/App.tsx", structure)
        self.assertIn("expo", structure["test-mobile/package.json"])

    def test_expo_screen_generation(self):
        screen = self.expo_gen.generate_screen("Profile", "ProfileScreen")
        self.assertIn("import { StyleSheet", screen)
        self.assertIn("Profile", screen)

    def test_pipeline_routing_nextjs(self):
        result = self.pipeline.run("Create a project", target="nextjs")
        self.assertIsInstance(result, dict)
        self.assertIn("my-next-app/package.json", result)

    def test_pipeline_routing_expo(self):
        result = self.pipeline.run("Create a mobile app", target="expo")
        self.assertIsInstance(result, dict)
        self.assertIn("my-expo-app/app.json", result)

if __name__ == '__main__':
    unittest.main()
