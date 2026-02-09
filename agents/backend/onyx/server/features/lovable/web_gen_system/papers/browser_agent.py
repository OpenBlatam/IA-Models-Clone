from typing import List, Dict, Any
from ..agents.base import BaseAgent

class PlaywrightTestGenerator(BaseAgent):
    """
    Agent responsible for generating Playwright tests for the web application.
    Inspired by BrowserAgent and WebArena papers.
    """

    def __init__(self, name: str = "TestGenerator"):
        super().__init__(name, "QA Automation Engineer")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates Playwright test scripts based on the repository context and requirements.
        """
        try:
            self.log("Generating Playwright tests...")
            
            repo_context = context.get("repository_context")
            if not repo_context:
                self.log("No repository context found. Skipping test generation.", level="warning")
                return {"status": "skipped"}

            # Heuristic: Generate a basic test for the main page
            test_content = self._generate_basic_test()
            
            # Add the test file to the repository context
            repo_context.add_file("tests/e2e.spec.ts", test_content)
            
            self.log("Generated 'tests/e2e.spec.ts'.")
            return {"status": "success", "generated_files": ["tests/e2e.spec.ts"]}
            
        except Exception as e:
            self.log(f"Error in PlaywrightTestGenerator: {str(e)}", level="error")
            return {"status": "error", "error": str(e)}

    def _generate_basic_test(self) -> str:
        """
        Generates a basic Playwright test template.
        """
        return """
import { test, expect } from '@playwright/test';

test('homepage has title and main content', async ({ page }) => {
  await page.goto('http://localhost:3000');

  // Expect a title "to contain" a substring.
  await expect(page).toHaveTitle(/App/);

  // Expect the main heading to be visible
  await expect(page.locator('h1')).toBeVisible();
  
  // Check for common accessibility issues (basic check)
  const images = await page.locator('img').all();
  for (const img of images) {
    await expect(img).toHaveAttribute('alt');
  }
});
"""
