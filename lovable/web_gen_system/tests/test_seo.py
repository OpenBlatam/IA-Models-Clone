import unittest
from bs4 import BeautifulSoup
from ..seo import SEOOptimizer

class TestSEOOptimizer(unittest.TestCase):
    def setUp(self):
        self.optimizer = SEOOptimizer()

    def test_optimize_meta_tags_injects_all(self):
        html = '<html><head></head><body></body></html>'
        optimized = self.optimizer.optimize_meta_tags(html, "Test Title", "Test Desc", ["key1", "key2"])
        
        soup = BeautifulSoup(optimized, 'html.parser')
        self.assertEqual(soup.title.string, "Test Title")
        
        desc = soup.find('meta', attrs={'name': 'description'})
        self.assertIsNotNone(desc)
        self.assertEqual(desc['content'], "Test Desc")
        
        keywords = soup.find('meta', attrs={'name': 'keywords'})
        self.assertIsNotNone(keywords)
        self.assertEqual(keywords['content'], "key1, key2")

    def test_optimize_meta_tags_updates_existing(self):
        html = '<html><head><title>Old</title></head><body></body></html>'
        optimized = self.optimizer.optimize_meta_tags(html, "New Title", "Desc", [])
        
        soup = BeautifulSoup(optimized, 'html.parser')
        self.assertEqual(soup.title.string, "New Title")

    def test_analyze_content_relevance(self):
        html = '<p>This is a test content with test keyword.</p>'
        score = self.optimizer.analyze_content_relevance(html, "test")
        self.assertTrue(score > 0)

if __name__ == '__main__':
    unittest.main()
