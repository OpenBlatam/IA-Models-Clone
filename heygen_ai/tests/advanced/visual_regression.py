"""
Visual Regression Testing Framework for HeyGen AI Testing System.
Advanced visual testing including screenshot comparison, UI element detection,
and visual diff analysis.
"""

import os
import time
import json
import base64
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import logging
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont
import cv2
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException
import threading
import concurrent.futures

@dataclass
class VisualElement:
    """Represents a visual element for testing."""
    element_id: str
    selector: str
    element_type: str  # button, input, div, img, etc.
    position: Tuple[int, int, int, int]  # x, y, width, height
    text_content: str = ""
    attributes: Dict[str, str] = field(default_factory=dict)
    screenshot: Optional[bytes] = None

@dataclass
class VisualTest:
    """Represents a visual test case."""
    test_id: str
    test_name: str
    url: str
    viewport_size: Tuple[int, int] = (1920, 1080)
    elements: List[VisualElement] = field(default_factory=list)
    full_page: bool = True
    wait_time: int = 3
    threshold: float = 0.1  # Similarity threshold (0-1)
    baseline_image: Optional[str] = None
    current_image: Optional[str] = None
    diff_image: Optional[str] = None

@dataclass
class VisualTestResult:
    """Result of a visual test."""
    test_id: str
    test_name: str
    success: bool
    similarity_score: float
    differences: List[Dict[str, Any]] = field(default_factory=list)
    execution_time: float = 0.0
    error_message: str = ""
    baseline_path: str = ""
    current_path: str = ""
    diff_path: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

class ImageProcessor:
    """Processes and compares images for visual testing."""
    
    def __init__(self):
        self.supported_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.gif']
    
    def load_image(self, image_path: str) -> Optional[Image.Image]:
        """Load an image from file."""
        try:
            return Image.open(image_path)
        except Exception as e:
            logging.error(f"Error loading image {image_path}: {e}")
            return None
    
    def save_image(self, image: Image.Image, output_path: str) -> bool:
        """Save an image to file."""
        try:
            image.save(output_path)
            return True
        except Exception as e:
            logging.error(f"Error saving image {output_path}: {e}")
            return False
    
    def resize_image(self, image: Image.Image, size: Tuple[int, int]) -> Image.Image:
        """Resize an image to specified dimensions."""
        return image.resize(size, Image.Resampling.LANCZOS)
    
    def crop_image(self, image: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
        """Crop an image to specified bounding box."""
        return image.crop(bbox)
    
    def compare_images(self, image1: Image.Image, image2: Image.Image, 
                      threshold: float = 0.1) -> Tuple[float, Image.Image, List[Dict[str, Any]]]:
        """Compare two images and return similarity score, diff image, and differences."""
        # Convert to same size if needed
        if image1.size != image2.size:
            image2 = image2.resize(image1.size, Image.Resampling.LANCZOS)
        
        # Convert to RGB if needed
        if image1.mode != 'RGB':
            image1 = image1.convert('RGB')
        if image2.mode != 'RGB':
            image2 = image2.convert('RGB')
        
        # Calculate structural similarity
        similarity_score = self._calculate_similarity(image1, image2)
        
        # Create difference image
        diff_image = self._create_diff_image(image1, image2)
        
        # Find differences
        differences = self._find_differences(image1, image2, threshold)
        
        return similarity_score, diff_image, differences
    
    def _calculate_similarity(self, img1: Image.Image, img2: Image.Image) -> float:
        """Calculate structural similarity between two images."""
        # Convert to numpy arrays
        arr1 = np.array(img1)
        arr2 = np.array(img2)
        
        # Calculate mean squared error
        mse = np.mean((arr1 - arr2) ** 2)
        
        # Calculate peak signal-to-noise ratio
        if mse == 0:
            return 1.0
        
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        
        # Convert PSNR to similarity score (0-1)
        similarity = min(1.0, psnr / 40.0)
        
        return similarity
    
    def _create_diff_image(self, img1: Image.Image, img2: Image.Image) -> Image.Image:
        """Create a visual difference image."""
        # Calculate absolute difference
        diff = ImageChops.difference(img1, img2)
        
        # Enhance differences
        diff = diff.convert('L')
        diff = diff.point(lambda x: 255 if x > 30 else 0)
        
        # Create colored diff image
        diff_colored = Image.new('RGB', diff.size, (0, 0, 0))
        diff_colored.paste(diff, mask=diff)
        
        return diff_colored
    
    def _find_differences(self, img1: Image.Image, img2: Image.Image, 
                         threshold: float) -> List[Dict[str, Any]]:
        """Find specific differences between images."""
        differences = []
        
        # Convert to numpy arrays
        arr1 = np.array(img1)
        arr2 = np.array(img2)
        
        # Calculate difference
        diff = np.abs(arr1.astype(float) - arr2.astype(float))
        
        # Find regions with significant differences
        diff_mask = np.any(diff > threshold * 255, axis=2)
        
        if np.any(diff_mask):
            # Find contours of differences
            contours, _ = cv2.findContours(
                diff_mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            for i, contour in enumerate(contours):
                if cv2.contourArea(contour) > 100:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    differences.append({
                        'id': f'diff_{i}',
                        'bbox': (x, y, w, h),
                        'area': cv2.contourArea(contour),
                        'severity': 'high' if cv2.contourArea(contour) > 1000 else 'medium'
                    })
        
        return differences
    
    def highlight_differences(self, image: Image.Image, differences: List[Dict[str, Any]]) -> Image.Image:
        """Highlight differences on an image."""
        highlighted = image.copy()
        draw = ImageDraw.Draw(highlighted)
        
        for diff in differences:
            bbox = diff['bbox']
            x, y, w, h = bbox
            
            # Draw rectangle around difference
            color = (255, 0, 0) if diff['severity'] == 'high' else (255, 165, 0)
            draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
            
            # Add label
            try:
                font = ImageFont.load_default()
                draw.text((x, y - 20), f"Diff {diff['id']}", fill=color, font=font)
            except:
                pass
        
        return highlighted

class WebDriverManager:
    """Manages WebDriver instances for visual testing."""
    
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.drivers: Dict[str, webdriver.Chrome] = {}
        self.lock = threading.Lock()
    
    def get_driver(self, session_id: str = "default") -> webdriver.Chrome:
        """Get or create a WebDriver instance."""
        with self.lock:
            if session_id not in self.drivers:
                self.drivers[session_id] = self._create_driver()
            return self.drivers[session_id]
    
    def _create_driver(self) -> webdriver.Chrome:
        """Create a new WebDriver instance."""
        options = Options()
        
        if self.headless:
            options.add_argument('--headless')
        
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-plugins')
        options.add_argument('--disable-images')  # Faster loading
        
        try:
            driver = webdriver.Chrome(options=options)
            driver.set_page_load_timeout(30)
            return driver
        except WebDriverException as e:
            logging.error(f"Error creating WebDriver: {e}")
            raise
    
    def close_driver(self, session_id: str = "default"):
        """Close a WebDriver instance."""
        with self.lock:
            if session_id in self.drivers:
                try:
                    self.drivers[session_id].quit()
                except:
                    pass
                del self.drivers[session_id]
    
    def close_all_drivers(self):
        """Close all WebDriver instances."""
        with self.lock:
            for driver in self.drivers.values():
                try:
                    driver.quit()
                except:
                    pass
            self.drivers.clear()

class VisualTestRunner:
    """Runs visual regression tests."""
    
    def __init__(self, output_dir: str = "visual_tests"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.image_processor = ImageProcessor()
        self.driver_manager = WebDriverManager()
        self.baseline_dir = self.output_dir / "baselines"
        self.current_dir = self.output_dir / "current"
        self.diff_dir = self.output_dir / "differences"
        
        # Create directories
        self.baseline_dir.mkdir(exist_ok=True)
        self.current_dir.mkdir(exist_ok=True)
        self.diff_dir.mkdir(exist_ok=True)
    
    def capture_screenshot(self, test: VisualTest, session_id: str = "default") -> Optional[str]:
        """Capture a screenshot for a visual test."""
        driver = self.driver_manager.get_driver(session_id)
        
        try:
            # Navigate to URL
            driver.get(test.url)
            
            # Wait for page to load
            time.sleep(test.wait_time)
            
            # Set viewport size
            driver.set_window_size(*test.viewport_size)
            
            # Capture screenshot
            if test.full_page:
                # Full page screenshot
                screenshot = driver.get_screenshot_as_png()
            else:
                # Viewport screenshot
                screenshot = driver.get_screenshot_as_png()
            
            # Save screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{test.test_id}_{timestamp}.png"
            current_path = self.current_dir / filename
            
            with open(current_path, 'wb') as f:
                f.write(screenshot)
            
            return str(current_path)
            
        except Exception as e:
            logging.error(f"Error capturing screenshot for {test.test_id}: {e}")
            return None
    
    def capture_element_screenshot(self, test: VisualTest, element: VisualElement, 
                                 session_id: str = "default") -> Optional[str]:
        """Capture a screenshot of a specific element."""
        driver = self.driver_manager.get_driver(session_id)
        
        try:
            # Navigate to URL
            driver.get(test.url)
            time.sleep(test.wait_time)
            
            # Find element
            element_obj = driver.find_element(By.CSS_SELECTOR, element.selector)
            
            # Scroll element into view
            driver.execute_script("arguments[0].scrollIntoView(true);", element_obj)
            time.sleep(1)
            
            # Capture element screenshot
            screenshot = element_obj.screenshot_as_png()
            
            # Save screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{test.test_id}_{element.element_id}_{timestamp}.png"
            element_path = self.current_dir / filename
            
            with open(element_path, 'wb') as f:
                f.write(screenshot)
            
            return str(element_path)
            
        except Exception as e:
            logging.error(f"Error capturing element screenshot for {element.element_id}: {e}")
            return None
    
    def run_visual_test(self, test: VisualTest, session_id: str = "default") -> VisualTestResult:
        """Run a visual regression test."""
        start_time = time.time()
        
        try:
            # Capture current screenshot
            current_path = self.capture_screenshot(test, session_id)
            if not current_path:
                return VisualTestResult(
                    test_id=test.test_id,
                    test_name=test.test_name,
                    success=False,
                    similarity_score=0.0,
                    error_message="Failed to capture screenshot",
                    execution_time=time.time() - start_time
                )
            
            # Set current image path
            test.current_image = current_path
            
            # Check if baseline exists
            baseline_path = self.baseline_dir / f"{test.test_id}.png"
            
            if not baseline_path.exists():
                # Create baseline
                baseline_path.parent.mkdir(exist_ok=True)
                current_image = self.image_processor.load_image(current_path)
                if current_image:
                    self.image_processor.save_image(current_image, str(baseline_path))
                
                return VisualTestResult(
                    test_id=test.test_id,
                    test_name=test.test_name,
                    success=True,
                    similarity_score=1.0,
                    baseline_path=str(baseline_path),
                    current_path=current_path,
                    execution_time=time.time() - start_time
                )
            
            # Compare with baseline
            baseline_image = self.image_processor.load_image(str(baseline_path))
            current_image = self.image_processor.load_image(current_path)
            
            if not baseline_image or not current_image:
                return VisualTestResult(
                    test_id=test.test_id,
                    test_name=test.test_name,
                    success=False,
                    similarity_score=0.0,
                    error_message="Failed to load images for comparison",
                    execution_time=time.time() - start_time
                )
            
            # Compare images
            similarity_score, diff_image, differences = self.image_processor.compare_images(
                baseline_image, current_image, test.threshold
            )
            
            # Save diff image
            diff_path = self.diff_dir / f"{test.test_id}_diff.png"
            self.image_processor.save_image(diff_image, str(diff_path))
            
            # Determine success
            success = similarity_score >= (1.0 - test.threshold)
            
            # Highlight differences if any
            if differences:
                highlighted = self.image_processor.highlight_differences(current_image, differences)
                highlighted_path = self.diff_dir / f"{test.test_id}_highlighted.png"
                self.image_processor.save_image(highlighted, str(highlighted_path))
            
            execution_time = time.time() - start_time
            
            return VisualTestResult(
                test_id=test.test_id,
                test_name=test.test_name,
                success=success,
                similarity_score=similarity_score,
                differences=differences,
                execution_time=execution_time,
                baseline_path=str(baseline_path),
                current_path=current_path,
                diff_path=str(diff_path)
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return VisualTestResult(
                test_id=test.test_id,
                test_name=test.test_name,
                success=False,
                similarity_score=0.0,
                error_message=str(e),
                execution_time=execution_time
            )
    
    def run_element_visual_test(self, test: VisualTest, element: VisualElement, 
                              session_id: str = "default") -> VisualTestResult:
        """Run a visual test for a specific element."""
        start_time = time.time()
        
        try:
            # Capture element screenshot
            current_path = self.capture_element_screenshot(test, element, session_id)
            if not current_path:
                return VisualTestResult(
                    test_id=f"{test.test_id}_{element.element_id}",
                    test_name=f"{test.test_name} - {element.element_id}",
                    success=False,
                    similarity_score=0.0,
                    error_message="Failed to capture element screenshot",
                    execution_time=time.time() - start_time
                )
            
            # Check if baseline exists
            baseline_path = self.baseline_dir / f"{test.test_id}_{element.element_id}.png"
            
            if not baseline_path.exists():
                # Create baseline
                current_image = self.image_processor.load_image(current_path)
                if current_image:
                    self.image_processor.save_image(current_image, str(baseline_path))
                
                return VisualTestResult(
                    test_id=f"{test.test_id}_{element.element_id}",
                    test_name=f"{test.test_name} - {element.element_id}",
                    success=True,
                    similarity_score=1.0,
                    baseline_path=str(baseline_path),
                    current_path=current_path,
                    execution_time=time.time() - start_time
                )
            
            # Compare with baseline
            baseline_image = self.image_processor.load_image(str(baseline_path))
            current_image = self.image_processor.load_image(current_path)
            
            if not baseline_image or not current_image:
                return VisualTestResult(
                    test_id=f"{test.test_id}_{element.element_id}",
                    test_name=f"{test.test_name} - {element.element_id}",
                    success=False,
                    similarity_score=0.0,
                    error_message="Failed to load images for comparison",
                    execution_time=time.time() - start_time
                )
            
            # Compare images
            similarity_score, diff_image, differences = self.image_processor.compare_images(
                baseline_image, current_image, test.threshold
            )
            
            # Save diff image
            diff_path = self.diff_dir / f"{test.test_id}_{element.element_id}_diff.png"
            self.image_processor.save_image(diff_image, str(diff_path))
            
            # Determine success
            success = similarity_score >= (1.0 - test.threshold)
            
            execution_time = time.time() - start_time
            
            return VisualTestResult(
                test_id=f"{test.test_id}_{element.element_id}",
                test_name=f"{test.test_name} - {element.element_id}",
                success=success,
                similarity_score=similarity_score,
                differences=differences,
                execution_time=execution_time,
                baseline_path=str(baseline_path),
                current_path=current_path,
                diff_path=str(diff_path)
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return VisualTestResult(
                test_id=f"{test.test_id}_{element.element_id}",
                test_name=f"{test.test_name} - {element.element_id}",
                success=False,
                similarity_score=0.0,
                error_message=str(e),
                execution_time=execution_time
            )
    
    def cleanup(self):
        """Cleanup resources."""
        self.driver_manager.close_all_drivers()

class VisualRegressionFramework:
    """Main visual regression testing framework."""
    
    def __init__(self, output_dir: str = "visual_tests"):
        self.output_dir = Path(output_dir)
        self.runner = VisualTestRunner(str(output_dir))
        self.tests: List[VisualTest] = []
        self.results: List[VisualTestResult] = []
    
    def add_test(self, test: VisualTest):
        """Add a visual test."""
        self.tests.append(test)
    
    def create_test(self, test_id: str, test_name: str, url: str, 
                   viewport_size: Tuple[int, int] = (1920, 1080),
                   threshold: float = 0.1) -> VisualTest:
        """Create a new visual test."""
        test = VisualTest(
            test_id=test_id,
            test_name=test_name,
            url=url,
            viewport_size=viewport_size,
            threshold=threshold
        )
        self.add_test(test)
        return test
    
    def run_visual_tests(self, session_id: str = "default") -> List[VisualTestResult]:
        """Run all visual tests."""
        print("🖼️  Running Visual Regression Tests")
        print("=" * 50)
        
        results = []
        
        for test in self.tests:
            print(f"Testing: {test.test_name}")
            
            # Run full page test
            result = self.runner.run_visual_test(test, session_id)
            results.append(result)
            
            status_icon = "✅" if result.success else "❌"
            print(f"  {status_icon} Full page: {result.similarity_score:.3f} similarity")
            
            # Run element tests
            for element in test.elements:
                element_result = self.runner.run_element_visual_test(test, element, session_id)
                results.append(element_result)
                
                status_icon = "✅" if element_result.success else "❌"
                print(f"  {status_icon} Element {element.element_id}: {element_result.similarity_score:.3f} similarity")
        
        self.results.extend(results)
        return results
    
    def generate_visual_report(self, output_file: str = "visual_test_report.json"):
        """Generate visual testing report."""
        if not self.results:
            return
        
        report = {
            "summary": {
                "total_tests": len(self.results),
                "successful_tests": sum(1 for r in self.results if r.success),
                "failed_tests": sum(1 for r in self.results if not r.success),
                "success_rate": (sum(1 for r in self.results if r.success) / len(self.results) * 100),
                "average_similarity": sum(r.similarity_score for r in self.results) / len(self.results)
            },
            "results": [
                {
                    "test_id": r.test_id,
                    "test_name": r.test_name,
                    "success": r.success,
                    "similarity_score": r.similarity_score,
                    "differences": r.differences,
                    "execution_time": r.execution_time,
                    "error_message": r.error_message,
                    "baseline_path": r.baseline_path,
                    "current_path": r.current_path,
                    "diff_path": r.diff_path,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in self.results
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📊 Visual test report saved to: {output_file}")
    
    def cleanup(self):
        """Cleanup resources."""
        self.runner.cleanup()

# Example usage and demo
def demo_visual_regression():
    """Demonstrate visual regression testing capabilities."""
    print("🖼️  Visual Regression Testing Framework Demo")
    print("=" * 50)
    
    # Create visual regression framework
    framework = VisualRegressionFramework()
    
    # Create sample tests
    test1 = framework.create_test(
        "homepage_test",
        "Homepage Visual Test",
        "https://example.com",
        viewport_size=(1920, 1080),
        threshold=0.1
    )
    
    # Add elements to test
    test1.elements.append(VisualElement(
        element_id="header",
        selector="header",
        element_type="header",
        position=(0, 0, 1920, 100)
    ))
    
    test1.elements.append(VisualElement(
        element_id="main_content",
        selector="main",
        element_type="main",
        position=(0, 100, 1920, 800)
    ))
    
    # Run tests
    try:
        results = framework.run_visual_tests()
        
        # Print results
        print("\n📊 Visual Test Results:")
        for result in results:
            status_icon = "✅" if result.success else "❌"
            print(f"  {status_icon} {result.test_name}: {result.similarity_score:.3f} similarity")
        
        # Generate report
        framework.generate_visual_report()
        
    finally:
        # Cleanup
        framework.cleanup()

if __name__ == "__main__":
    # Run demo
    demo_visual_regression()
