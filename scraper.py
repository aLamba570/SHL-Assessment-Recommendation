import requests
from bs4 import BeautifulSoup
import json
import os
import time
import re
from tqdm import tqdm
import random
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
import undetected_chromedriver as uc
import sys

class SHLScraper:
    """
    Scraper for SHL product catalog pages with pagination
    """
    
    def __init__(self):
        self.assessments = []
        self.driver = None
        # Flag to indicate whether we're caught by CAPTCHA
        self.captcha_detected = False
        # Add delay between requests to avoid triggering anti-bot measures
        self.request_delay = 3  # seconds
    
    def setup_driver(self):
        """Set up undetected ChromeDriver which can bypass anti-bot measures"""
        print("Setting up undetected ChromeDriver...")
        try:
            # Try to use undetected_chromedriver first
            options = uc.ChromeOptions()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-gpu')
            options.add_argument('--window-size=1920,1080')
            options.add_argument('--disable-dev-shm-usage')
            
            driver = uc.Chrome(options=options)
            print("Successfully set up undetected ChromeDriver")
            return driver
        except Exception as e:
            print(f"Error setting up undetected ChromeDriver: {str(e)}")
            print("Falling back to regular ChromeDriver...")
            
            try:
                # Fall back to regular ChromeDriver
                chrome_options = Options()
                chrome_options.add_argument("--headless")
                chrome_options.add_argument("--no-sandbox")
                chrome_options.add_argument("--disable-dev-shm-usage")
                chrome_options.add_argument("--disable-gpu")
                chrome_options.add_argument("--window-size=1920,1080")
                
                # Add browser-like headers
                chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36")
                
                # Disable automation flags
                chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
                chrome_options.add_experimental_option('useAutomationExtension', False)
                
                service = Service(ChromeDriverManager().install())
                driver = webdriver.Chrome(service=service, options=chrome_options)
                
                # Execute CDP commands to disable automation flags
                driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
                    "source": """
                    Object.defineProperty(navigator, 'webdriver', {
                      get: () => undefined
                    })
                    """
                })
                
                return driver
            except Exception as e:
                print(f"Error setting up regular ChromeDriver: {str(e)}")
                return None
    
    def extract_test_type_codes(self, test_type_text):
        """Extract test type codes from the test type column"""
        if not test_type_text:
            return []
        
        # Map of symbols to their meanings
        type_map = {
            'A': 'Ability',
            'B': 'Behavior',
            'C': 'Cognitive',
            'E': 'English',
            'G': 'General',
            'H': 'Personality',
            'K': 'Knowledge',
            'L': 'Language',
            'M': 'Management',
            'P': 'Professional',
            'S': 'Skills',
            'T': 'Technical'
        }
        
        codes = []
        for char in test_type_text:
            if char in type_map:
                codes.append(type_map[char])
        
        return codes
    
    def is_captcha_page(self, page_content):
        """Check if we're on a CAPTCHA or bot detection page"""
        captcha_indicators = [
            "let's confirm you are human",
            "confirm you are not a robot",
            "captcha",
            "prove you're human",
            "complete the security check",
            "bot protection",
            "human verification"
        ]
        
        page_lower = page_content.lower()
        return any(indicator in page_lower for indicator in captcha_indicators)
    
    def wait_for_element(self, selector, timeout=10, by=By.CSS_SELECTOR):
        """Wait for an element to be present on the page"""
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, selector))
            )
            return element
        except TimeoutException:
            print(f"Timed out waiting for element with selector: {selector}")
            return None
    
    def wait_for_elements(self, selector, timeout=10, by=By.CSS_SELECTOR):
        """Wait for elements to be present on the page"""
        try:
            elements = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_all_elements_located((by, selector))
            )
            return elements
        except TimeoutException:
            print(f"Timed out waiting for elements with selector: {selector}")
            return []
    
    def scrape_catalog_page(self, url):
        """Scrape assessments directly from a catalog page URL"""
        if not self.driver:
            return []
        
        assessments = []
        
        try:
            print(f"Scraping catalog page: {url}")
            self.driver.get(url)
            time.sleep(self.request_delay)
            
            # Check for CAPTCHA
            if self.is_captcha_page(self.driver.page_source):
                print("CAPTCHA detected! Cannot scrape the catalog page.")
                self.captcha_detected = True
                # Save the CAPTCHA page for analysis
                with open("debug_captcha_page.html", "w", encoding="utf-8") as f:
                    f.write(self.driver.page_source)
                print("Saved CAPTCHA page to debug_captcha_page.html")
                return []
            
            # Save page for debugging (only the first page)
            if "start=0" in url or "start=12" in url:
                debug_file = f"debug_catalog_page_{url.split('start=')[1].split('&')[0]}.html"
                with open(debug_file, "w", encoding="utf-8") as f:
                    f.write(self.driver.page_source)
                print(f"Saved {debug_file} for inspection")
            
            # Find the table with assessments
            table_selectors = [
                "table.product-table", 
                "table.wp-block-table", 
                ".product-listing table",
                "table",
                ".wp-block-table"
            ]
            
            table = None
            for selector in table_selectors:
                tables = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if tables:
                    table = tables[0]
                    break
            
            if not table:
                print(f"No table found on page {url}")
                return []
            
            # Find all assessment rows in the table
            rows = table.find_elements(By.TAG_NAME, "tr")
            print(f"Found {len(rows)} potential assessment rows")
            
            # Skip header row if present
            start_index = 1 if len(rows) > 1 else 0
            
            for row in rows[start_index:]:
                try:
                    # Extract cells
                    cells = row.find_elements(By.TAG_NAME, "td")
                    
                    if len(cells) < 3:
                        continue  # Not enough data in this row
                    
                    # Extract name from first cell
                    name_cell = cells[0]
                    name = name_cell.text.strip()
                    
                    if not name:
                        continue  # Skip rows with no name
                    
                    # Extract URL if available
                    url = ""
                    try:
                        links = name_cell.find_elements(By.TAG_NAME, "a")
                        if links:
                            href = links[0].get_attribute("href")
                            if href and href.startswith("http"):
                                url = href
                    except Exception as e:
                        print(f"Error extracting URL for {name}: {str(e)}")
                    
                    # Extract remote testing status (usually 2nd column)
                    remote_testing = False
                    try:
                        if len(cells) > 1:
                            remote_cell = cells[1]
                            remote_indicators = remote_cell.find_elements(By.CSS_SELECTOR, 
                                ".green-circle, [class*='check'], .checkmark")
                            remote_testing = len(remote_indicators) > 0 or "Yes" in remote_cell.text
                    except:
                        pass
                    
                    # Extract adaptive/IRT status (usually 3rd column)
                    adaptive_irt = False
                    try:
                        if len(cells) > 2:
                            adaptive_cell = cells[2]
                            adaptive_indicators = adaptive_cell.find_elements(By.CSS_SELECTOR, 
                                ".green-circle, [class*='check'], .checkmark")
                            adaptive_irt = len(adaptive_indicators) > 0 or "Yes" in adaptive_cell.text
                    except:
                        pass
                    
                    # Extract test type codes (usually last column)
                    test_type_codes = ""
                    test_types = []
                    try:
                        if len(cells) > 3:
                            test_type_cell = cells[-1]
                            test_type_codes = test_type_cell.text.strip()
                            test_types = self.extract_test_type_codes(test_type_codes)
                    except:
                        pass
                    
                    assessment = {
                        'name': name,
                        'url': url,
                        'remote_testing': remote_testing,
                        'adaptive_irt': adaptive_irt,
                        'duration': "25 minutes",  # Default value as it's not shown in the catalog table
                        'test_type': test_types,
                        'description': f"{name} is an assessment by SHL designed to evaluate candidates.",
                        'test_type_codes': test_type_codes
                    }
                    
                    print(f"Extracted assessment: {name}")
                    assessments.append(assessment)
                    
                except Exception as e:
                    print(f"Error processing row: {str(e)}")
                    continue
            
            print(f"Successfully scraped {len(assessments)} assessments from page {url}")
            return assessments
            
        except Exception as e:
            print(f"Error scraping catalog page {url}: {str(e)}")
            return []
    
    def scrape_catalog_urls(self, catalog_urls):
        """Scrape multiple catalog pages using the provided URLs"""
        self.assessments = []
        self.driver = self.setup_driver()
        
        if not self.driver:
            print("Failed to set up WebDriver.")
            return []
        
        try:
            # Process each catalog URL
            for url in tqdm(catalog_urls, desc="Scraping catalog pages"):
                # Add random delay between requests
                time.sleep(random.uniform(2, 4))
                
                # Scrape catalog page
                page_assessments = self.scrape_catalog_page(url)
                self.assessments.extend(page_assessments)
                
                # Stop if we hit a CAPTCHA
                if self.captcha_detected:
                    print("Stopping scrape due to CAPTCHA detection.")
                    break
            
            print(f"Successfully scraped a total of {len(self.assessments)} assessments from all pages")
            
        except Exception as e:
            print(f"Error during catalog scraping: {str(e)}")
        finally:
            # Always close the driver
            if self.driver:
                try:
                    self.driver.quit()
                except:
                    pass
        
        return self.assessments
    
    def save_to_json(self, output_file="data/assessments.json"):
        """Save scraped assessments to JSON file"""
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save to JSON file
        with open(output_file, 'w') as f:
            json.dump(self.assessments, f, indent=2)
        
        print(f"Saved {len(self.assessments)} assessments to {output_file}")
        return len(self.assessments)

def main():
    # First, install undetected_chromedriver if it's not already installed
    try:
        import undetected_chromedriver
    except ImportError:
        print("Installing undetected_chromedriver...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "undetected-chromedriver"])
        print("undetected_chromedriver installed successfully")
    
    # List of catalog pagination URLs to scrape
    catalog_urls = [
        "https://www.shl.com/products/product-catalog/?start=0&type=1&type=1",
        "https://www.shl.com/products/product-catalog/?start=12&type=1&type=1",
        "https://www.shl.com/products/product-catalog/?start=24&type=1&type=1",
        "https://www.shl.com/products/product-catalog/?start=36&type=1&type=1",
        "https://www.shl.com/products/product-catalog/?start=48&type=1&type=1",
        "https://www.shl.com/products/product-catalog/?start=60&type=1&type=1",
        "https://www.shl.com/products/product-catalog/?start=72&type=1&type=1",
        "https://www.shl.com/products/product-catalog/?start=84&type=1&type=1",
        "https://www.shl.com/products/product-catalog/?start=96&type=1&type=1",
        "https://www.shl.com/products/product-catalog/?start=108&type=1&type=1",
        "https://www.shl.com/products/product-catalog/?start=120&type=1&type=1",
        "https://www.shl.com/products/product-catalog/?start=132&type=1&type=1",
        "https://www.shl.com/products/product-catalog/?start=144&type=1&type=1",
        "https://www.shl.com/products/product-catalog/?start=156&type=1&type=1",
        "https://www.shl.com/products/product-catalog/?start=168&type=1&type=1",
        "https://www.shl.com/products/product-catalog/?start=180&type=1&type=1",
        "https://www.shl.com/products/product-catalog/?start=192&type=1&type=1",
        "https://www.shl.com/products/product-catalog/?start=204&type=1&type=1",
        "https://www.shl.com/products/product-catalog/?start=216&type=1&type=1",
        "https://www.shl.com/products/product-catalog/?start=228&type=1&type=1",
        "https://www.shl.com/products/product-catalog/?start=240&type=1&type=1",
        "https://www.shl.com/products/product-catalog/?start=252&type=1&type=1",
        "https://www.shl.com/products/product-catalog/?start=264&type=1&type=1",
        "https://www.shl.com/products/product-catalog/?start=276&type=1&type=1",
        "https://www.shl.com/products/product-catalog/?start=288&type=1&type=1",
        "https://www.shl.com/products/product-catalog/?start=300&type=1&type=1",
        "https://www.shl.com/products/product-catalog/?start=312&type=1&type=1",
        "https://www.shl.com/products/product-catalog/?start=324&type=1&type=1",
        "https://www.shl.com/products/product-catalog/?start=336&type=1&type=1",
        "https://www.shl.com/products/product-catalog/?start=348&type=1&type=1",
        "https://www.shl.com/products/product-catalog/?start=360&type=1&type=1",
        "https://www.shl.com/products/product-catalog/?start=372&type=1&type=1",
        "https://www.shl.com/products/product-catalog/?start=12&type=2",
        "https://www.shl.com/products/product-catalog/?start=24&type=2&type=2",
        "https://www.shl.com/products/product-catalog/?start=36&type=2&type=2",
        "https://www.shl.com/products/product-catalog/?start=48&type=2&type=2",
        "https://www.shl.com/products/product-catalog/?start=60&type=2&type=2",
        "https://www.shl.com/products/product-catalog/?start=72&type=2&type=2",
        "https://www.shl.com/products/product-catalog/?start=84&type=2&type=2",
        "https://www.shl.com/products/product-catalog/?start=96&type=2&type=2",
        "https://www.shl.com/products/product-catalog/?start=108&type=2&type=2",
        "https://www.shl.com/products/product-catalog/?start=120&type=2&type=2",
        "https://www.shl.com/products/product-catalog/?start=132&type=2&type=2"
    ]
    
    # Use command-line arguments if provided, otherwise use the default list
    if len(sys.argv) > 1:
        catalog_urls = sys.argv[1:]
    
    # Initialize and run scraper
    scraper = SHLScraper()
    scraper.scrape_catalog_urls(catalog_urls)
    scraper.save_to_json()

if __name__ == "__main__":
    main()