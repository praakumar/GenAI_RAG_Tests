import requests
from bs4 import BeautifulSoup
import json
import os
from typing import List, Dict, Optional
import time
import random
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from fake_useragent import UserAgent
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@dataclass
class GmatQuestion:
    question_text: str
    options: List[str]  # For multiple choice questions
    correct_answer: str
    explanation: str
    category: str  # e.g., "Quantitative", "Verbal", "Data Insights"
    sub_category: str  # e.g., "Problem Solving", "Data Sufficiency", "Critical Reasoning"
    difficulty: str  # "Easy", "Medium", "Hard"
    source_url: str
    scraped_date: str

class GmatScraper:
    def __init__(self, use_selenium: bool = False):
        self.ua = UserAgent()
        self.questions: List[GmatQuestion] = []
        self.use_selenium = use_selenium
        self.driver = None
        
        if use_selenium:
            self._setup_selenium()
        
        self.base_urls = {
            'gmatclub': {
                'base': 'https://gmatclub.com/forum/questions/',
                'categories': {
                    'quant': 'ds-problem-solving-number-properties-etc-34/',
                    'verbal': 'verbal-gmat-preparation-34/',
                    'ir': 'integrated-reasoning-34/'
                }
            },
            'veritas': {
                'base': 'https://www.veritasprep.com/gmat/practice-questions/',
                'categories': {
                    'quant': 'math/',
                    'verbal': 'verbal/',
                    'ir': 'integrated-reasoning/'
                }
            },
            'official': {
                'base': 'https://www.mba.com/exam-prep/gmat-official-starter-kit-practice-questions',
                'categories': {
                    'sample': 'sample-questions/'
                }
            }
        }

    def _setup_selenium(self):
        """Set up Selenium WebDriver with Chrome in headless mode."""
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        self.driver = webdriver.Chrome(options=chrome_options)

    def _get_random_headers(self) -> Dict[str, str]:
        """Generate random headers for requests."""
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

    def _make_request(self, url: str, use_selenium: bool = False) -> Optional[BeautifulSoup]:
        """Make HTTP request with error handling, retries, and optional Selenium support."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if use_selenium and self.driver:
                    self.driver.get(url)
                    # Wait for the content to load
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.TAG_NAME, "body"))
                    )
                    return BeautifulSoup(self.driver.page_source, 'html.parser')
                else:
                    response = requests.get(
                        url,
                        headers=self._get_random_headers(),
                        timeout=10
                    )
                    response.raise_for_status()
                    return BeautifulSoup(response.text, 'html.parser')
            except Exception as e:
                logging.error(f"Request failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(random.uniform(2, 5))
                continue
        return None

    def _extract_text_safely(self, element, selector: str = None, class_name: str = None, 
                           attribute: str = None) -> str:
        """Enhanced safe text extraction with multiple fallback methods."""
        try:
            if not element:
                return ""

            # Try direct selector with class
            if selector and class_name:
                found = element.find(selector, class_=class_name)
                if found:
                    return found.get_text(strip=True)

            # Try finding by attribute
            if attribute:
                found = element.find(attrs={attribute: True})
                if found:
                    return found.get_text(strip=True)

            # Try direct class name
            if class_name:
                found = element.find(class_=class_name)
                if found:
                    return found.get_text(strip=True)

            # Try direct selector
            if selector:
                found = element.find(selector)
                if found:
                    return found.get_text(strip=True)

            # If all else fails, try to get text directly
            text = element.get_text(strip=True)
            return text if text else ""

        except Exception as e:
            logging.warning(f"Error extracting text: {str(e)}")
            return ""

    def scrape_gmatclub(self, category: str = 'quant', pages: int = 5) -> List[GmatQuestion]:
        """Scrape GMAT questions from GMAT Club."""
        questions = []
        logging.info(f"Scraping GMAT Club - {category}...")

        base_url = self.base_urls['gmatclub']['base']
        category_path = self.base_urls['gmatclub']['categories'].get(category, '')
        
        for page in range(1, pages + 1):
            url = f"{base_url}{category_path}page{page}"
            soup = self._make_request(url, use_selenium=True)
            
            if not soup:
                continue

            question_threads = soup.find_all('div', class_='question-thread')
            
            for thread in question_threads:
                try:
                    # Extract question details
                    question_text = self._extract_text_safely(thread, 'div', 'question-content')
                    if not question_text:
                        continue

                    options = []
                    options_container = thread.find('div', class_='answer-choices')
                    if options_container:
                        for opt in options_container.find_all(['div', 'p'], class_='choice'):
                            opt_text = opt.get_text(strip=True)
                            if opt_text:
                                options.append(opt_text)

                    # Extract other metadata
                    difficulty = self._extract_text_safely(thread, 'span', 'difficulty-label')
                    explanation = self._extract_text_safely(thread, 'div', 'explanation-content')
                    correct_answer = self._extract_text_safely(thread, 'div', 'correct-answer')

                    if question_text and options:
                        question = GmatQuestion(
                            question_text=question_text,
                            options=options,
                            correct_answer=correct_answer or "Not provided",
                            explanation=explanation or "Not provided",
                            category=category.capitalize(),
                            sub_category=self._determine_subcategory(question_text),
                            difficulty=difficulty or self._determine_difficulty(question_text),
                            source_url=url,
                            scraped_date=datetime.now().isoformat()
                        )
                        questions.append(question)
                        logging.info(f"Successfully scraped question from GMAT Club: {question_text[:50]}...")

                except Exception as e:
                    logging.error(f"Error processing GMAT Club question: {str(e)}")
                    continue

            # Add delay between pages
            time.sleep(random.uniform(3, 7))

        return questions

    def scrape_veritas(self) -> List[GmatQuestion]:
        """Scrape GMAT questions from Veritas Prep."""
        questions = []
        logging.info("Scraping Veritas Prep...")

        for category, path in self.base_urls['veritas']['categories'].items():
            url = f"{self.base_urls['veritas']['base']}{path}"
            soup = self._make_request(url, use_selenium=True)
            
            if not soup:
                continue

            question_containers = soup.find_all('div', class_=['practice-question', 'question-container'])
            
            for container in question_containers:
                try:
                    question_text = self._extract_text_safely(container, 'div', 'question-stem')
                    if not question_text:
                        continue

                    options = []
                    options_div = container.find('div', class_='answer-choices')
                    if options_div:
                        for opt in options_div.find_all(['div', 'p'], class_='choice'):
                            opt_text = opt.get_text(strip=True)
                            if opt_text:
                                options.append(opt_text)

                    correct_answer = self._extract_text_safely(container, 'div', 'correct-answer')
                    explanation = self._extract_text_safely(container, 'div', 'solution')
                    difficulty = self._extract_text_safely(container, 'span', 'difficulty')

                    if question_text and options:
                        question = GmatQuestion(
                            question_text=question_text,
                            options=options,
                            correct_answer=correct_answer or "Not provided",
                            explanation=explanation or "Not provided",
                            category=category.capitalize(),
                            sub_category=self._determine_subcategory(question_text),
                            difficulty=difficulty or self._determine_difficulty(question_text),
                            source_url=url,
                            scraped_date=datetime.now().isoformat()
                        )
                        questions.append(question)
                        logging.info(f"Successfully scraped question from Veritas: {question_text[:50]}...")

                except Exception as e:
                    logging.error(f"Error processing Veritas question: {str(e)}")
                    continue

            time.sleep(random.uniform(2, 5))

        return questions

    def _determine_category(self, question_text: str) -> str:
        """Enhanced category determination with more keywords."""
        keywords = {
            'Quantitative': [
                'calculate', 'solve', 'equation', 'number', 'percentage', 'ratio', 'math',
                'quantity', 'geometric', 'algebra', 'arithmetic', 'probability', 'average',
                'mean', 'median', 'mode', 'standard deviation', 'profit', 'loss', 'interest'
            ],
            'Verbal': [
                'passage', 'argument', 'sentence', 'grammar', 'correct', 'read', 'text',
                'paragraph', 'conclusion', 'premise', 'author', 'meaning', 'vocabulary',
                'structure', 'reasoning', 'inference', 'strengthen', 'weaken'
            ],
            'Data Insights': [
                'graph', 'table', 'data', 'chart', 'interpret', 'analysis', 'trend',
                'statistics', 'correlation', 'relationship', 'pattern', 'visualization',
                'dashboard', 'metric', 'measurement', 'indicator'
            ]
        }
        
        # Convert question text to lowercase for case-insensitive matching
        question_lower = question_text.lower()
        
        # Count matches for each category
        category_scores = {
            category: sum(1 for word in words if word.lower() in question_lower)
            for category, words in keywords.items()
        }
        
        # Return the category with the most matches, or "Uncategorized" if no matches
        max_score = max(category_scores.values())
        if max_score > 0:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        return "Uncategorized"

    def _determine_subcategory(self, question_text: str) -> str:
        """Enhanced subcategory determination."""
        subcategory_patterns = {
            'Problem Solving': ['solve', 'calculate', 'find', 'what is', 'how many'],
            'Data Sufficiency': ['sufficient', 'determine', 'could be determined', 'statement'],
            'Critical Reasoning': ['argument', 'conclusion', 'premise', 'strengthen', 'weaken'],
            'Reading Comprehension': ['passage', 'author', 'paragraph', 'according to'],
            'Sentence Correction': ['sentence', 'grammar', 'correct', 'properly'],
            'Integrated Reasoning': ['table', 'graph', 'data', 'information']
        }

        question_lower = question_text.lower()
        for subcategory, patterns in subcategory_patterns.items():
            if any(pattern in question_lower for pattern in patterns):
                return subcategory
        return "Other"

    def _determine_difficulty(self, question_text: str) -> str:
        """Enhanced difficulty determination based on various factors."""
        # Word count as a basic complexity indicator
        word_count = len(question_text.split())
        
        # Look for complexity indicators
        complexity_indicators = {
            'Hard': ['however', 'nevertheless', 'conversely', 'complex', 'challenging'],
            'Medium': ['therefore', 'consequently', 'furthermore', 'moreover'],
            'Easy': ['simple', 'straightforward', 'basic', 'direct']
        }

        question_lower = question_text.lower()
        
        # Check for explicit difficulty indicators
        for difficulty, indicators in complexity_indicators.items():
            if any(indicator in question_lower for indicator in indicators):
                return difficulty

        # Use word count as a fallback
        if word_count > 100:
            return "Hard"
        elif word_count > 50:
            return "Medium"
        return "Easy"

    def scrape_all_sources(self) -> None:
        """Scrape questions from all available sources."""
        try:
            # Scrape GMAT Club
            for category in self.base_urls['gmatclub']['categories'].keys():
                self.questions.extend(self.scrape_gmatclub(category))

            # Scrape Veritas Prep
            self.questions.extend(self.scrape_veritas())

            logging.info(f"Total questions scraped: {len(self.questions)}")

        except Exception as e:
            logging.error(f"Error during scraping: {str(e)}")
        finally:
            if self.driver:
                self.driver.quit()

    def save_questions(self, output_file: str = 'gmat_questions.json') -> None:
        """Save scraped questions to a JSON file."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump([asdict(q) for q in self.questions], f, indent=2, ensure_ascii=False)
            logging.info(f"Successfully saved {len(self.questions)} questions to {output_file}")
        except Exception as e:
            logging.error(f"Error saving questions: {str(e)}")

def main():
    # Create scraper instance with Selenium support
    scraper = GmatScraper(use_selenium=True)
    
    try:
        # Scrape all sources
        scraper.scrape_all_sources()
        
        # Save the results
        scraper.save_questions()
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
    finally:
        # Ensure the browser is closed
        if scraper.driver:
            scraper.driver.quit()

if __name__ == "__main__":
    main() 