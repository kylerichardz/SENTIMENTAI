import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import logging
import time
import random

class ReviewScraper:
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        }
        self.session.headers.update(self.headers)
        self.logger = logging.getLogger(__name__)

    def _make_request(self, url: str, retry_count: int = 3) -> requests.Response:
        """Make a request with retry logic and random delays"""
        for attempt in range(retry_count):
            try:
                # Random delay between requests (1-3 seconds)
                delay = random.uniform(1, 3)
                time.sleep(delay)
                
                response = self.session.get(url)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                self.logger.warning(f"Attempt {attempt + 1}/{retry_count} failed: {str(e)}")
                if attempt == retry_count - 1:
                    raise
                # Exponential backoff with jitter
                time.sleep((2 ** attempt) + random.uniform(0, 1))
        
        raise requests.RequestException("All retry attempts failed")

    def get_trending_products(self) -> List[Dict]:
        """Returns a list of trending products from Amazon Best Sellers"""
        trending_products = []
        try:
            url = "https://www.amazon.com/Best-Sellers/zgbs"
            response = self._make_request(url)
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find product elements
            product_elements = soup.find_all('div', {'class': 'zg-grid-general-faceout'})

            for product in product_elements[:10]:
                try:
                    title_element = product.find('div', {'class': '_p13n-zg-list-grid-desktop_truncationStyles_p13n-sc-css-line-clamp-3__g3dy1'})
                    price_element = product.find('span', {'class': '_p13n-zg-list-grid-desktop_price_p13n-sc-price__3mJ9Z'})
                    link_element = product.find('a', {'class': 'a-link-normal'})
                    reviews_element = product.find('span', {'class': 'a-size-small'})

                    if title_element and link_element:
                        num_reviews = 0
                        if reviews_element:
                            try:
                                num_reviews = int(''.join(filter(str.isdigit, reviews_element.get_text())))
                            except ValueError:
                                pass

                        trending_products.append({
                            'title': title_element.get_text().strip(),
                            'price': price_element.get_text().strip() if price_element else "Price not available",
                            'url': "https://www.amazon.com" + link_element.get('href') if link_element.get('href') else None,
                            'num_reviews': num_reviews
                        })

                except Exception as e:
                    self.logger.error(f"Error processing product: {str(e)}")
                    continue

        except Exception as e:
            self.logger.error(f"Error fetching trending products: {str(e)}")
            # Return fallback products only if no products were found
            if not trending_products:
                trending_products = self._get_fallback_products()

        return trending_products

    def _get_fallback_products(self) -> List[Dict]:
        """Returns a list of fallback products when scraping fails"""
        return [
            {
                'title': 'Echo Dot (5th Gen)',
                'price': '$49.99',
                'url': 'https://www.amazon.com/dp/B09B8V1LZ3',
                'num_reviews': 1000  # Example review count
            },
            {
                'title': 'Kindle Paperwhite',
                'price': '$139.99',
                'url': 'https://www.amazon.com/dp/B08KTZ8249',
                'num_reviews': 1000  # Example review count
            }
        ]

    def search_products(self, search_term: str, max_results: int = 20) -> List[Dict]:
        """Search for products on Amazon and return their details"""
        try:
            encoded_search = requests.utils.quote(search_term)
            url = f"https://www.amazon.com/s?k={encoded_search}"
            
            response = self._make_request(url)
            soup = BeautifulSoup(response.content, 'html.parser')

            products = soup.find_all('div', {'data-component-type': 's-search-result'})
            search_results = []

            for product in products[:max_results]:
                try:
                    title_element = product.find('span', {'class': 'a-text-normal'})
                    price_element = product.find('span', {'class': 'a-offscreen'})
                    rating_element = product.find('span', {'class': 'a-icon-alt'})
                    reviews_element = product.find('span', {'class': 'a-size-base'})
                    link_element = product.find('a', {'class': 'a-link-normal s-no-outline'})

                    if title_element and link_element:
                        num_reviews = 0
                        if reviews_element:
                            try:
                                num_reviews = int(''.join(filter(str.isdigit, reviews_element.get_text())))
                            except ValueError:
                                pass

                        search_results.append({
                            'title': title_element.get_text().strip(),
                            'url': "https://www.amazon.com" + link_element.get('href'),
                            'price': price_element.get_text() if price_element else "Price not available",
                            'rating': rating_element.get_text().split(' ')[0] if rating_element else "No rating",
                            'num_reviews': num_reviews
                        })

                except Exception as e:
                    self.logger.error(f"Error processing search result: {str(e)}")
                    continue

            return search_results

        except Exception as e:
            self.logger.error(f"Error searching products: {str(e)}")
            return []

    def scrape_reviews(self, url: str, max_reviews: int = 100) -> List[Dict]:
        """
        Scrape product reviews from the given URL.
        Returns a list of dictionaries containing review data.
        """
        if not self.validate_url(url):
            raise ValueError("Unsupported website. Currently only supporting Amazon.")

        reviews = []
        try:
            response = self._make_request(url)  # Using our new _make_request method
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find the reviews section
            review_elements = soup.find_all('div', {'data-hook': 'review'})

            for review in review_elements[:max_reviews]:
                try:
                    review_data = {
                        'text': self._extract_review_text(review),
                        'rating': self._extract_rating(review),
                        'date': self._extract_date(review),
                        'title': self._extract_title(review)
                    }
                    if review_data['text']:  # Only add reviews with actual text
                        reviews.append(review_data)
                    
                    # Small delay between processing reviews
                    time.sleep(random.uniform(0.1, 0.3))

                except Exception as e:
                    self.logger.error(f"Error processing review: {str(e)}")
                    continue

        except requests.RequestException as e:
            self.logger.error(f"Error scraping reviews: {str(e)}")
            raise

        return reviews

    def validate_url(self, url: str) -> bool:
        """Validate if the URL is from a supported e-commerce site."""
        supported_domains = ['amazon.com', 'amazon.co.uk']  # Add more as needed
        try:
            return any(domain in url.lower() for domain in supported_domains)
        except (AttributeError, TypeError):
            self.logger.error(f"Invalid URL format: {url}")
            return False

    def _extract_review_text(self, review_element) -> str:
        """Extract the main review text."""
        try:
            text_element = review_element.find('span', {'data-hook': 'review-body'})
            if text_element:
                return text_element.get_text().strip()
            
            # Fallback to alternative selectors
            text_element = review_element.find('div', {'class': 'a-row review-data'})
            if text_element:
                return text_element.get_text().strip()
            
            return ""
        except Exception as e:
            self.logger.error(f"Error extracting review text: {str(e)}")
            return ""

    def _extract_rating(self, review_element) -> float:
        """Extract the rating."""
        try:
            rating_element = review_element.find('i', {'data-hook': 'review-star-rating'})
            if rating_element:
                rating_text = rating_element.get_text()
                return float(rating_text.split(' ')[0])
            
            # Fallback to alternative selector
            rating_element = review_element.find('span', {'class': 'a-icon-alt'})
            if rating_element:
                rating_text = rating_element.get_text()
                return float(rating_text.split(' ')[0])
            
            return 0.0
        except Exception as e:
            self.logger.error(f"Error extracting rating: {str(e)}")
            return 0.0

    def _extract_date(self, review_element) -> str:
        """Extract the review date."""
        try:
            date_element = review_element.find('span', {'data-hook': 'review-date'})
            if date_element:
                return date_element.get_text().strip()
            
            # Fallback to alternative selector
            date_element = review_element.find('span', {'class': 'review-date'})
            if date_element:
                return date_element.get_text().strip()
            
            return ""
        except Exception as e:
            self.logger.error(f"Error extracting date: {str(e)}")
            return ""

    def _extract_title(self, review_element) -> str:
        """Extract the review title."""
        try:
            title_element = review_element.find('a', {'data-hook': 'review-title'})
            if title_element:
                return title_element.get_text().strip()
            
            # Fallback to alternative selector
            title_element = review_element.find('span', {'class': 'review-title'})
            if title_element:
                return title_element.get_text().strip()
            
            return ""
        except Exception as e:
            self.logger.error(f"Error extracting title: {str(e)}")
            return ""