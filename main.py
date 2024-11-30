import sys
print(f"Python version: {sys.version}")

import logging
import threading
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                            QTextEdit, QProgressBar, QListWidget, QSplitter)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from src.scraper.review_scraper import ReviewScraper
from src.preprocessing.data_cleaner import DataCleaner
import time
import traceback
import random
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import itertools
from textblob import TextBlob
import numpy as np

print("All imports successful")

class TrendingProductsWorker(QThread):
    products_ready = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, scraper):
        super().__init__()
        self.scraper = scraper

    def run(self):
        try:
            # Add validation for scraper
            if not hasattr(self, 'scraper') or self.scraper is None:
                self.error.emit("Scraper not properly initialized")
                self.products_ready.emit([])
                return
                
            max_retries = 3
            retry_delay = 2  # seconds
            
            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        delay = random.uniform(2, 5)
                        self.error.emit(f"Retry attempt {attempt + 1}/{max_retries} (waiting {delay:.1f}s)...")
                        time.sleep(delay)
                    
                    trending_products = self.scraper.get_trending_products()
                    # Filter products with reviews
                    if trending_products:
                        products_with_reviews = [
                            product for product in trending_products 
                            if product.get('num_reviews', 0) > 0
                        ]
                        if products_with_reviews:
                            self.products_ready.emit(products_with_reviews)
                            return
                        else:
                            self.error.emit("No products with reviews found")
                            self.products_ready.emit([])
                            return
                    else:
                        self.error.emit("No trending products found")
                        self.products_ready.emit([])
                        return
                    
                except Exception as e:
                    error_msg = f"Failed to fetch trending products: {str(e)}"
                    self.error.emit(error_msg)
                    logging.error(error_msg)  # Add logging
                    
                    if attempt == max_retries - 1:  # Last attempt
                        self.products_ready.emit([])
                        return
                    # If not the last attempt, continue to next retry
        except Exception as e:  # Add this outer exception handler
            self.error.emit(f"Unexpected error: {str(e)}")
            self.products_ready.emit([])

class AnalysisWorker(QThread):
    progress = pyqtSignal(float)
    status = pyqtSignal(str)
    log = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, url, scraper, cleaner):
        super().__init__()
        self.url = url
        self.scraper = scraper
        self.cleaner = cleaner

    def run(self):
        try:
            # Validate URL
            if not self.url or not self.url.strip():
                self.error.emit("Invalid URL provided")
                return
                
            # Validate dependencies
            if not hasattr(self, 'scraper') or self.scraper is None:
                self.error.emit("Scraper not properly initialized")
                return
            if not hasattr(self, 'cleaner') or self.cleaner is None:
                self.error.emit("Data cleaner not properly initialized")
                return

            # Add memory monitoring
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            self.log.emit(f"Initial memory usage: {initial_memory:.2f} MB")

            # Add more detailed logging for review scraping
            self.log.emit("Starting review scraping...")
            self.progress.emit(0.2)
            
            try:
                # Limit the number of reviews to prevent memory issues
                reviews_data = self.scraper.scrape_reviews(self.url, max_reviews=50)
                self.log.emit(f"Scraped reviews successfully: {len(reviews_data) if reviews_data else 0} reviews")
            except Exception as e:
                self.error.emit(f"Failed to scrape reviews: {str(e)}\nTraceback: {traceback.format_exc()}")
                return

            # Validate reviews structure before processing
            if not reviews_data:
                self.error.emit("No reviews found")
                return

            # Convert to DataFrame with error handling
            try:
                if not isinstance(reviews_data, pd.DataFrame):
                    reviews_df = pd.DataFrame(reviews_data)
                else:
                    reviews_df = reviews_data

                self.log.emit(f"Reviews loaded. Shape: {reviews_df.shape}")
                
            except Exception as e:
                self.error.emit(f"Failed to create DataFrame: {str(e)}")
                return

            self.progress.emit(0.4)
            self.log.emit("Cleaning and analyzing reviews...")
            
            try:
                # Clean the data
                cleaned_data = self.cleaner.clean_data(reviews_df)
                
                # Add sentiment analysis with error handling
                text_column = next((col for col in cleaned_data.columns 
                                  if col.lower() in ['text', 'content', 'review', 'review_text']), None)
                
                if text_column is None:
                    raise ValueError(f"No review text column found. Available columns: {list(cleaned_data.columns)}")
                
                # Add batch processing and better error handling for sentiment analysis
                def safe_sentiment(text):
                    try:
                        if pd.isna(text) or not isinstance(text, str):
                            return 0.0
                        return TextBlob(str(text)[:5000]).sentiment.polarity  # Limit text length
                    except Exception as e:
                        logging.error(f"Error in sentiment analysis: {str(e)}")
                        return 0.0

                # Process sentiment in smaller batches
                batch_size = 100
                sentiments = []
                for i in range(0, len(cleaned_data), batch_size):
                    batch = cleaned_data[text_column].iloc[i:i+batch_size]
                    batch_sentiments = [safe_sentiment(text) for text in batch]
                    sentiments.extend(batch_sentiments)
                    
                cleaned_data['sentiment_score'] = sentiments
                
                self.progress.emit(0.7)
                
                # Calculate summary statistics
                stats = {
                    'total_reviews': len(cleaned_data),
                    'avg_rating': cleaned_data['rating'].mean(),
                    'avg_sentiment': cleaned_data['sentiment_score'].mean(),
                    'positive_reviews': len(cleaned_data[cleaned_data['sentiment_score'] > 0]),
                    'negative_reviews': len(cleaned_data[cleaned_data['sentiment_score'] < 0]),
                    'neutral_reviews': len(cleaned_data[cleaned_data['sentiment_score'] == 0]),
                    'rating_distribution': cleaned_data['rating'].value_counts().sort_index().to_dict()
                }
                
                # Calculate product recommendation metrics
                recommendation_score = self.calculate_recommendation_score(cleaned_data, stats)
                
                # Log results
                self.log.emit("\n=== Analysis Results ===")
                self.log.emit(f"Total Reviews Analyzed: {stats['total_reviews']}")
                self.log.emit(f"Average Rating: {stats['avg_rating']:.2f}/5")
                self.log.emit(f"Average Sentiment: {stats['avg_sentiment']:.2f} (-1 to 1)")
                self.log.emit(f"Positive Reviews: {stats['positive_reviews']} ({stats['positive_reviews']/stats['total_reviews']*100:.1f}%)")
                self.log.emit(f"Negative Reviews: {stats['negative_reviews']} ({stats['negative_reviews']/stats['total_reviews']*100:.1f}%)")
                self.log.emit(f"Neutral Reviews: {stats['neutral_reviews']} ({stats['neutral_reviews']/stats['total_reviews']*100:.1f}%)")
                
                self.log.emit("\nRating Distribution:")
                for rating, count in stats['rating_distribution'].items():
                    self.log.emit(f"{rating} stars: {count} reviews ({count/stats['total_reviews']*100:.1f}%)")
                
                # Add recommendation summary
                self.log.emit("\n=== Product Recommendation ===")
                self.log.emit(recommendation_score['summary'])
                self.log.emit(recommendation_score['details'])
                
                self.progress.emit(1.0)
                self.log.emit("\nAnalysis complete!")
                
            except Exception as e:
                self.error.emit(f"Error during analysis: {str(e)}")
                logging.error("Analysis error", exc_info=True)
                
        except Exception as e:
            self.error.emit(f"Analysis failed: {str(e)}")
            logging.error(f"Analysis error: {str(e)}", exc_info=True)
        finally:
            self.finished.emit()

    def calculate_recommendation_score(self, cleaned_data, stats):
        """
        Calculate a recommendation score based on multiple factors:
        - Average rating
        - Sentiment analysis
        - Review count
        - Ratio of positive to negative reviews
        """
        try:
            # Initialize scoring factors
            rating_score = stats['avg_rating'] / 5.0  # Normalize to 0-1
            sentiment_score = (stats['avg_sentiment'] + 1) / 2  # Convert from -1,1 to 0-1
            
            # Calculate review volume score (logarithmic scale)
            import math
            max_reviews = 1000  # Consider this as maximum expected reviews
            review_volume_score = min(math.log(stats['total_reviews'] + 1) / math.log(max_reviews + 1), 1)
            
            # Calculate positive/negative ratio
            if stats['negative_reviews'] == 0:
                pos_neg_ratio = 1.0
            else:
                pos_neg_ratio = stats['positive_reviews'] / (stats['negative_reviews'] + 1)
                pos_neg_ratio = min(pos_neg_ratio / 3, 1)  # Normalize to 0-1, consider 3:1 ratio as optimal
            
            # Calculate final score (weighted average)
            weights = {
                'rating': 0.35,
                'sentiment': 0.25,
                'volume': 0.20,
                'ratio': 0.20
            }
            
            final_score = (
                rating_score * weights['rating'] +
                sentiment_score * weights['sentiment'] +
                review_volume_score * weights['volume'] +
                pos_neg_ratio * weights['ratio']
            ) * 100  # Convert to percentage
            
            # Generate recommendation
            if final_score >= 80:
                recommendation = "HIGHLY RECOMMENDED"
                color = "green"
            elif final_score >= 60:
                recommendation = "RECOMMENDED"
                color = "blue"
            elif final_score >= 40:
                recommendation = "MIXED REVIEWS"
                color = "orange"
            else:
                recommendation = "NOT RECOMMENDED"
                color = "red"
            
            # Create detailed summary
            summary = f"\n{'='*20} {recommendation} {'='*20}"
            details = f"""
Recommendation Score: {final_score:.1f}%

Factors Considered:
• Rating Score: {rating_score*100:.1f}% (Weight: {weights['rating']*100}%)
• Sentiment Score: {sentiment_score*100:.1f}% (Weight: {weights['sentiment']*100}%)
• Review Volume Score: {review_volume_score*100:.1f}% (Weight: {weights['volume']*100}%)
• Positive/Negative Ratio Score: {pos_neg_ratio*100:.1f}% (Weight: {weights['ratio']*100}%)

Key Insights:
• {stats['total_reviews']} total reviews analyzed
• {stats['avg_rating']:.1f}/5 average rating
• {stats['positive_reviews']/stats['total_reviews']*100:.1f}% positive reviews
• {stats['negative_reviews']/stats['total_reviews']*100:.1f}% negative reviews

Recommendation Details:
This product is {recommendation.lower()} based on a comprehensive analysis of 
customer reviews, sentiment, and rating patterns. The score indicates 
{'excellent' if final_score >= 80 else 'good' if final_score >= 60 else 'moderate' if final_score >= 40 else 'poor'} 
overall customer satisfaction.
"""
            
            return {
                'score': final_score,
                'recommendation': recommendation,
                'color': color,
                'summary': summary,
                'details': details
            }
            
        except Exception as e:
            logging.error(f"Error calculating recommendation score: {str(e)}")
            return {
                'score': 0,
                'recommendation': "ANALYSIS ERROR",
                'color': "gray",
                'summary': "\n=== Unable to Calculate Recommendation ===",
                'details': f"Error: {str(e)}"
            }

class SearchProductsWorker(QThread):
    products_ready = pyqtSignal(list)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, scraper, search_term):
        super().__init__()
        self.scraper = scraper
        self.search_term = search_term
        # Add logging
        self.logger = logging.getLogger(__name__)

    def run(self):
        try:
            # Add debug logging
            self.logger.debug(f"Starting search for: {self.search_term}")
            
            # Validate search term
            if not self.search_term or not self.search_term.strip():
                self.error.emit("Empty search term provided")
                self.products_ready.emit([])
                return

            # Validate scraper
            if not hasattr(self, 'scraper') or self.scraper is None:
                self.error.emit("Scraper not properly initialized")
                self.products_ready.emit([])
                return
                
            max_retries = 3
            retry_delay = 2  # seconds
            
            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        delay = random.uniform(2, 5)  # Add randomization to retry delay
                        self.error.emit(f"Retry attempt {attempt + 1}/{max_retries} (waiting {delay:.1f}s)...")
                        time.sleep(delay)
                    
                    self.logger.debug("Calling search_products method")
                    search_results = self.scraper.search_products(self.search_term)
                    self.logger.debug(f"Received {len(search_results) if search_results else 0} results")
                    
                    # Filter products with reviews
                    if search_results:
                        products_with_reviews = [
                            product for product in search_results 
                            if product.get('num_reviews', 0) > 0
                        ]
                        if products_with_reviews:
                            self.logger.debug(f"Found {len(products_with_reviews)} products with reviews")
                            self.products_ready.emit(products_with_reviews)
                            return
                        else:
                            self.error.emit("No products with reviews found")
                            self.products_ready.emit([])
                            return
                    else:
                        self.error.emit("No products found")
                        self.products_ready.emit([])
                        return
                    
                except Exception as e:
                    self.logger.error(f"Search attempt {attempt + 1} failed: {str(e)}", exc_info=True)
                    if attempt == max_retries - 1:  # Last attempt
                        self.error.emit(f"Error searching products: {str(e)}")
                        self.products_ready.emit([])
                        return

        except Exception as e:
            self.logger.error(f"Search worker failed: {str(e)}", exc_info=True)
            self.error.emit(f"Error searching products: {str(e)}")
            self.products_ready.emit([])
        finally:
            self.finished.emit()  # Make sure we always emit finished signal

class LoadingSpinner:
    def __init__(self, message="Analyzing"):
        self.spinner = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
        self.running = False
        self.message = message
        self.thread = None

    def spin(self):
        while self.running:
            sys.stdout.write(f"\r{next(self.spinner)} {self.message}...")
            sys.stdout.flush()
            time.sleep(0.1)
            sys.stdout.write('\b' * (len(self.message) + 5))

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.spin)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        sys.stdout.write('\r' + ' ' * (len(self.message) + 5) + '\r')
        sys.stdout.flush()

class SentimentAnalyzerGUI(QMainWindow):
    def __init__(self):
        try:
            super().__init__()
            self.setWindowTitle("Amazon Review Sentiment Analyzer")
            self.setMinimumSize(1200, 800)
            
            # Set matplotlib backend at the very beginning
            matplotlib.use('Agg')
            
            # Initialize components
            self.scraper = ReviewScraper()
            self.cleaner = DataCleaner()
            
            # Create main widget and layout
            main_widget = QWidget()
            self.setCentralWidget(main_widget)
            layout = QVBoxLayout(main_widget)
            
            # Title
            title = QLabel("Amazon Review Analyzer")
            title.setStyleSheet("font-size: 24px; font-weight: bold;")
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(title)

            # Create splitter for trending products and analysis
            splitter = QSplitter(Qt.Orientation.Horizontal)
            layout.addWidget(splitter)

            # Left panel for trending products
            trending_widget = QWidget()
            trending_layout = QVBoxLayout(trending_widget)
            
            trending_label = QLabel("Trending Products")
            trending_label.setStyleSheet("font-size: 18px; font-weight: bold;")
            trending_layout.addWidget(trending_label)
            
            self.trending_list = QListWidget()
            self.trending_list.itemClicked.connect(self.product_selected)
            trending_layout.addWidget(self.trending_list)
            
            splitter.addWidget(trending_widget)

            # Right panel for analysis
            analysis_widget = QWidget()
            analysis_layout = QVBoxLayout(analysis_widget)
            
            # URL Input area
            url_layout = QHBoxLayout()
            
            self.search_entry = QLineEdit()
            self.search_entry.setPlaceholderText("Search for products...")
            self.search_button = QPushButton("Search")
            self.search_button.clicked.connect(self.search_products)
            
            self.url_entry = QLineEdit()
            self.url_entry.setPlaceholderText("Or enter Amazon Product URL...")
            self.analyze_button = QPushButton("Analyze")
            self.analyze_button.clicked.connect(self.start_analysis)
            
            url_layout.addWidget(self.search_entry)
            url_layout.addWidget(self.search_button)
            analysis_layout.addLayout(url_layout)
            
            url_direct_layout = QHBoxLayout()
            url_direct_layout.addWidget(self.url_entry)
            url_direct_layout.addWidget(self.analyze_button)
            analysis_layout.addLayout(url_direct_layout)
            
            # Status label
            self.status_label = QLabel("Ready to analyze")
            self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            analysis_layout.addWidget(self.status_label)
            
            # Progress bar
            self.progress_bar = QProgressBar()
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            analysis_layout.addWidget(self.progress_bar)
            
            # Results area
            self.results_text = QTextEdit()
            self.results_text.setReadOnly(True)
            analysis_layout.addWidget(self.results_text)
            
            splitter.addWidget(analysis_widget)
            
            # Set initial splitter sizes
            splitter.setSizes([400, 800])
            
            # Setup logging
            self.setup_logging()
            
            # Load trending products
            self.load_trending_products()
            
            # Add suggested searches
            self.suggested_list = QListWidget()
            self.suggested_list.itemClicked.connect(self.suggestion_selected)
            trending_layout.addWidget(self.suggested_list)
            
            # Add this line to populate the suggested searches
            self.update_suggested_searches()
            
            # Modify text display settings
            self.results_text.setReadOnly(True)
            # Set a larger font size for better readability
            font = self.results_text.font()
            font.setPointSize(12)  # Increase font size
            self.results_text.setFont(font)
            
            # Optional: Set a minimum height for the results area
            self.results_text.setMinimumHeight(400)
            
            # Make the status label more visible
            self.status_label.setStyleSheet("""
                QLabel {
                    font-size: 14px;
                    font-weight: bold;
                    padding: 5px;
                }
            """)
            
            # Make the progress bar taller
            self.progress_bar.setMinimumHeight(20)
            self.progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 2px solid grey;
                    border-radius: 5px;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #4CAF50;
                    width: 10px;
                }
            """)
            
        except Exception as e:
            logging.critical(f"Failed to initialize GUI: {str(e)}", exc_info=True)
            raise

    def load_trending_products(self):
        self.trending_worker = TrendingProductsWorker(self.scraper)
        self.trending_worker.products_ready.connect(self.update_trending_list)
        self.trending_worker.error.connect(self.handle_error)
        self.trending_worker.start()

    def update_trending_list(self, products):
        self.trending_list.clear()
        if not products:
            self.status_label.setText("No products found")
            return

        try:
            for product in products:
                item_text = f"{product.get('title', 'Unknown Title')}"
                if 'price' in product:
                    item_text += f" - ${product['price']}"
                if 'rating' in product:
                    item_text += f" | {product['rating']}★"
                if 'num_reviews' in product:
                    item_text += f" ({product['num_reviews']} reviews)"
                
                self.trending_list.addItem(item_text)
                self.trending_list.item(self.trending_list.count() - 1).setData(
                    Qt.ItemDataRole.UserRole, 
                    {'url': product.get('url', ''), 'seller': product.get('seller', 'Unknown')}
                )
            
            self.status_label.setText(f"Found {len(products)} products")
        except Exception as e:
            self.status_label.setText(f"Error displaying products: {str(e)}")
            logging.error(f"Error in update_trending_list: {str(e)}")

    def product_selected(self, item):
        # Get the dictionary from UserRole data
        product_data = item.data(Qt.ItemDataRole.UserRole)
        # Extract just the URL from the dictionary
        url = product_data['url']
        self.url_entry.setText(url)
        self.start_analysis()

    def setup_logging(self):
        class TextHandler(logging.Handler):
            def __init__(self, text_widget):
                super().__init__()
                self.text_widget = text_widget

            def emit(self, record):
                msg = self.format(record)
                self.text_widget.append(msg)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        logger = logging.getLogger()
        text_handler = TextHandler(self.results_text)
        text_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(text_handler)

    def start_analysis(self):
        try:
            url = self.url_entry.text().strip()
            
            # URL validation
            if not url:
                self.status_label.setText("Please enter a valid URL")
                return
                
            if not url.startswith(('http://', 'https://')):
                self.status_label.setText("Please enter a valid URL starting with http:// or https://")
                return
                
            if 'amazon.com' not in url.lower():
                self.status_label.setText("Please enter a valid Amazon product URL")
                return

            # Clear previous results
            self.results_text.clear()
            self.analyze_button.setEnabled(False)
            self.progress_bar.setValue(0)
            
            # Start loading animation
            self.spinner = LoadingSpinner("Analyzing product reviews")
            self.spinner.start()
            
            # Create and start worker thread
            self.worker = AnalysisWorker(url, self.scraper, self.cleaner)
            self.worker.progress.connect(self.update_progress)
            self.worker.status.connect(self.status_label.setText)
            self.worker.log.connect(self.results_text.append)
            self.worker.error.connect(self.handle_error)
            self.worker.finished.connect(lambda: self.cleanup_analysis())
            self.worker.start()
            
            logging.info("Analysis worker started")
            
        except Exception as e:
            self.handle_error(f"Failed to start analysis: {str(e)}")
            self.cleanup_analysis()
            logging.error("Analysis start failed", exc_info=True)

    def cleanup_analysis(self):
        """Clean up after analysis is complete or fails"""
        if hasattr(self, 'spinner'):
            self.spinner.stop()
        self.analyze_button.setEnabled(True)

    def update_progress(self, value):
        self.progress_bar.setValue(int(value * 100))

    def handle_error(self, error_msg):
        self.status_label.setText("Analysis failed!")
        self.results_text.append(f"Error: {error_msg}")

    def analysis_complete(self, spinner):
        spinner.stop()
        self.analyze_button.setEnabled(True)
        self.status_label.setText("Analysis complete!")

    def search_products(self):
        search_term = self.search_entry.text().strip()
        if not search_term:
            self.status_label.setText("Please enter a search term")
            return

        self.status_label.setText("Searching products...")
        self.search_button.setEnabled(False)
        self.trending_list.clear()

        self.search_worker = SearchProductsWorker(self.scraper, search_term)
        self.search_worker.products_ready.connect(self.update_trending_list)
        self.search_worker.error.connect(self.handle_error)
        self.search_worker.finished.connect(lambda: self.search_button.setEnabled(True))
        self.search_worker.start()

    def update_suggested_searches(self):
        try:
            self.suggested_list.clear()
            
            # Default popular categories and trending items
            default_suggestions = [
                # Electronics
                "Wireless Earbuds",
                "Smart Watch",
                "Bluetooth Speaker",
                "Gaming Headset",
                # Home & Kitchen
                "Air Fryer",
                "Coffee Maker",
                "Robot Vacuum",
                "Stand Mixer",
                # Popular Brands
                "Apple AirPods",
                "Samsung Galaxy",
                "Nintendo Switch",
                "Instant Pot",
                # Seasonal Items
                "Christmas Lights",
                "Winter Gloves",
                "Holiday Decor",
                # Tech Accessories
                "Phone Case",
                "Power Bank",
                "USB Cable",
                # Health & Personal Care
                "Vitamin D",
                "Face Mask",
                "Hand Sanitizer",
                # Entertainment
                "Board Games",
                "Video Games",
                "Kindle Books"
            ]
            
            # Add default suggestions to the list
            for suggestion in default_suggestions:
                if suggestion and isinstance(suggestion, str):  # Add validation
                    self.suggested_list.addItem(suggestion)
                
            # Try to get trending products as well
            try:
                trending_products = self.scraper.get_trending_products()
                if not trending_products:
                    logging.info("No trending products found")
                    return
                
                # Extract keywords from trending product titles
                keywords = set()
                for product in trending_products:
                    if not isinstance(product, dict):  # Add validation
                        continue
                    title = product.get('title', '').lower()
                    if not title:  # Skip empty titles
                        continue
                    words = [word for word in title.split() 
                            if len(word) > 3 and word not in {
                                'with', 'and', 'the', 'for', 'from', 'this',
                                'that', 'these', 'those', 'pack', 'size'
                            }]
                    keywords.update(words[:2])
                
                # Add trending keywords to the list
                for keyword in sorted(keywords):
                    if keyword:  # Add validation
                        self.suggested_list.addItem(keyword.capitalize())
            except Exception as e:
                logging.warning(f"Could not fetch trending products: {str(e)}")
                
        except Exception as e:
            logging.error(f"Error updating suggested searches: {str(e)}")
            self.handle_error(f"Failed to update suggestions: {str(e)}")

    def suggestion_selected(self, item):
        """Handle clicks on suggested search items"""
        search_term = item.text()
        self.search_entry.setText(search_term)
        self.search_products()

# Remove or comment out debug logging
logging.getLogger().setLevel(logging.ERROR)

def main():
    try:
        print("Starting application...")
        app = QApplication(sys.argv)
        print("Created QApplication")
        
        # Set up logging before creating the window
        logging.basicConfig(
            level=logging.DEBUG,  # Changed to DEBUG for more detailed logs
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)
        logger.info("Logging configured")
        
        app.setStyle('Fusion')
        logger.info("Set style")
        
        try:
            window = SentimentAnalyzerGUI()
            logger.info("Created window")
        except Exception as e:
            logger.error(f"Failed to create window: {str(e)}", exc_info=True)
            raise
            
        window.show()
        logger.info("Window should be visible now")
        
        return app.exec()
    except Exception as e:
        logger.error(f"Application failed to start: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    print("Script is running")
    sys.exit(main()) 