import pandas as pd
import re
from typing import Union, List
import logging

class DataCleaner:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def clean_text(self, text: str) -> str:
        """Clean individual text strings."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text

    def clean_reviews(self, reviews: Union[List[dict], pd.DataFrame]) -> pd.DataFrame:
        """Clean a collection of reviews."""
        try:
            # Convert to DataFrame if necessary
            if isinstance(reviews, list):
                df = pd.DataFrame(reviews)
            else:
                df = reviews.copy()
            
            # Log the initial state
            self.logger.info(f"Initial columns: {df.columns.tolist()}")
            
            # Check if required columns exist
            if 'text' not in df.columns:
                raise ValueError("Input data must contain a 'text' column")

            # Clean text columns
            df['cleaned_text'] = df['text'].fillna('').apply(self.clean_text)
            
            if 'title' in df.columns:
                df['cleaned_title'] = df['title'].fillna('').apply(self.clean_text)

            # Remove duplicate reviews (only if cleaned_text exists)
            if 'cleaned_text' in df.columns:
                df = df.drop_duplicates(subset=['cleaned_text'])
                # Remove empty reviews more safely
                df = df[df['cleaned_text'].fillna('').str.strip().str.len() > 0]

            return df

        except Exception as e:
            self.logger.error(f"Error cleaning reviews: {str(e)}")
            raise 

    def clean_data(self, df):
        """
        Clean and preprocess the review data
        
        Args:
            df (pandas.DataFrame): DataFrame containing review data
            
        Returns:
            pandas.DataFrame: Cleaned DataFrame
        """
        try:
            # Create a copy to avoid modifying the original
            cleaned_df = df.copy()
            
            # Log the incoming columns
            logging.debug(f"Input DataFrame columns: {list(cleaned_df.columns)}")
            
            # Convert ratings to numeric, looking for common rating column names
            rating_columns = ['rating', 'stars', 'score']
            rating_col = next((col for col in cleaned_df.columns if col.lower() in rating_columns), None)
            
            if rating_col:
                cleaned_df[rating_col] = pd.to_numeric(cleaned_df[rating_col], errors='coerce')
            
            # Clean review text, looking for common text column names
            text_columns = ['text', 'content', 'review', 'review_text']
            text_col = next((col for col in cleaned_df.columns if col.lower() in text_columns), None)
            
            if text_col:
                # Remove null values
                cleaned_df = cleaned_df.dropna(subset=[text_col])
                
                # Basic text cleaning
                cleaned_df[text_col] = cleaned_df[text_col].apply(self._clean_text)
                
                # Remove empty reviews
                cleaned_df = cleaned_df[cleaned_df[text_col].str.strip().str.len() > 0]
            else:
                logging.error(f"No text column found. Available columns: {list(cleaned_df.columns)}")
                raise ValueError(f"No text column found in DataFrame. Available columns: {list(cleaned_df.columns)}")
            
            return cleaned_df
            
        except Exception as e:
            logging.error(f"Error in clean_data: {str(e)}")
            raise
            
    def _clean_text(self, text):
        """Helper method to clean text data"""
        if not isinstance(text, str):
            return ""
            
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove special characters and extra whitespace
            import re
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            
            return text.strip()
            
        except Exception as e:
            logging.error(f"Error in _clean_text: {str(e)}")
            return ""