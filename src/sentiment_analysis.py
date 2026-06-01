"""
Customer Feedback Sentiment Analysis
A terminal-based Data Analytics project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from wordcloud import WordCloud
from collections import Counter
import os
import sys

# Set up matplotlib for non-interactive backend
plt.switch_backend('Agg')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class SentimentAnalyzer:
    """Main class for sentiment analysis"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.df = None
        self.positive_keywords = []
        self.negative_keywords = []
        
    def load_data(self, filepath):
        """Load customer reviews dataset"""
        print("=" * 80)
        print("LOADING DATASET")
        print("=" * 80)
        self.df = pd.read_csv(filepath)
        print(f"[OK] Dataset loaded successfully: {len(self.df)} reviews")
        print(f"[OK] Columns: {list(self.df.columns)}")
        print()
        
    def clean_text(self, text):
        """Clean text by removing punctuation, stopwords, and converting to lowercase"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)
    
    def preprocess_data(self):
        """Preprocess the data"""
        print("=" * 80)
        print("DATA PREPROCESSING")
        print("=" * 80)
        
        # Convert date to datetime
        self.df['Review Date'] = pd.to_datetime(self.df['Review Date'])
        print("[OK] Review Date converted to datetime")
        
        # Clean feedback text
        print("[OK] Cleaning feedback text...")
        self.df['Cleaned_Feedback'] = self.df['Customer Feedback'].apply(self.clean_text)
        print("[OK] Text cleaning completed")
        
        # Calculate sentiment scores using TextBlob
        print("[OK] Calculating sentiment scores...")
        self.df['Sentiment_Score'] = self.df['Customer Feedback'].apply(
            lambda x: TextBlob(x).sentiment.polarity
        )
        
        # Classify sentiment based on score
        def classify_sentiment(score):
            if score > 0.1:
                return 'Positive'
            elif score < -0.1:
                return 'Negative'
            else:
                return 'Neutral'
        
        self.df['Calculated_Sentiment'] = self.df['Sentiment_Score'].apply(classify_sentiment)
        print("[OK] Sentiment classification completed")
        print()
        
    def extract_keywords(self, sentiment_type, top_n=15):
        """Extract most common keywords for a given sentiment"""
        reviews = self.df[self.df['Calculated_Sentiment'] == sentiment_type]['Cleaned_Feedback']
        all_words = []
        
        for text in reviews:
            words = word_tokenize(text)
            all_words.extend(words)
        
        word_counts = Counter(all_words)
        
        # Remove common words
        common_words_to_remove = ['product', 'purchase', 'buy', 'item', 'use', 'using', 'get', 'got']
        for word in common_words_to_remove:
            if word in word_counts:
                del word_counts[word]
        
        return word_counts.most_common(top_n)
    
    def analyze_sentiment(self):
        """Perform sentiment analysis"""
        print("=" * 80)
        print("SENTIMENT ANALYSIS")
        print("=" * 80)
        
        # Extract keywords
        self.positive_keywords = self.extract_keywords('Positive')
        self.negative_keywords = self.extract_keywords('Negative')
        
        # Print sentiment distribution
        sentiment_counts = self.df['Calculated_Sentiment'].value_counts()
        sentiment_percentages = self.df['Calculated_Sentiment'].value_counts(normalize=True) * 100
        
        print("\nOverall Sentiment Distribution:")
        print("-" * 80)
        for sentiment in sentiment_counts.index:
            count = sentiment_counts[sentiment]
            percentage = sentiment_percentages[sentiment]
            print(f"  {sentiment}: {count:,} ({percentage:.1f}%)")
        
        print("\nTop 10 Positive Keywords:")
        print("-" * 80)
        for word, count in self.positive_keywords[:10]:
            print(f"  {word}: {count}")
        
        print("\nTop 10 Negative Keywords:")
        print("-" * 80)
        for word, count in self.negative_keywords[:10]:
            print(f"  {word}: {count}")
        
        print()
        
    def create_visualizations(self, output_dir):
        """Create and save all visualizations"""
        print("=" * 80)
        print("GENERATING VISUALIZATIONS")
        print("=" * 80)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Sentiment Distribution
        print("[OK] Creating: Sentiment Distribution Chart")
        self._plot_sentiment_distribution(output_dir)
        
        # 2. Sentiment by Category
        print("[OK] Creating: Sentiment by Category Chart")
        self._plot_sentiment_by_category(output_dir)
        
        # 3. Positive Word Frequency
        print("[OK] Creating: Positive Word Frequency Chart")
        self._plot_word_frequency(self.positive_keywords[:10], 'Positive', output_dir)
        
        # 4. Negative Word Frequency
        print("[OK] Creating: Negative Word Frequency Chart")
        self._plot_word_frequency(self.negative_keywords[:10], 'Negative', output_dir)
        
        # 5. Rating Distribution
        print("[OK] Creating: Rating Distribution Chart")
        self._plot_rating_distribution(output_dir)
        
        # 6. Monthly Sentiment Trend
        print("[OK] Creating: Monthly Sentiment Trend Chart")
        self._plot_monthly_trend(output_dir)
        
        print(f"\n[OK] All visualizations saved to: {output_dir}")
        print()
        
    def _plot_sentiment_distribution(self, output_dir):
        """Plot sentiment distribution pie chart"""
        sentiment_counts = self.df['Calculated_Sentiment'].value_counts()
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
        ax.set_title('Sentiment Distribution', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/01_sentiment_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_sentiment_by_category(self, output_dir):
        """Plot sentiment by product category"""
        sentiment_category = pd.crosstab(self.df['Product Category'], self.df['Calculated_Sentiment'])
        sentiment_category_pct = sentiment_category.div(sentiment_category.sum(axis=1), axis=0) * 100
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sentiment_category_pct.plot(kind='bar', stacked=True, 
                                    color=['#2ecc71', '#f39c12', '#e74c3c'],
                                    ax=ax, edgecolor='black', linewidth=1)
        ax.set_title('Sentiment Distribution by Product Category', fontsize=14, fontweight='bold')
        ax.set_xlabel('Product Category', fontsize=12, fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/02_sentiment_by_category.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_word_frequency(self, keywords, sentiment_type, output_dir):
        """Plot word frequency for given sentiment"""
        words = [word for word, count in keywords]
        counts = [count for word, count in keywords]
        color = '#2ecc71' if sentiment_type == 'Positive' else '#e74c3c'
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(words, counts, color=color, edgecolor='black', linewidth=1)
        ax.set_title(f'Top 10 {sentiment_type} Keywords', fontsize=14, fontweight='bold')
        ax.set_xlabel('Frequency', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{int(width)}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/03_{sentiment_type.lower()}_word_frequency.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_rating_distribution(self, output_dir):
        """Plot rating distribution histogram"""
        rating_counts = self.df['Customer Rating'].value_counts().sort_index()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(rating_counts.index, rating_counts.values, color='#3498db', 
                      edgecolor='black', linewidth=2, alpha=0.8)
        ax.set_title('Customer Rating Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Rating (1-5 Stars)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_xticks(range(1, 6))
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/04_rating_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_monthly_trend(self, output_dir):
        """Plot monthly sentiment trend"""
        self.df['Month'] = self.df['Review Date'].dt.to_period('M')
        monthly_sentiment = pd.crosstab(self.df['Month'], self.df['Calculated_Sentiment'], normalize='index') * 100
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for sentiment in ['Positive', 'Neutral', 'Negative']:
            if sentiment in monthly_sentiment.columns:
                color = '#2ecc71' if sentiment == 'Positive' else '#f39c12' if sentiment == 'Neutral' else '#e74c3c'
                ax.plot(monthly_sentiment.index.astype(str), monthly_sentiment[sentiment], 
                        marker='o', label=sentiment, color=color, linewidth=2.5, markersize=6)
        
        ax.set_title('Monthly Sentiment Trend', fontsize=14, fontweight='bold')
        ax.set_xlabel('Month', fontsize=12, fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax.legend(title='Sentiment')
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/05_monthly_sentiment_trend.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_report(self):
        """Generate and print business summary report"""
        print("=" * 80)
        print("BUSINESS SUMMARY REPORT")
        print("=" * 80)
        
        # Calculate statistics
        total_reviews = len(self.df)
        avg_rating = self.df['Customer Rating'].mean()
        sentiment_counts = self.df['Calculated_Sentiment'].value_counts()
        sentiment_percentages = self.df['Calculated_Sentiment'].value_counts(normalize=True) * 100
        
        # CSI Calculation
        positive_count = sentiment_counts.get('Positive', 0)
        negative_count = sentiment_counts.get('Negative', 0)
        csi = (positive_count / total_reviews * 100) - (negative_count / total_reviews * 100)
        
        # Category analysis
        category_analysis = self.df.groupby('Product Category').agg({
            'Customer Rating': 'mean',
            'Calculated_Sentiment': lambda x: (x == 'Positive').mean() * 100
        }).round(2)
        category_analysis.columns = ['Avg Rating', 'Positive %']
        
        # Regional analysis
        region_analysis = self.df.groupby('Region').agg({
            'Customer Rating': 'mean',
            'Calculated_Sentiment': lambda x: (x == 'Positive').mean() * 100
        }).round(2)
        region_analysis.columns = ['Avg Rating', 'Positive %']
        
        print("\nEXECUTIVE SUMMARY")
        print("-" * 80)
        print(f"Total Reviews Analyzed: {total_reviews:,}")
        print(f"Average Customer Rating: {avg_rating:.2f}/5.0")
        print(f"Positive Reviews: {sentiment_counts.get('Positive', 0):,} ({sentiment_percentages.get('Positive', 0):.1f}%)")
        print(f"Neutral Reviews: {sentiment_counts.get('Neutral', 0):,} ({sentiment_percentages.get('Neutral', 0):.1f}%)")
        print(f"Negative Reviews: {sentiment_counts.get('Negative', 0):,} ({sentiment_percentages.get('Negative', 0):.1f}%)")
        
        print("\nCUSTOMER SATISFACTION INDEX (CSI)")
        print("-" * 80)
        print(f"CSI Score: {csi:.2f}")
        if csi >= 30:
            print("Status: Good - High customer satisfaction")
        elif csi >= 10:
            print("Status: Satisfactory - Moderate customer satisfaction")
        elif csi >= 0:
            print("Status: Fair - Balanced customer sentiment")
        else:
            print("Status: Poor - More negative than positive feedback")
        
        print("\nPRODUCT CATEGORY PERFORMANCE")
        print("-" * 80)
        for category, row in category_analysis.iterrows():
            print(f"{category}:")
            print(f"  Average Rating: {row['Avg Rating']}/5.0 | Positive: {row['Positive %']:.1f}%")
        
        print("\nREGIONAL PERFORMANCE")
        print("-" * 80)
        for region, row in region_analysis.iterrows():
            print(f"{region}:")
            print(f"  Average Rating: {row['Avg Rating']}/5.0 | Positive: {row['Positive %']:.1f}%")
        
        print("\nKEY INSIGHTS")
        print("-" * 80)
        best_category = category_analysis['Positive %'].idxmax()
        worst_category = category_analysis['Positive %'].idxmin()
        print(f"[OK] Best Performing Category: {best_category}")
        print(f"[!] Category Needing Attention: {worst_category}")
        
        best_region = region_analysis['Positive %'].idxmax()
        worst_region = region_analysis['Positive %'].idxmin()
        print(f"[OK] Best Performing Region: {best_region}")
        print(f"[!] Region Needing Attention: {worst_region}")
        
        print(f"\n[OK] Top Positive Keywords: {', '.join([word for word, count in self.positive_keywords[:5]])}")
        print(f"[OK] Top Negative Keywords: {', '.join([word for word, count in self.negative_keywords[:5]])}")
        
        print("\nBUSINESS RECOMMENDATIONS")
        print("-" * 80)
        recommendations = [
            "1. Implement robust customer feedback collection and response system",
            "2. Focus on improving product quality in underperforming categories",
            f"3. Enhance customer support and product offerings in {worst_region}",
            "4. Leverage positive reviews and testimonials for marketing campaigns",
            "5. Address common complaints identified in negative reviews",
            "6. Monitor sentiment trends monthly to track improvement progress",
            "7. Implement stricter quality control for products with high negative feedback",
            "8. Consider regional customization of products and services"
        ]
        
        for rec in recommendations:
            print(rec)
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print()


def main():
    """Main function to run the sentiment analysis"""
    print("\n")
    print("=" * 80)
    print("CUSTOMER FEEDBACK SENTIMENT ANALYSIS")
    print("=" * 80)
    print()
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Get paths
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(script_dir, 'data', 'customer_reviews.csv')
    output_dir = os.path.join(script_dir, 'outputs', 'charts')
    
    # Check if dataset exists
    if not os.path.exists(data_path):
        print(f"[ERROR] Dataset not found at {data_path}")
        print("Please run generate_dataset.py first to create the dataset.")
        sys.exit(1)
    
    try:
        # Run analysis pipeline
        analyzer.load_data(data_path)
        analyzer.preprocess_data()
        analyzer.analyze_sentiment()
        analyzer.create_visualizations(output_dir)
        analyzer.generate_report()
        
        print("[OK] All tasks completed successfully!")
        print(f"[OK] Visualizations saved to: {output_dir}")
        print()
        
    except Exception as e:
        print(f"[ERROR] Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
