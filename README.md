# Customer Feedback Sentiment Analysis

## A Terminal-Based Data Analytics Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Pandas](https://img.shields.io/badge/Pandas-Latest-green)
![NLP](https://img.shields.io/badge/NLP-NLTK%2FTextBlob-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Project Overview

This is a complete terminal-based Data Analytics project that analyzes customer reviews and feedback to understand customer satisfaction, identify positive and negative sentiments, discover common complaints, and generate actionable business recommendations using Natural Language Processing (NLP) techniques.

**Key Features:**
- Terminal-based execution (no Jupyter, no web interface)
- Automated data processing and analysis
- Professional visualizations saved as PNG files
- Business-style summary report printed to terminal
- Beginner-friendly and internship-ready

---

## Business Objective

**Problem Statement:**
In today's competitive business environment, understanding customer feedback is crucial for:
- Improving product quality and customer satisfaction
- Identifying areas that need immediate attention
- Making data-driven business decisions
- Enhancing customer experience and loyalty

**Objectives:**
1. Analyze customer reviews to understand sentiment patterns
2. Identify the most loved and most complained products
3. Discover regional and category-based sentiment differences
4. Calculate a Customer Satisfaction Index (CSI)
5. Provide actionable business recommendations

---

## Dataset Description

### Dataset Overview
- **Total Reviews:** 1,500 customer reviews
- **Date Range:** June 2025 - June 2026
- **Product Categories:** 5 (Electronics, Fashion, Home Appliances, Beauty, Books)
- **Unique Products:** 50
- **Regions:** 5 (North America, Europe, Asia Pacific, Latin America, Middle East)

### Dataset Columns
| Column | Description |
|--------|-------------|
| Review ID | Unique identifier for each review |
| Review Date | Date when the review was posted |
| Product Category | Category of the product |
| Product Name | Name of the specific product |
| Customer Rating | Rating given by customer (1-5 stars) |
| Customer Feedback | Textual feedback from customer |
| Region | Geographic region of the customer |
| Sentiment | Pre-labeled sentiment (Positive, Negative, Neutral) |

### Sample Data
```
Review ID: REV0001
Review Date: 2025-06-15
Product Category: Electronics
Product Name: Wireless Bluetooth Headphones
Customer Rating: 5
Customer Feedback: Absolutely love this product! Best purchase I've made this year.
Region: North America
Sentiment: Positive
```

---

## Methodology

### 1. Data Preprocessing
- **Date Conversion:** Convert Review Date to datetime format
- **Text Cleaning:**
  - Convert to lowercase
  - Remove punctuation and special characters
  - Remove numbers
  - Remove stopwords (common words like "the", "is", "at")
  - Tokenization (split text into words)

### 2. Sentiment Analysis
- **Sentiment Score Calculation:** Using TextBlob's polarity score (-1 to +1)
  - Score > 0.1: Positive
  - Score < -0.1: Negative
  - Otherwise: Neutral
- **Sentiment Classification:** Categorize each review as Positive, Negative, or Neutral

### 3. Keyword Extraction
- **Positive Keywords:** Identify most common words in positive reviews
- **Negative Keywords:** Identify most common words in negative reviews
- **Word Frequency Analysis:** Count and rank word occurrences

### 4. Statistical Analysis
- **Sentiment Distribution:** Calculate percentage of each sentiment type
- **Category Analysis:** Compare sentiment across product categories
- **Regional Analysis:** Compare sentiment across geographic regions
- **CSI Calculation:** Customer Satisfaction Index metric

### 5. Visualization Generation
- 6 professional charts automatically generated and saved as PNG files
- High-resolution (300 DPI) for professional use

---

## Visualizations

The project generates **6 professional visualizations** saved in `outputs/charts/`:

### 1. Sentiment Distribution (Pie Chart)
- Visual representation of overall sentiment distribution
- Shows percentage of Positive, Neutral, and Negative reviews
- **File:** `01_sentiment_distribution.png`

### 2. Sentiment by Product Category (Stacked Bar Chart)
- Sentiment distribution across 5 product categories
- Identifies best and worst performing categories
- **File:** `02_sentiment_by_category.png`

### 3. Positive Word Frequency (Horizontal Bar Chart)
- Top 10 most common positive keywords
- Shows frequency of each keyword
- **File:** `03_positive_word_frequency.png`

### 4. Negative Word Frequency (Horizontal Bar Chart)
- Top 10 most common negative keywords
- Identifies key pain points and complaints
- **File:** `03_negative_word_frequency.png`

### 5. Rating Distribution (Bar Chart)
- Distribution of customer ratings (1-5 stars)
- Identifies most common rating patterns
- **File:** `04_rating_distribution.png`

### 6. Monthly Sentiment Trend (Line Chart)
- Sentiment trends over 12 months
- Tracks changes in customer sentiment over time
- **File:** `05_monthly_sentiment_trend.png`

---

## Key Findings

### Overall Sentiment
- **Total Reviews Analyzed:** 1,500
- **Average Rating:** 3.30/5.0
- **Positive Reviews:** 58.0%
- **Neutral Reviews:** 22.5%
- **Negative Reviews:** 19.5%

### Customer Satisfaction Index (CSI)
- **CSI Score:** 38.53
- **Status:** Good - High customer satisfaction
- **Formula:** CSI = (Positive %) - (Negative %)

### Product Category Performance
- **Best Performing:** Electronics (62.2% positive)
- **Category Needing Attention:** Beauty (55.1% positive)

### Regional Performance
- **Best Performing:** North America (64.8% positive)
- **Region Needing Attention:** Latin America (52.5% positive)

### Key Insights
- Top Positive Keywords: great, nothing, best, quality, works
- Top Negative Keywords: poor, quality, money, experience, terrible

---

## Recommendations

Based on the analysis, the following business recommendations are provided:

1. **Implement Feedback System:** Establish a robust customer feedback collection and response system

2. **Improve Product Quality:** Focus on improving product quality in underperforming categories (Beauty)

3. **Enhance Regional Support:** Enhance customer support and product offerings in Latin America

4. **Leverage Positive Reviews:** Use positive reviews and testimonials for marketing campaigns

5. **Address Complaints:** Address common complaints identified in negative reviews (poor quality, money, experience)

6. **Monitor Trends:** Monitor sentiment trends monthly to track improvement progress

7. **Quality Control:** Implement stricter quality control for products with high negative feedback

8. **Regional Customization:** Consider regional customization of products and services

---

## Tech Stack

### Core Technologies
- **Python 3.8+** - Primary programming language

### Libraries Used
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical data visualization
- **NLTK** - Natural language processing
- **TextBlob** - Sentiment analysis

---

## Project Structure

```
customer_feedback_sentiment/
│
├── data/
│   └── customer_reviews.csv          # Dataset with 1,500 reviews
│
├── src/
│   └── sentiment_analysis.py         # Main analysis script
│
├── outputs/
│   └── charts/                       # Generated visualizations
│       ├── 01_sentiment_distribution.png
│       ├── 02_sentiment_by_category.png
│       ├── 03_positive_word_frequency.png
│       ├── 03_negative_word_frequency.png
│       ├── 04_rating_distribution.png
│       └── 05_monthly_sentiment_trend.png
│
├── README.md                         # Project documentation
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore file
└── generate_dataset.py               # Dataset generator script
```

---

## How to Run

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation Steps

#### 1. Navigate to Project Directory
```bash
cd C:\customer_feedback_sentiment
```

#### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 3. Generate Dataset (First Time Only)
```bash
python generate_dataset.py
```

This will create `data/customer_reviews.csv` with 1,500 realistic customer reviews.

#### 4. Run Sentiment Analysis
```bash
python src\sentiment_analysis.py
```

This will:
- Load the dataset
- Clean and preprocess text
- Perform sentiment analysis
- Generate 6 visualization charts
- Print a business summary report to the terminal
- Save all charts to `outputs/charts/`

### Expected Output

After running the analysis, you will see:
- Progress messages for each step
- Sentiment distribution statistics
- Top positive and negative keywords
- Business summary report with:
  - Executive summary
  - Customer Satisfaction Index
  - Product category performance
  - Regional performance
  - Key insights
  - Business recommendations

All visualizations will be saved to `outputs/charts/` as high-resolution PNG files.

---

## Sample Terminal Output

```
================================================================================
CUSTOMER FEEDBACK SENTIMENT ANALYSIS
================================================================================

================================================================================
LOADING DATASET
================================================================================
[OK] Dataset loaded successfully: 1500 reviews
[OK] Columns: ['Review ID', 'Review Date', 'Product Category', ...]

================================================================================
DATA PREPROCESSING
================================================================================
[OK] Review Date converted to datetime
[OK] Cleaning feedback text...
[OK] Text cleaning completed
[OK] Calculating sentiment scores...
[OK] Sentiment classification completed

================================================================================
SENTIMENT ANALYSIS
================================================================================

Overall Sentiment Distribution:
--------------------------------------------------------------------------------
  Positive: 870 (58.0%)
  Neutral: 338 (22.5%)
  Negative: 292 (19.5%)

[... more output ...]

================================================================================
BUSINESS SUMMARY REPORT
================================================================================

EXECUTIVE SUMMARY
--------------------------------------------------------------------------------
Total Reviews Analyzed: 1,500
Average Customer Rating: 3.30/5.0
Positive Reviews: 870 (58.0%)
Neutral Reviews: 338 (22.5%)
Negative Reviews: 292 (19.5%)

CUSTOMER SATISFACTION INDEX (CSI)
--------------------------------------------------------------------------------
CSI Score: 38.53
Status: Good - High customer satisfaction

[... full business report ...]

================================================================================
ANALYSIS COMPLETED SUCCESSFULLY
================================================================================

[OK] All tasks completed successfully!
[OK] Visualizations saved to: C:\customer_feedback_sentiment\outputs\charts
```

---

## Project Highlights

### What Makes This Project Unique?

1. **Terminal-Based:** No Jupyter, no web interface - runs entirely from command line
2. **Automated:** Single command executes entire analysis pipeline
3. **Professional Output:** High-resolution visualizations and business report
4. **Beginner-Friendly:** Well-commented code, easy to understand
5. **Portfolio-Ready:** Professional structure and documentation
6. **CSI Metric:** Custom Customer Satisfaction Index for quick assessment

### Skills Demonstrated

- **Data Analysis:** Pandas, NumPy
- **NLP:** Text cleaning, tokenization, sentiment analysis
- **Visualization:** Matplotlib, Seaborn
- **Business Intelligence:** CSI calculation, reporting
- **Problem Solving:** End-to-end analytics pipeline

---

## Internship Information

**Project Type:** Data Analytics & NLP Internship Project  
**Domain:** Customer Experience & Sentiment Analysis  
**Tech Stack:** Python, NLP, Data Visualization  
**Dataset Size:** 1,500 Customer Reviews  
**Duration:** 2-3 Weeks (Recommended)  
**Role:** Data Analyst / NLP Engineer  

**Learning Outcomes:**
- Text preprocessing and NLP techniques
- Sentiment analysis with TextBlob
- Data visualization with Matplotlib and Seaborn
- Business intelligence and reporting
- Command-line application development

---

## Future Scope

### Potential Enhancements
1. **Real-time Analysis:** Implement real-time sentiment analysis for live feedback
2. **Machine Learning Models:** Train custom ML models for improved sentiment accuracy
3. **Multi-language Support:** Extend analysis to reviews in multiple languages
4. **Aspect-based Sentiment:** Analyze sentiment for specific product features
5. **Predictive Analytics:** Predict customer churn based on sentiment patterns
6. **Email Alerts:** Set up automated alerts for negative sentiment spikes

---

## License

This project is licensed under the MIT License - feel free to use it for learning and portfolio purposes.

---

## Author

**Data Analyst / NLP Engineer Intern**

**Skills Demonstrated:**
- Python Programming
- Data Analysis
- Natural Language Processing
- Data Visualization
- Business Intelligence

---

## Acknowledgments

- NLTK team for excellent NLP tools
- TextBlob for simple sentiment analysis
- Pandas and NumPy communities
- Matplotlib and Seaborn for visualization tools

---

## Star This Project

If you find this project helpful for your learning or portfolio, please consider giving it a star! ⭐

---

**Project Status:** ✅ Complete and Ready for Submission

**Last Updated:** June 2026

---

**Happy Analyzing! 🚀**
