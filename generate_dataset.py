"""
Customer Reviews Dataset Generator
Generates realistic customer review data for sentiment analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Product categories and products
product_data = {
    'Electronics': [
        'Wireless Bluetooth Headphones', 'Smart Watch Pro', 'Laptop Stand',
        'USB-C Hub', 'Wireless Mouse', 'Mechanical Keyboard',
        'Portable Charger', 'Webcam HD', 'Smartphone Case',
        'Tablet Screen Protector'
    ],
    'Fashion': [
        'Cotton T-Shirt', 'Denim Jeans', 'Running Shoes',
        'Winter Jacket', 'Summer Dress', 'Leather Belt',
        'Sunglasses', 'Backpack', 'Wrist Watch',
        'Casual Sneakers'
    ],
    'Home Appliances': [
        'Air Fryer', 'Coffee Maker', 'Blender',
        'Vacuum Cleaner', 'Iron', 'Toaster',
        'Electric Kettle', 'Microwave Oven', 'Fan',
        'Heater'
    ],
    'Beauty': [
        'Face Moisturizer', 'Lipstick Set', 'Hair Dryer',
        'Makeup Kit', 'Perfume', 'Skincare Set',
        'Hair Straightener', 'Nail Polish', 'Body Lotion',
        'Sunscreen'
    ],
    'Books': [
        'Python Programming Book', 'Novel Collection', 'Self-Help Book',
        'Cooking Recipes', 'Science Fiction', 'History Book',
        'Business Strategy', 'Art Design', 'Children Stories',
        'Biography'
    ]
}

# Regions
regions = ['North America', 'Europe', 'Asia Pacific', 'Latin America', 'Middle East']

# Review templates by sentiment
positive_reviews = [
    "Absolutely love this {product}! Best purchase I've made this year.",
    "Great quality and fast shipping. The {product} exceeded my expectations.",
    "Perfect! Exactly what I was looking for. Highly recommend the {product}.",
    "Amazing product! The {product} works perfectly and looks great.",
    "Very satisfied with my {product}. Will definitely buy again.",
    "Excellent quality for the price. The {product} is fantastic.",
    "Super happy with this {product}. It's even better than described.",
    "Outstanding product! The {product} has made my life so much easier.",
    "Best {product} I've ever owned. Worth every penny!",
    "Couldn't be happier with my {product}. Five stars all the way!",
    "The {product} is exactly what I needed. Great value for money.",
    "Fantastic quality and arrived quickly. Love my new {product}!",
    "This {product} is a game-changer. So happy with my purchase.",
    "Perfect condition and works perfectly. The {product} is amazing!",
    "Great investment! The {product} has exceeded all my expectations."
]

negative_reviews = [
    "Very disappointed with this {product}. Poor quality and broke after a week.",
    "Not worth the money. The {product} stopped working after a few days.",
    "Terrible experience. The {product} arrived damaged and customer service was unhelpful.",
    "Would not recommend this {product}. Complete waste of money.",
    "The {product} is nothing like the pictures. Very misleading.",
    "Poor quality control. My {product} had multiple defects.",
    "Regret buying this {product}. It's overpriced for what you get.",
    "Worst purchase ever. The {product} doesn't work as advertised.",
    "Very frustrated with this {product}. Save your money and buy something else.",
    "The {product} arrived late and in poor condition. Very disappointed.",
    "Absolutely terrible {product}. Broke after first use.",
    "Not happy at all. The {product} is cheaply made and unreliable.",
    "Avoid this {product}. It's a complete waste of time and money.",
    "The {product} is nothing special. Expected much better for the price.",
    "Poor quality and bad customer service. Do not buy this {product}."
]

neutral_reviews = [
    "The {product} is okay. Nothing special but does the job.",
    "Average quality for the price. The {product} meets basic expectations.",
    "It's a decent {product}. Not great, not terrible.",
    "The {product} is fine for basic use. Could be better.",
    "Received the {product} as described. No complaints, no praise.",
    "Standard quality {product}. Does what it's supposed to do.",
    "The {product} is adequate. Meets minimum requirements.",
    "It's an okay {product}. Nothing to write home about.",
    "Basic functionality works. The {product} is average at best.",
    "The {product} is decent value. Nothing extraordinary.",
    "Middle of the road {product}. Acceptable but not impressive.",
    "The {product} works as expected. Nothing more, nothing less.",
    "It's a standard {product}. Does the job but nothing special.",
    "Average experience with the {product}. Could use some improvements.",
    "The {product} is satisfactory. Meets basic needs."
]

# Generate random reviews
def generate_review_text(product, sentiment):
    if sentiment == 'Positive':
        return random.choice(positive_reviews).format(product=product)
    elif sentiment == 'Negative':
        return random.choice(negative_reviews).format(product=product)
    else:
        return random.choice(neutral_reviews).format(product=product)

# Generate dataset
np.random.seed(42)
random.seed(42)

num_reviews = 1500
reviews = []

for i in range(num_reviews):
    review_id = f"REV{i+1:04d}"
    
    # Random date within last 12 months
    start_date = datetime.now() - timedelta(days=365)
    random_days = random.randint(0, 365)
    review_date = start_date + timedelta(days=random_days)
    
    # Random category and product
    category = random.choice(list(product_data.keys()))
    product = random.choice(product_data[category])
    
    # Rating based on sentiment
    sentiment_roll = random.random()
    if sentiment_roll < 0.45:
        sentiment = 'Positive'
        rating = random.choice([4, 5])
    elif sentiment_roll < 0.75:
        sentiment = 'Neutral'
        rating = random.choice([3])
    else:
        sentiment = 'Negative'
        rating = random.choice([1, 2])
    
    # Generate review text
    feedback = generate_review_text(product, sentiment)
    
    # Random region
    region = random.choice(regions)
    
    reviews.append({
        'Review ID': review_id,
        'Review Date': review_date.strftime('%Y-%m-%d'),
        'Product Category': category,
        'Product Name': product,
        'Customer Rating': rating,
        'Customer Feedback': feedback,
        'Region': region,
        'Sentiment': sentiment
    })

# Create DataFrame
df = pd.DataFrame(reviews)

# Save to CSV
df.to_csv('data/customer_reviews.csv', index=False)

print(f"Dataset generated successfully with {len(df)} reviews!")
print(f"\nDataset Info:")
print(f"- Total Reviews: {len(df)}")
print(f"- Date Range: {df['Review Date'].min()} to {df['Review Date'].max()}")
print(f"- Product Categories: {df['Product Category'].nunique()}")
print(f"- Unique Products: {df['Product Name'].nunique()}")
print(f"- Regions: {df['Region'].nunique()}")
print(f"\nSentiment Distribution:")
print(df['Sentiment'].value_counts())
print(f"\nRating Distribution:")
print(df['Customer Rating'].value_counts().sort_index())
