# Shopper-Spectrum-Customer-Segmentation-and-Product-Recommendations-in-E-Commerce
Customer segmentation using RFM analysis and KMeans clustering, combined with a product recommendation system based on collaborative filtering. Includes data preprocessing, similarity computation, and an interactive Streamlet app for real-time predictions and recommendations in e-commerce.

## Project Overview

This project focuses on analyzing e-commerce transaction data to segment customers using RFM (Recency, Frequency, Monetary) analysis and KMeans clustering. It also builds a product recommendation engine using item-based collaborative filtering. The goal is to help online retailers better understand their customer base and improve customer experience through data-driven product suggestions.

## Problem Statement

E-commerce businesses generate massive amounts of transaction data. However, most do not fully leverage this data to understand customer behavior or personalize shopping experiences. This project solves that by:
- Segmenting customers into actionable groups using unsupervised learning.
- Recommending similar products based on historical purchase behavior.

## Key Features

- **Customer Segmentation:** Predicts customer segment using RFM values and a trained KMeans model.
- **Product Recommendations:** Returns top 5 similar products using precomputed cosine similarity between products.
- **Streamlit Dashboard:** A user-friendly interface for real-time input and results display.

## Dataset Description

The dataset used includes transactions from an online retail store and has the following columns:

- `InvoiceNo`: Unique invoice number for each transaction
- `StockCode`: Unique identifier for each product
- `Description`: Product name
- `Quantity`: Number of products purchased
- `InvoiceDate`: Date and time of purchase
- `UnitPrice`: Price per product
- `CustomerID`: Unique identifier for each customer
- `Country`: Country of the customer

## Technical Details

### Customer Segmentation

- Calculated RFM values per customer.
- Scaled RFM values using StandardScaler.
- Applied KMeans clustering.
- Cluster labels were interpreted as:
  - High-Value
  - Regular
  - Occasional
  - At-Risk

### Product Recommendation

- Created a product-user purchase matrix.
- Calculated cosine similarity between products.
- Stored the similarity matrix using pickle (`product_similarity.pkl`).

### Tools and Libraries

- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Matplotlib / Seaborn (for EDA)
- Git / Git LFS (for large file handling)

## Folder Structure

├── Streamlit/
│ ├── customer_recomandation.py # Streamlit app
│ ├── kmeans.pkl # Trained clustering model
│ ├── scaler.pkl # Scaler used for RFM features
│ ├── product_similarity.pkl # Precomputed product similarity matrix
├── Customer Segmentation.ipynb # EDA, RFM analysis, clustering
├── README.md # Project documentation
├── online_retail.csv # Dataset used
