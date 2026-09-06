# 🛍️ ShopSense | E-Commerce Product Recommendation System

An AI-powered e-commerce product recommendation system that provides personalized product suggestions based on customer purchase behavior.

The system uses **Collaborative Filtering with Singular Value Decomposition (SVD)** to learn customer-product interaction patterns and recommend products that a customer is likely to be interested in.

---

## 🚀 Project Overview

Online stores have thousands of products, making it difficult for customers to discover relevant items.

ShopSense solves this problem by analyzing historical customer-product interactions and generating personalized recommendations.

The project includes:

- Data cleaning and preprocessing
- Customer-product interaction matrix
- SVD-based collaborative filtering
- Personalized product recommendations
- Recommendation model evaluation
- Interactive Streamlit dashboard

---

## 📊 Dataset

**Dataset:** Online Retail Dataset

**Source:** UCI Machine Learning Repository

The dataset contains transactional data from a UK-based online retail business.

### Dataset Features

| Feature | Description |
|---|---|
| InvoiceNo | Invoice number |
| StockCode | Product code |
| Description | Product description |
| Quantity | Quantity purchased |
| InvoiceDate | Date and time of transaction |
| UnitPrice | Product unit price |
| CustomerID | Customer identifier |
| Country | Customer country |

### Dataset Size

- Original records: **541,909**
- Cleaned records: **392,692**
- Customers: **4,334**
- Products: **3,662**

---

## 🧹 Data Preprocessing

The dataset was cleaned before building the recommendation model.

The following steps were performed:

- Removed duplicate transactions
- Removed cancelled invoices
- Removed transactions with non-positive quantities
- Removed transactions with non-positive prices
- Removed records with missing Customer IDs
- Removed records with missing product descriptions
- Removed non-product/service codes

After preprocessing, the cleaned transaction data was used to construct the customer-product interaction matrix.

---

## 🧠 Recommendation Methodology

### 1. Customer-Product Interaction Matrix

A matrix was created where:

- Rows represent customers
- Columns represent products
- Values represent whether a customer interacted with a product

This produced an interaction matrix of:

**4,334 customers × 3,662 products**

### 2. Singular Value Decomposition

The interaction matrix was decomposed using **Truncated SVD**.

The model uses latent customer and product factors to estimate how relevant an unseen product may be for a customer.

### 3. Personalized Ranking

Products already purchased by the selected customer are removed from the recommendation candidates.

The remaining products are ranked according to their predicted recommendation scores.

---

## 📈 Model Evaluation

The recommender was evaluated using a leave-one-item-out test strategy.

| Model | Precision@10 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|
| SVD Recommender | 1.65% | 16.52% | 9.86% |
| Popularity Baseline | 0.53% | 5.33% | 2.76% |

The SVD recommender outperformed the popularity-based baseline across all three evaluation metrics.

---

## 💻 Streamlit Application

The project includes an interactive Streamlit dashboard where users can:

- Select a customer
- Choose the number of recommendations
- Generate personalized recommendations
- View recommendation scores
- Explore recommendation analytics
- View model performance
- Understand how the recommendation system works

---

## 🖥️ Project Structure

```text
E-Commerce-Product-Recommendation/
│
├── data/
│   └── Online Retail.xlsx
│
├── models/
│   ├── customer_factors.pkl
│   ├── product_factors.pkl
│   ├── train_matrix.pkl
│   ├── product_info.pkl
│   └── svd_model.pkl
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── app.py
├── requirements.txt
└── README.md