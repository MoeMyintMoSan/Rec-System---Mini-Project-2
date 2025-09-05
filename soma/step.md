# Mini Project 2: Recommendation Systems - Step-by-Step Guide

## Overview
This project implements two recommendation algorithms:
1. **Part I**: User-based Nearest Neighbor algorithm (8%)
2. **Part II**: Collaborative-based filtering algorithm (7%)

## Dataset Structure
- `rating10user91_trainset.csv`: Training data (1142 records)
- `rating10user91_testset.csv`: Test data (202 records)
- Each file contains: userid, isbn, rating (1-10 scale)

---

## Part I: User-based Nearest Neighbor Algorithm (8%)

### Step 1: Environment Setup
1. Create a Python script (e.g., `part1_user_based.py`)
2. Import necessary libraries:
   ```python
   import pandas as pd
   import numpy as np
   from scipy.stats import pearsonr
   from sklearn.metrics.pairwise import cosine_similarity
   ```

### Step 2: Data Loading
1. Load training dataset: `rating10user91_trainset.csv`
2. Load test dataset: `rating10user91_testset.csv`
3. Examine data structure and basic statistics

### Step 3: Data Preprocessing
1. Create user-item matrix from training data
2. Handle missing values (users who haven't rated certain books)
3. Identify unique users and books

### Step 4: Similarity Calculation
1. **Implement Pearson Correlation Coefficient**:
   - Calculate similarity between all pairs of users
   - Store results in similarity matrix
   - Handle cases where correlation cannot be calculated

### Step 5: Prediction Algorithm
1. **For each user in test set**:
   - Find k=5 most similar users from training set
   - For each unrated book in test set:
     - Calculate predicted rating using weighted average
     - Weight by similarity scores of k nearest neighbors

### Step 6: Generate Recommendations
1. **For each user in test set**:
   - Identify books not yet rated (from test set)
   - Calculate predicted ratings for these books
   - Select top 5 books with highest predicted ratings

### Step 7: Output Results
1. **Create output file**: `P2Part1_2Recommendation_Group[group_no].csv`
2. **Format**: TargetUserID, 1stNNUserID, 2ndNNUserID, 3rdNNUserID, 4thNNUserID, 5thNNUserID, Book's ISBN, predicted rating
3. Display first 5 not-yet-read books for each user

---

## Part II: Collaborative-based Filtering Algorithm (7%)

### Step 1: Choose Different Approach
Select one of these collaborative filtering methods (different from Part I):
- **Item-based Collaborative Filtering**
- **Matrix Factorization (SVD)**
- **Non-negative Matrix Factorization (NMF)**
- **Cosine Similarity-based approach**

### Step 2: Implementation Steps

#### Option A: Item-based Collaborative Filtering
1. Create item-item similarity matrix
2. For each user-item pair in test set:
   - Find similar items the user has rated
   - Predict rating based on weighted average

#### Option B: Matrix Factorization (SVD)
1. Apply SVD to user-item matrix
2. Reduce dimensionality
3. Reconstruct ratings matrix
4. Extract predictions for test set

#### Option C: NMF Approach
1. Apply Non-negative Matrix Factorization
2. Learn user and item latent factors
3. Predict ratings using learned factors

### Step 3: Model Training
1. **Create user profiles**: Store in `P2Part2_1Profile_Group[group_no].csv`
2. **Store model/similarity matrix**: Save in `P2Part2_2Model_Group[group_no].csv`

### Step 4: Generate Predictions
1. **For each user in test set**:
   - Generate top 10 book recommendations
   - Include: User ID, Book's ISBN, model's calculated value, predicted rating

### Step 5: Evaluation
1. **Calculate RMSE** for all unseen items in test set
2. Compare predicted vs actual ratings
3. Store results in `P2Part2_4RMSE_Group[group_no].csv`

### Step 6: Output Results
1. **Recommendations**: `P2Part2_3Recommendation_Group[group_no].csv`
2. **RMSE Results**: `P2Part2_4RMSE_Group[group_no].csv`

---

## Implementation Tips

### Data Handling
- Use pandas for CSV operations
- Handle missing values appropriately
- Ensure consistent user/item IDs across datasets

### Similarity Calculations
- Implement efficient similarity computation
- Handle edge cases (users with no common ratings)
- Consider memory optimization for large datasets

### Prediction Strategy
- Implement weighted average properly
- Handle cases where no similar users exist
- Set reasonable defaults for edge cases

### Validation
- Verify output file formats match requirements
- Check that predictions are within valid range (1-10)
- Ensure all required users are included in output

---

## File Deliverables

### Part I Files:
- `P2Part1_2Recommendation_Group[group_no].csv`

### Part II Files:
- `P2Part2_1Profile_Group[group_no].csv`
- `P2Part2_2Model_Group[group_no].csv`
- `P2Part2_3Recommendation_Group[group_no].csv`
- `P2Part2_4RMSE_Group[group_no].csv`

### Code Files:
- `part1_user_based.py` (or similar)
- `part2_collaborative.py` (or similar)

---

## Testing and Debugging

1. **Verify data loading**: Check dataset dimensions and structure
2. **Test similarity calculations**: Manually verify a few similarity scores
3. **Validate predictions**: Ensure predicted ratings are reasonable
4. **Check output formats**: Match exactly with required CSV structure
5. **Performance testing**: Ensure algorithms run in reasonable time

Remember to replace `[group_no]` with your actual group number in all output files!
---

# Detailed Explanation & Presentation Preparation

## Project Walkthrough

This project demonstrates two recommendation algorithms using a real-world book rating dataset:
- **Part I:** User-based Nearest Neighbor
- **Part II:** Collaborative-based Filtering (e.g., Item-based, SVD, NMF)

### Part I: User-based Nearest Neighbor
1. **Environment Setup:**
   - Create `part1_user_based.py` and import pandas, numpy, scipy, sklearn.
2. **Data Loading:**
   - Load train/test CSVs with pandas. Inspect for missing values.
3. **Data Preprocessing:**
   - Build user-item rating matrix. Fill missing ratings. Identify unique users/books.
4. **Similarity Calculation:**
   - Calculate Pearson Correlation for all user pairs. Store similarity matrix.
5. **Prediction Algorithm:**
   - For each test user, find 5 nearest neighbors. Predict ratings for unseen books using weighted average.
6. **Generate Recommendations:**
   - For each user, select top 5 books with highest predicted ratings. Output user ID, neighbor IDs, ISBN, predicted rating.
7. **Output Results:**
   - Save recommendations to `P2Part1_2Recommendation_Group[group_no].csv`.

### Part II: Collaborative-based Filtering
1. **Choose a Different Approach:**
   - Use Item-based, SVD, NMF, or Cosine Similarity (not user-based).
2. **Implementation Steps:**
   - For Item-based: Compute item-item similarity, predict ratings.
   - For SVD/NMF: Factorize matrix, reconstruct ratings, predict for test users.
3. **Model Training:**
   - Store user/item profiles and model/similarity matrix in required CSVs.
4. **Generate Predictions:**
   - Recommend top 10 books for each user, save to CSV.
5. **Evaluation:**
   - Calculate RMSE for all unseen items, save results.

### Implementation Tips
- Use pandas for data handling.
- Handle missing values and edge cases.
- Validate output formats and predictions.
- Check performance and accuracy.

---

# Slide Presentation Outline

**Slide 1: Title**
- Mini Project 2: Recommendation Systems

**Slide 2: Project Overview**
- Two algorithms: User-based Nearest Neighbor & Collaborative Filtering

**Slide 3: Dataset**
- Description of train/test files and data structure

**Slide 4: Part I – User-based Nearest Neighbor**
- Steps: Data loading, preprocessing, similarity calculation, prediction, recommendation, output

**Slide 5: Part I – Key Techniques**
- Pearson Correlation, k-Nearest Neighbors, Weighted Average

**Slide 6: Part II – Collaborative Filtering**
- Options: Item-based, SVD, NMF, Cosine Similarity

**Slide 7: Part II – Steps**
- Model training, prediction, evaluation, output

**Slide 8: Evaluation**
- RMSE calculation and interpretation

**Slide 9: Implementation Tips**
- Data handling, validation, performance

**Slide 10: Results & Deliverables**
- Output files, code files, summary

**Slide 11: Q&A**

---

# Presentation Script

**Slide 1: Title**
"Hello everyone, today I’ll present Mini Project 2: Recommendation Systems, where we implement and compare two popular algorithms for book recommendations."

**Slide 2: Project Overview**
"This project is divided into two parts: Part I uses a user-based nearest neighbor approach, and Part II explores collaborative filtering methods such as item-based and matrix factorization."

**Slide 3: Dataset**
"We use two datasets: a training set and a test set, each containing user IDs, book ISBNs, and ratings from 1 to 10."

**Slide 4: Part I – User-based Nearest Neighbor**
"In Part I, we load and preprocess the data, build a user-item matrix, and calculate user similarities using the Pearson Correlation Coefficient. For each test user, we find their 5 most similar neighbors and predict ratings for books they haven’t read."

**Slide 5: Part I – Key Techniques**
"The key techniques are Pearson correlation for similarity and weighted average for rating prediction, focusing on the top 5 recommendations per user."

**Slide 6: Part II – Collaborative Filtering**
"In Part II, we use a different collaborative filtering method, such as item-based similarity or matrix factorization, to generate recommendations."

**Slide 7: Part II – Steps**
"We train the model, generate user/item profiles, predict ratings for unseen books, and recommend the top 10 books for each user."

**Slide 8: Evaluation**
"To evaluate our predictions, we calculate the RMSE between predicted and actual ratings, providing a measure of accuracy."

**Slide 9: Implementation Tips**
"Throughout, we use pandas for data handling, carefully manage missing values, and ensure our outputs match the required formats."

**Slide 10: Results & Deliverables**
"Our deliverables include recommendation CSV files, model profiles, RMSE results, and the Python scripts for each part."

**Slide 11: Q&A**
"Thank you for your attention. I’m happy to answer any questions!"