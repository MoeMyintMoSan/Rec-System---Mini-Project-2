import pandas as pd
import numpy as np
import csv
import math

def load_data(train_file, test_file):
    """Load training and test datasets"""
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    print(f"Training set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    print(f"Unique users in training: {train_df['userid'].nunique()}")
    print(f"Unique users in test: {test_df['userid'].nunique()}")
    print(f"Unique books in training: {train_df['isbn'].nunique()}")
    print(f"Unique books in test: {test_df['isbn'].nunique()}")
    
    return train_df, test_df

def create_user_item_matrix(df):
    """Create user-item rating matrix"""
    user_item_matrix = df.pivot_table(index='userid', columns='isbn', values='rating', fill_value=0)
    return user_item_matrix

def create_item_user_matrix(df):
    """Create item-user rating matrix (transpose of user-item)"""
    item_user_matrix = df.pivot_table(index='isbn', columns='userid', values='rating', fill_value=0)
    return item_user_matrix

def calculate_item_similarity(item1_ratings, item2_ratings):
    """Calculate Pearson correlation coefficient between two items using manual calculation"""
    # Find common users (users who have rated both items)
    common_users = (item1_ratings > 0) & (item2_ratings > 0)
    
    if common_users.sum() < 2:  # Need at least 2 common users
        return 0
    
    item1_common = item1_ratings[common_users]
    item2_common = item2_ratings[common_users]
    
    # Manual Pearson correlation calculation
    n = len(item1_common)
    
    # Calculate means
    mean1 = sum(item1_common) / n
    mean2 = sum(item2_common) / n
    
    # Calculate numerator and denominators
    numerator = sum((item1_common[i] - mean1) * (item2_common[i] - mean2) for i in range(n))
    
    sum_sq1 = sum((item1_common[i] - mean1) ** 2 for i in range(n))
    sum_sq2 = sum((item2_common[i] - mean2) ** 2 for i in range(n))
    
    denominator = math.sqrt(sum_sq1 * sum_sq2)
    
    # Handle division by zero (when one item has constant ratings)
    if denominator == 0:
        return 0
    
    correlation = numerator / denominator
    
    return correlation

def compute_item_similarity_matrix(item_user_matrix):
    """Compute Pearson correlation matrix for all item pairs"""
    items = item_user_matrix.index.tolist()
    n_items = len(items)
    
    # Initialize similarity matrix
    similarity_matrix = np.zeros((n_items, n_items))
    
    print("Computing item similarity matrix...")
    for i in range(n_items):
        for j in range(i, n_items):
            if i == j:
                similarity_matrix[i][j] = 1.0  # Item is perfectly similar to itself
            else:
                item1_ratings = item_user_matrix.iloc[i].values
                item2_ratings = item_user_matrix.iloc[j].values
                
                correlation = calculate_item_similarity(item1_ratings, item2_ratings)
                similarity_matrix[i][j] = correlation
                similarity_matrix[j][i] = correlation  # Symmetric matrix
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{n_items} items")
    
    return similarity_matrix, items

def create_user_profiles(train_df):
    """Create user-item rating matrix (sparse) where rows=users, columns=items, values=ratings"""
    user_item_matrix = train_df.pivot_table(index='userid', columns='isbn', values='rating', fill_value=0)
    return user_item_matrix

def save_user_profiles(user_item_matrix, filename):
    """Save user-item rating matrix to CSV file"""
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header (userid followed by all book ISBNs)
        header = ['userid'] + [str(isbn) for isbn in user_item_matrix.columns]
        writer.writerow(header)
        
        # Write user ratings
        for user_id in user_item_matrix.index:
            row = [str(user_id)] + [str(int(rating)) for rating in user_item_matrix.loc[user_id]]
            writer.writerow(row)
    
    print(f"User-item rating matrix saved to {filename}")

def save_item_similarity_matrix(similarity_matrix, items, filename):
    """Save item similarity matrix to CSV file"""
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        header = ['isbn'] + [str(item) for item in items]
        writer.writerow(header)
        
        # Write similarity values
        for i, item in enumerate(items):
            row = [str(item)] + [f"{similarity_matrix[i][j]:.6f}" for j in range(len(items))]
            writer.writerow(row)
    
    print(f"Item similarity matrix saved to {filename}")

def get_top_k_similar_items(target_item_idx, similarity_matrix, k=10):
    """Get top k most similar items for a target item"""
    similarities = similarity_matrix[target_item_idx]
    
    # Get indices of items sorted by similarity (excluding the item itself)
    sorted_indices = np.argsort(similarities)[::-1]
    
    # Remove the target item from the list (similarity = 1.0 with itself)
    sorted_indices = sorted_indices[sorted_indices != target_item_idx]
    
    # Return top k similar items
    return sorted_indices[:k]

def predict_item_rating(user_id, target_item_idx, user_item_matrix, similarity_matrix, items, k=10):
    """Predict rating for a user-item pair using k most similar items"""
    if user_id not in user_item_matrix.index:
        return 5.0  # Default rating if user not found
    
    user_ratings = user_item_matrix.loc[user_id]
    
    # Get top k similar items
    similar_item_indices = get_top_k_similar_items(target_item_idx, similarity_matrix, k)
    
    # Calculate weighted average rating
    numerator = 0
    denominator = 0
    
    for similar_item_idx in similar_item_indices:
        # Check if user has rated this similar item
        if user_ratings.iloc[similar_item_idx] > 0:
            similarity = similarity_matrix[target_item_idx][similar_item_idx]
            rating = user_ratings.iloc[similar_item_idx]
            
            numerator += similarity * rating
            denominator += abs(similarity)
    
    if denominator == 0:
        # If no similar items are rated by user, return user's average rating
        user_avg = user_ratings[user_ratings > 0].mean()
        return user_avg if not np.isnan(user_avg) else 5.0
    
    predicted_rating = numerator / denominator
    
    # Ensure rating is within valid range [1, 10]
    predicted_rating = max(1, min(10, predicted_rating))
    
    return predicted_rating

def generate_item_based_recommendations(test_df, train_df, user_item_matrix, similarity_matrix, items):
    """Generate top-10 recommendations for each test user using item-based filtering"""
    recommendations = []
    
    test_users = test_df['userid'].unique()
    
    for user_id in test_users:
        print(f"Generating recommendations for user {user_id}")
        
        if user_id not in user_item_matrix.index:
            print(f"Warning: User {user_id} not found in training data")
            continue
        
        # Get books this user has already rated
        user_rated_books = set(train_df[train_df['userid'] == user_id]['isbn'].tolist())
        
        # Get all available books
        all_books = set(items)
        
        # Find unrated books
        unrated_books = all_books - user_rated_books
        
        # Predict ratings for unrated books
        book_predictions = []
        
        for book in unrated_books:
            if book in items:
                book_idx = items.index(book)
                predicted_rating = predict_item_rating(user_id, book_idx, user_item_matrix, similarity_matrix, items)
                book_predictions.append((book, predicted_rating))
        
        # Sort by predicted rating (descending) and take top 10
        book_predictions.sort(key=lambda x: x[1], reverse=True)
        top_10_books = book_predictions[:10]
        
        # Create recommendation entries
        for book, predicted_rating in top_10_books:
            recommendation = {
                'UserID': user_id,
                'Book_ISBN': book,
                'Predicted_Rating': round(predicted_rating, 4)
            }
            recommendations.append(recommendation)
    
    return recommendations

def save_recommendations(recommendations, filename):
    """Save recommendations to CSV file"""
    if not recommendations:
        print("No recommendations to save")
        return
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['UserID', 'Book_ISBN', 'Predicted_Rating']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for rec in recommendations:
            writer.writerow(rec)
    
    print(f"Recommendations saved to {filename}")

def calculate_rmse(test_df, user_item_matrix, similarity_matrix, items):
    """Calculate RMSE for all test ratings"""
    predictions = []
    actual_ratings = []
    rmse_details = []
    
    for _, row in test_df.iterrows():
        user_id = row['userid']
        book_isbn = row['isbn']
        actual_rating = row['rating']
        
        if book_isbn in items and user_id in user_item_matrix.index:
            book_idx = items.index(book_isbn)
            predicted_rating = predict_item_rating(user_id, book_idx, user_item_matrix, similarity_matrix, items)
            
            predictions.append(predicted_rating)
            actual_ratings.append(actual_rating)
            
            # Store individual prediction details
            rmse_detail = {
                'UserID': user_id,
                'Book_ISBN': book_isbn,
                'Actual_Rating': actual_rating,
                'Predicted_Rating': round(predicted_rating, 4),
                'Squared_Error': round((actual_rating - predicted_rating) ** 2, 4)
            }
            rmse_details.append(rmse_detail)
    
    # Calculate overall RMSE
    if len(predictions) > 0:
        mse = sum((a - p) ** 2 for a, p in zip(actual_ratings, predictions)) / len(predictions)
        rmse = math.sqrt(mse)
    else:
        rmse = 0
    
    return rmse, rmse_details

def save_rmse_results(rmse, rmse_details, filename):
    """Save RMSE results to CSV file"""
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        # Write overall RMSE first
        f.write(f"Overall_RMSE,{rmse:.6f}\n")
        f.write(f"Total_Predictions,{len(rmse_details)}\n")
        f.write("\n")
        
        # Write detailed results
        fieldnames = ['UserID', 'Book_ISBN', 'Actual_Rating', 'Predicted_Rating', 'Squared_Error']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for detail in rmse_details:
            writer.writerow(detail)
    
    print(f"RMSE results saved to {filename}")
    print(f"Overall RMSE: {rmse:.6f}")

# File paths
train_file = 'rating10user91_trainset.csv'
test_file = 'rating10user91_testset.csv'
profile_file = 'P2Part2_1Profile_Group7.csv'
model_file = 'P2Part2_2Model_Group7.csv'
recommendation_file = 'P2Part2_3Recommendation_Group7.csv'
rmse_file = 'P2Part2_4RMSE_Group7.csv'

print("=== Item-based Collaborative Filtering Recommendation System ===\n")

# Step 1: Load data
print("1. Loading datasets...")
train_df, test_df = load_data(train_file, test_file)

# Step 2: Create matrices
print("\n2. Creating user-item and item-user matrices...")
user_item_matrix = create_user_item_matrix(train_df)
item_user_matrix = create_item_user_matrix(train_df)
print(f"User-item matrix shape: {user_item_matrix.shape}")
print(f"Item-user matrix shape: {item_user_matrix.shape}")

# Step 3: Create user profiles
print("\n3. Creating user-item rating matrix...")
user_profiles = create_user_profiles(train_df)
save_user_profiles(user_profiles, profile_file)

# Step 4: Compute item similarity matrix
print("\n4. Computing item similarity matrix...")
similarity_matrix, items = compute_item_similarity_matrix(item_user_matrix)
save_item_similarity_matrix(similarity_matrix, items, model_file)

# Step 5: Generate recommendations
print("\n5. Generating item-based recommendations...")
recommendations = generate_item_based_recommendations(test_df, train_df, user_item_matrix, similarity_matrix, items)
save_recommendations(recommendations, recommendation_file)

# Step 6: Calculate RMSE
print("\n6. Calculating RMSE...")
rmse, rmse_details = calculate_rmse(test_df, user_item_matrix, similarity_matrix, items)
save_rmse_results(rmse, rmse_details, rmse_file)

print(f"\nTotal recommendations generated: {len(recommendations)}")
print(f"Total RMSE calculations: {len(rmse_details)}")
print("\n=== Item-based Collaborative Filtering completed successfully! ===")
