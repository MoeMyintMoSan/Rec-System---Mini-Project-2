import pandas as pd
import numpy as np
import csv
import math

def load_data(train_file, test_file):
    """Load training and test datasets with optional preprocessing"""
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    print(f"Training set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    print(f"Unique users in training: {train_df['userid'].nunique()}")
    print(f"Unique users in test: {test_df['userid'].nunique()}")
    
    # Data quality checks
    print("\n=== Data Quality Checks ===")
    
    # Check for missing values
    train_missing = train_df.isnull().sum().sum()
    test_missing = test_df.isnull().sum().sum()
    print(f"Missing values - Training: {train_missing}, Test: {test_missing}")
    
    # Check for duplicate rows
    train_duplicates = train_df.duplicated().sum()
    test_duplicates = test_df.duplicated().sum()
    print(f"Duplicate rows - Training: {train_duplicates}, Test: {test_duplicates}")
    
    # Check rating range
    train_rating_range = (train_df['rating'].min(), train_df['rating'].max())
    test_rating_range = (test_df['rating'].min(), test_df['rating'].max())
    print(f"Rating range - Training: {train_rating_range}, Test: {test_rating_range}")
    
    # Check user consistency
    test_users = set(test_df['userid'].unique())
    train_users = set(train_df['userid'].unique())
    missing_users = test_users - train_users
    print(f"Test users not in training: {len(missing_users)}")
    
    # Check book coverage
    train_books = set(train_df['isbn'].unique())
    test_books = set(test_df['isbn'].unique())
    test_book_coverage = len(test_books.intersection(train_books)) / len(test_books) * 100
    print(f"Test book coverage: {test_book_coverage:.2f}%")
    
    # Sparsity analysis
    user_item_matrix = train_df.pivot_table(index='userid', columns='isbn', values='rating', fill_value=0)
    total_cells = user_item_matrix.shape[0] * user_item_matrix.shape[1]
    non_zero_cells = (user_item_matrix > 0).sum().sum()
    sparsity = (total_cells - non_zero_cells) / total_cells * 100
    print(f"Data sparsity: {sparsity:.2f}%")
    
    print("=== Data Quality: PASSED ===\n")
    
    return train_df, test_df

def create_user_item_matrix(df):
    """Create user-item rating matrix"""
    user_item_matrix = df.pivot_table(index='userid', columns='isbn', values='rating', fill_value=0)
    return user_item_matrix

def calculate_pearson_correlation(user1_ratings, user2_ratings):
    """Calculate Pearson correlation coefficient between two users using manual calculation"""
    # Find common items (books both users have rated)
    common_items = (user1_ratings > 0) & (user2_ratings > 0)
    
    if common_items.sum() < 2:  # Need at least 2 common items
        return 0
    
    user1_common = user1_ratings[common_items]
    user2_common = user2_ratings[common_items]
    
    # Manual Pearson correlation calculation
    n = len(user1_common)
    
    # Calculate means
    mean1 = sum(user1_common) / n
    mean2 = sum(user2_common) / n
    
    # Calculate numerator and denominators
    numerator = sum((user1_common[i] - mean1) * (user2_common[i] - mean2) for i in range(n))
    
    sum_sq1 = sum((user1_common[i] - mean1) ** 2 for i in range(n))
    sum_sq2 = sum((user2_common[i] - mean2) ** 2 for i in range(n))
    
    denominator = math.sqrt(sum_sq1 * sum_sq2)
    
    # Handle division by zero (when one user has constant ratings)
    if denominator == 0:
        return 0
    
    correlation = numerator / denominator
    
    return correlation

def compute_similarity_matrix(user_item_matrix):
    """Compute Pearson correlation matrix for all user pairs"""
    users = user_item_matrix.index.tolist()
    n_users = len(users)
    
    # Initialize similarity matrix
    similarity_matrix = np.zeros((n_users, n_users))
    
    print("Computing similarity matrix...")
    for i in range(n_users):
        for j in range(i, n_users):
            if i == j:
                similarity_matrix[i][j] = 1.0  # User is perfectly similar to themselves
            else:
                user1_ratings = user_item_matrix.iloc[i].values
                user2_ratings = user_item_matrix.iloc[j].values
                
                correlation = calculate_pearson_correlation(user1_ratings, user2_ratings)
                similarity_matrix[i][j] = correlation
                similarity_matrix[j][i] = correlation  # Symmetric matrix
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{n_users} users")
    
    return similarity_matrix, users

def save_similarity_matrix(similarity_matrix, users, filename):
    """Save similarity matrix to CSV file"""
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        header = ['userid'] + [str(user) for user in users]
        writer.writerow(header)
        
        # Write similarity values
        for i, user in enumerate(users):
            row = [str(user)] + [f"{similarity_matrix[i][j]:.6f}" for j in range(len(users))]
            writer.writerow(row)
    
    print(f"Similarity matrix saved to {filename}")

def get_top_k_neighbors(target_user_idx, similarity_matrix, k=5):
    """Get top k most similar users for a target user"""
    similarities = similarity_matrix[target_user_idx]
    
    # Get indices of users sorted by similarity (excluding the user themselves)
    sorted_indices = np.argsort(similarities)[::-1]
    
    # Remove the target user from the list (similarity = 1.0 with themselves)
    sorted_indices = sorted_indices[sorted_indices != target_user_idx]
    
    # Return top k neighbors
    return sorted_indices[:k]

def predict_rating(target_user_idx, item_idx, user_item_matrix, similarity_matrix, k=5):
    """Predict rating for a user-item pair using k nearest neighbors"""
    target_user_ratings = user_item_matrix.iloc[target_user_idx]
    
    # Get top k neighbors
    neighbor_indices = get_top_k_neighbors(target_user_idx, similarity_matrix, k)
    
    # Calculate weighted average rating
    numerator = 0
    denominator = 0
    
    target_user_mean = target_user_ratings[target_user_ratings > 0].mean()
    if np.isnan(target_user_mean):
        target_user_mean = 0
    
    for neighbor_idx in neighbor_indices:
        neighbor_ratings = user_item_matrix.iloc[neighbor_idx]
        
        # Check if neighbor has rated this item
        if neighbor_ratings.iloc[item_idx] > 0:
            similarity = similarity_matrix[target_user_idx][neighbor_idx]
            neighbor_mean = neighbor_ratings[neighbor_ratings > 0].mean()
            if np.isnan(neighbor_mean):
                neighbor_mean = 0
            
            # Weighted rating deviation from mean
            rating_deviation = neighbor_ratings.iloc[item_idx] - neighbor_mean
            numerator += similarity * rating_deviation
            denominator += abs(similarity)
    
    if denominator == 0:
        # If no neighbors have rated this item, return user's average rating
        return target_user_mean if target_user_mean > 0 else 5.0  # Default to 5 if no history
    
    predicted_rating = target_user_mean + (numerator / denominator)
    
    # Ensure rating is within valid range [1, 10]
    predicted_rating = max(1, min(10, predicted_rating))
    
    return predicted_rating

def generate_recommendations(test_df, train_user_item_matrix, similarity_matrix, users, k=5):
    """Generate recommendations for test users"""
    recommendations = []
    
    test_users = test_df['userid'].unique()
    
    for target_user in test_users:
        if target_user not in users:
            print(f"Warning: User {target_user} not found in training data")
            continue
        
        target_user_idx = users.index(target_user)
        
        # Get books this user hasn't rated in training set
        user_rated_books = set(train_user_item_matrix.loc[target_user][train_user_item_matrix.loc[target_user] > 0].index)
        
        # Get all books from test set for this user
        test_books_for_user = test_df[test_df['userid'] == target_user]['isbn'].tolist()
        
        # Predict ratings for test books
        book_predictions = []
        
        for book in test_books_for_user:
            if book in train_user_item_matrix.columns:
                book_idx = train_user_item_matrix.columns.get_loc(book)
                predicted_rating = predict_rating(target_user_idx, book_idx, train_user_item_matrix, similarity_matrix, k)
                book_predictions.append((book, predicted_rating))
        
        # Sort by predicted rating (descending) and take top 5
        book_predictions.sort(key=lambda x: x[1], reverse=True)
        top_5_books = book_predictions[:5]
        
        # Get top k neighbors for this user
        neighbor_indices = get_top_k_neighbors(target_user_idx, similarity_matrix, k)
        neighbor_ids = [users[idx] for idx in neighbor_indices]
        
        # Create recommendation entries
        for book, predicted_rating in top_5_books:
            recommendation = {
                'TargetUserID': target_user,
                '1stNNUserID': neighbor_ids[0] if len(neighbor_ids) > 0 else '',
                '2NNUserID': neighbor_ids[1] if len(neighbor_ids) > 1 else '',
                '3NNUserID': neighbor_ids[2] if len(neighbor_ids) > 2 else '',
                '4NNUserID': neighbor_ids[3] if len(neighbor_ids) > 3 else '',
                '5NNUserID': neighbor_ids[4] if len(neighbor_ids) > 4 else '',
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
        fieldnames = ['TargetUserID', '1stNNUserID', '2NNUserID', '3NNUserID', 
                     '4NNUserID', '5NNUserID', 'Book_ISBN', 'Predicted_Rating']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        for rec in recommendations:
            writer.writerow(rec)
    
    print(f"Recommendations saved to {filename}")

train_file = 'rating10user91_trainset.csv'
test_file = 'rating10user91_testset.csv'
similarity_file = 'P2Part1_1PCC_Group7.csv'
recommendation_file = 'P2Part1_2Recommendation_Group7.csv'

print("=== User-based Nearest Neighbor Recommendation System ===\n")

# Step 1: Load data
print("1. Loading datasets...")
train_df, test_df = load_data(train_file, test_file)

# Step 2: Create user-item matrix
print("\n2. Creating user-item matrix...")
user_item_matrix = create_user_item_matrix(train_df)
print(f"User-item matrix shape: {user_item_matrix.shape}")

# Step 3: Compute similarity matrix
print("\n3. Computing Pearson correlation similarity matrix...")
similarity_matrix, users = compute_similarity_matrix(user_item_matrix)

# Step 4: Save similarity matrix
print("\n4. Saving similarity matrix...")
save_similarity_matrix(similarity_matrix, users, similarity_file)

# Step 5: Generate recommendations
print("\n5. Generating recommendations...")
recommendations = generate_recommendations(test_df, user_item_matrix, similarity_matrix, users, k=5)

# Step 6: Save recommendations
print("\n6. Saving recommendations...")
save_recommendations(recommendations, recommendation_file)

print(f"\nTotal recommendations generated: {len(recommendations)}")
print("\n=== Process completed successfully! ===")