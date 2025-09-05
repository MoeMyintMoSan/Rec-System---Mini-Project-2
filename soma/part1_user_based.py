import csv
import math

def load_csv_data(filename):
    """Load CSV data into list of dictionaries"""
    data = []
    with open(filename, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append({
                'userid': int(row['userid']),
                'isbn': row['isbn'],
                'rating': float(row['rating'])
            })
    return data

def get_unique_users(data):
    """Get list of unique users"""
    users = set()
    for row in data:
        users.add(row['userid'])
    return sorted(list(users))

def get_unique_items(data):
    """Get list of unique items (books)"""
    items = set()
    for row in data:
        items.add(row['isbn'])
    return sorted(list(items))

def create_user_item_matrix(data, users, items):
    """Create user-item matrix as dictionary of dictionaries"""
    matrix = {}
    
    # Initialize matrix with zeros
    for user in users:
        matrix[user] = {}
        for item in items:
            matrix[user][item] = 0
    
    # Fill in actual ratings
    for row in data:
        matrix[row['userid']][row['isbn']] = row['rating']
    
    return matrix

def calculate_mean(values):
    """Calculate mean of a list of values"""
    return sum(values) / len(values) if values else 0

def calculate_pearson_correlation(ratings1, ratings2):
    """Calculate Pearson correlation coefficient between two rating lists"""
    # Find common items (both users have rated)
    common_items = []
    user1_common = []
    user2_common = []
    
    for i in range(len(ratings1)):
        if ratings1[i] > 0 and ratings2[i] > 0:
            common_items.append(i)
            user1_common.append(ratings1[i])
            user2_common.append(ratings2[i])
    
    if len(user1_common) < 2:  # Need at least 2 common items
        return 0
    
    # Calculate means
    mean1 = calculate_mean(user1_common)
    mean2 = calculate_mean(user2_common)
    
    # Calculate numerator and denominators
    numerator = 0
    sum_sq1 = 0
    sum_sq2 = 0
    
    for i in range(len(user1_common)):
        diff1 = user1_common[i] - mean1
        diff2 = user2_common[i] - mean2
        
        numerator += diff1 * diff2
        sum_sq1 += diff1 * diff1
        sum_sq2 += diff2 * diff2
    
    # Calculate correlation
    denominator = math.sqrt(sum_sq1 * sum_sq2)
    
    if denominator == 0:
        return 0
    
    correlation = numerator / denominator
    
    # Handle NaN cases
    if math.isnan(correlation):
        return 0
    
    return correlation

def build_similarity_matrix(user_item_matrix, users, items):
    """Build user-user similarity matrix using Pearson correlation"""
    print("Building similarity matrix...")
    
    n_users = len(users)
    similarity_matrix = {}
    
    # Initialize similarity matrix
    for user1 in users:
        similarity_matrix[user1] = {}
        for user2 in users:
            similarity_matrix[user1][user2] = 0
    
    for i, user1 in enumerate(users):
        for j, user2 in enumerate(users):
            if i <= j:  # Only calculate upper triangle and diagonal
                if user1 == user2:
                    similarity = 1.0
                else:
                    # Get user rating vectors
                    user1_ratings = [user_item_matrix[user1][item] for item in items]
                    user2_ratings = [user_item_matrix[user2][item] for item in items]
                    
                    similarity = calculate_pearson_correlation(user1_ratings, user2_ratings)
                
                similarity_matrix[user1][user2] = similarity
                similarity_matrix[user2][user1] = similarity  # Matrix is symmetric
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{n_users} users")
    
    print("Similarity matrix completed!")
    return similarity_matrix

def get_k_nearest_neighbors(target_user, similarity_matrix, users, k=5):
    """Get k most similar users for target user"""
    user_similarities = []
    
    for user in users:
        if user != target_user:
            similarity = similarity_matrix[target_user][user]
            user_similarities.append((user, similarity))
    
    # Sort by similarity (descending) and take top k
    user_similarities.sort(key=lambda x: x[1], reverse=True)
    
    return [user for user, sim in user_similarities[:k]]

def predict_rating(target_user, item, user_item_matrix, similarity_matrix, users, k=5):
    """Predict rating for a user-item pair using k-NN"""
    # Get k nearest neighbors
    neighbor_users = get_k_nearest_neighbors(target_user, similarity_matrix, users, k)
    
    numerator = 0
    denominator = 0
    
    for neighbor_user in neighbor_users:
        # Get neighbor's rating for this item
        neighbor_rating = user_item_matrix[neighbor_user][item]
        
        if neighbor_rating > 0:  # Neighbor has rated this item
            similarity = similarity_matrix[target_user][neighbor_user]
            
            if similarity > 0:  # Only consider positive similarities
                numerator += similarity * neighbor_rating
                denominator += abs(similarity)
    
    if denominator == 0:
        # If no similar users have rated this item, return average rating
        all_ratings = []
        for user in users:
            for itm in user_item_matrix[user]:
                if user_item_matrix[user][itm] > 0:
                    all_ratings.append(user_item_matrix[user][itm])
        return calculate_mean(all_ratings) if all_ratings else 5.0
    
    predicted_rating = numerator / denominator
    
    # Ensure rating is within valid range [1, 10]
    return max(1, min(10, predicted_rating))

def generate_recommendations(train_data, test_data, user_item_matrix, similarity_matrix, users, items, group_no="7"):
    """Generate recommendations for all users in test set"""
    print("Generating recommendations...")
    
    recommendations = []
    test_users = set(row['userid'] for row in test_data)
    
    for target_user in test_users:
        print(f"Processing user {target_user}...")
        
        # Check if user exists in training data
        if target_user not in users:
            print(f"Warning: User {target_user} not in training data")
            continue
        
        # Get k nearest neighbors
        neighbor_users = get_k_nearest_neighbors(target_user, similarity_matrix, users, k=5)
        
        # Get books this user needs to rate (from test set)
        user_test_books = set(row['isbn'] for row in test_data if row['userid'] == target_user)
        
        # Calculate predictions for each book
        book_predictions = []
        
        for book in user_test_books:
            if book in items:
                predicted_rating = predict_rating(target_user, book, user_item_matrix, similarity_matrix, users, k=5)
                book_predictions.append((book, predicted_rating))
        
        # Sort by predicted rating and take top 5
        book_predictions.sort(key=lambda x: x[1], reverse=True)
        top_5_books = book_predictions[:5]
        
        # Create output rows
        for book, pred_rating in top_5_books:
            row = {
                'TargetUserID': target_user,
                '1stNNUserID': neighbor_users[0] if len(neighbor_users) > 0 else '',
                '2ndNNUserID': neighbor_users[1] if len(neighbor_users) > 1 else '',
                '3rdNNUserID': neighbor_users[2] if len(neighbor_users) > 2 else '',
                '4thNNUserID': neighbor_users[3] if len(neighbor_users) > 3 else '',
                '5thNNUserID': neighbor_users[4] if len(neighbor_users) > 4 else '',
                'BookISBN': book,
                'PredictedRating': round(pred_rating, 2)
            }
            recommendations.append(row)
    
    return recommendations

def save_csv(data, filename, fieldnames):
    """Save data to CSV file"""
    with open(filename, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def main():
    # Load data
    print("Loading datasets...")
    train_data = load_csv_data('rating10user91_trainset.csv')
    test_data = load_csv_data('rating10user91_testset.csv')
    
    print(f"Training data: {len(train_data)} records")
    print(f"Test data: {len(test_data)} records")
    
    # Get unique users and items
    users = get_unique_users(train_data)
    items = get_unique_items(train_data)
    
    print(f"Unique users: {len(users)}")
    print(f"Unique books: {len(items)}")
    
    # Create user-item matrix
    print("Creating user-item matrix...")
    user_item_matrix = create_user_item_matrix(train_data, users, items)
    
    # Build similarity matrix
    similarity_matrix = build_similarity_matrix(user_item_matrix, users, items)
    
    # Save similarity matrix (PCC) to CSV
    print("Saving Pearson Correlation Coefficient matrix...")
    pcc_data = []
    for user1 in users:
        row = {'UserID': user1}
        for user2 in users:
            row[f'User_{user2}'] = round(similarity_matrix[user1][user2], 4)
        pcc_data.append(row)
    
    pcc_fieldnames = ['UserID'] + [f'User_{user}' for user in users]
    save_csv(pcc_data, "P2Part1_1PCC_Group7.csv", pcc_fieldnames)
    print("PCC matrix saved to P2Part1_1PCC_Group7.csv")
    
    # Also create a readable sample of the similarity matrix (first 10 users)
    print("Creating readable similarity matrix sample...")
    sample_users = users[:10]  # First 10 users for sample
    sample_similarity = []
    
    for user1 in sample_users:
        row = {'UserID': user1}
        for user2 in sample_users:
            row[f'User_{user2}'] = round(similarity_matrix[user1][user2], 4)
        sample_similarity.append(row)
    
    sample_fieldnames = ['UserID'] + [f'User_{user}' for user in sample_users]
    save_csv(sample_similarity, "P2Part1_SimilarityMatrix_Sample_Group7.csv", sample_fieldnames)
    print("Sample similarity matrix saved to P2Part1_SimilarityMatrix_Sample_Group7.csv")
    
    # Generate recommendations (replace "XX" with your group number)
    recommendations = generate_recommendations(train_data, test_data, user_item_matrix, similarity_matrix, users, items, group_no="7")
    
    # Save to CSV
    output_file = "P2Part1_2Recommendation_Group7.csv"
    fieldnames = ['TargetUserID', '1stNNUserID', '2ndNNUserID', '3rdNNUserID', '4thNNUserID', '5thNNUserID', 'BookISBN', 'PredictedRating']
    save_csv(recommendations, output_file, fieldnames)
    
    print(f"Recommendations saved to {output_file}")
    print(f"Total recommendations: {len(recommendations)}")
    
    # Display sample results
    print("\nSample recommendations:")
    for i, rec in enumerate(recommendations[:10]):
        neighbors = [rec['1stNNUserID'], rec['2ndNNUserID'], rec['3rdNNUserID'], rec['4thNNUserID'], rec['5thNNUserID']]
        neighbors = [n for n in neighbors if n != '']
        print(f"{i+1}. User {rec['TargetUserID']}: Book {rec['BookISBN']} - Rating {rec['PredictedRating']} (Neighbors: {neighbors})")

if __name__ == "__main__":
    main()
