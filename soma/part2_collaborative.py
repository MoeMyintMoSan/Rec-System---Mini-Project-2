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

def calculate_cosine_similarity(vector1, vector2):
    """Calculate cosine similarity between two vectors"""
    dot_product = 0
    norm1 = 0
    norm2 = 0
    
    for i in range(len(vector1)):
        dot_product += vector1[i] * vector2[i]
        norm1 += vector1[i] * vector1[i]
        norm2 += vector2[i] * vector2[i]
    
    if norm1 == 0 or norm2 == 0:
        return 0
    
    return dot_product / (math.sqrt(norm1) * math.sqrt(norm2))

def build_item_similarity_matrix(user_item_matrix, users, items):
    """Build item-item similarity matrix using cosine similarity"""
    print("Building item-item similarity matrix...")
    
    similarity_matrix = {}
    
    # Initialize similarity matrix
    for item1 in items:
        similarity_matrix[item1] = {}
        for item2 in items:
            similarity_matrix[item1][item2] = 0
    
    # Calculate similarities
    for i, item1 in enumerate(items):
        for j, item2 in enumerate(items):
            if i <= j:  # Only calculate upper triangle and diagonal
                if item1 == item2:
                    similarity = 1.0
                else:
                    # Get item vectors (ratings by all users)
                    vector1 = [user_item_matrix[user][item1] for user in users]
                    vector2 = [user_item_matrix[user][item2] for user in users]
                    
                    similarity = calculate_cosine_similarity(vector1, vector2)
                
                similarity_matrix[item1][item2] = similarity
                similarity_matrix[item2][item1] = similarity  # Matrix is symmetric
    
    print("Item similarity matrix completed!")
    return similarity_matrix

def predict_rating_item_based(user_id, item_id, user_item_matrix, item_similarity_matrix, users, items, k=10):
    """Predict rating using item-based collaborative filtering"""
    if user_id not in users or item_id not in items:
        # Return average rating as default
        total_ratings = 0
        count = 0
        for user in users:
            for item in items:
                if user_item_matrix[user][item] > 0:
                    total_ratings += user_item_matrix[user][item]
                    count += 1
        return total_ratings / count if count > 0 else 5.0
    
    # Find items that the user has rated
    rated_items = []
    for item in items:
        if user_item_matrix[user_id][item] > 0:
            rated_items.append(item)
    
    if not rated_items:
        return 5.0  # Default rating
    
    # Get similarities between target item and rated items
    item_similarities = []
    for rated_item in rated_items:
        similarity = item_similarity_matrix[item_id][rated_item]
        item_similarities.append((rated_item, similarity))
    
    # Sort by similarity and take top k
    item_similarities.sort(key=lambda x: x[1], reverse=True)
    top_k_items = item_similarities[:k]
    
    # Calculate weighted average
    numerator = 0
    denominator = 0
    
    for similar_item, similarity in top_k_items:
        if similarity > 0:
            user_rating = user_item_matrix[user_id][similar_item]
            numerator += similarity * user_rating
            denominator += similarity
    
    if denominator == 0:
        return 5.0  # Default rating
    
    predicted_rating = numerator / denominator
    return max(1, min(10, predicted_rating))

def create_user_profiles(data, users):
    """Create user profiles with statistical information"""
    print("Creating user profiles...")
    
    profiles = []
    for user in users:
        user_ratings = [row['rating'] for row in data if row['userid'] == user]
        
        if user_ratings:
            avg_rating = sum(user_ratings) / len(user_ratings)
            min_rating = min(user_ratings)
            max_rating = max(user_ratings)
            
            # Calculate standard deviation
            variance = sum((x - avg_rating) ** 2 for x in user_ratings) / len(user_ratings)
            std_rating = math.sqrt(variance)
        else:
            avg_rating = 0
            min_rating = 0
            max_rating = 0
            std_rating = 0
        
        profile = {
            'UserID': user,
            'TotalRatings': len(user_ratings),
            'AvgRating': round(avg_rating, 2),
            'StdRating': round(std_rating, 2),
            'MinRating': min_rating,
            'MaxRating': max_rating,
            'RatedBooks': len(user_ratings)
        }
        profiles.append(profile)
    
    return profiles

def generate_recommendations(test_data, user_item_matrix, item_similarity_matrix, users, items):
    """Generate top-10 recommendations for each user in test set"""
    print("Generating recommendations...")
    
    recommendations = []
    test_users = set(row['userid'] for row in test_data)
    
    for target_user in test_users:
        print(f"Processing user {target_user}...")
        
        if target_user not in users:
            print(f"Warning: User {target_user} not in training data")
            continue
        
        # Get books this user needs to rate (from test set)
        user_test_books = set(row['isbn'] for row in test_data if row['userid'] == target_user)
        
        # Calculate predictions for each book
        book_predictions = []
        
        for book in user_test_books:
            if book in items:
                predicted_rating = predict_rating_item_based(
                    target_user, book, user_item_matrix, item_similarity_matrix, users, items
                )
                book_predictions.append((book, predicted_rating))
        
        # Sort by predicted rating and take top 10
        book_predictions.sort(key=lambda x: x[1], reverse=True)
        top_10_books = book_predictions[:10]
        
        # Create output rows
        for rank, (book, pred_rating) in enumerate(top_10_books, 1):
            row = {
                'UserID': target_user,
                'BookISBN': book,
                'ModelCalculatedValue': pred_rating,
                'PredictedRating': round(pred_rating, 2),
                'Rank': rank
            }
            recommendations.append(row)
    
    return recommendations

def calculate_rmse(test_data, user_item_matrix, item_similarity_matrix, users, items):
    """Calculate RMSE for all predictions in test set"""
    print("Calculating RMSE...")
    
    actual_ratings = []
    predicted_ratings = []
    
    for row in test_data:
        user_id = row['userid']
        item_id = row['isbn']
        actual_rating = row['rating']
        
        predicted_rating = predict_rating_item_based(
            user_id, item_id, user_item_matrix, item_similarity_matrix, users, items
        )
        
        actual_ratings.append(actual_rating)
        predicted_ratings.append(predicted_rating)
    
    # Calculate RMSE
    mse = sum((actual - predicted) ** 2 for actual, predicted in zip(actual_ratings, predicted_ratings)) / len(actual_ratings)
    rmse = math.sqrt(mse)
    
    print(f"RMSE: {rmse:.4f}")
    
    return rmse, actual_ratings, predicted_ratings

def save_csv(data, filename, fieldnames):
    """Save data to CSV file"""
    with open(filename, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def save_all_outputs(train_data, test_data, user_item_matrix, item_similarity_matrix, users, items, group_no="7"):
    """Save all required output files"""
    
    # 1. Save user profiles
    print("Saving user profiles...")
    user_profiles = create_user_profiles(train_data, users)
    profiles_file = f"P2Part2_1Profile_Group{7}.csv"
    save_csv(user_profiles, profiles_file, ['UserID', 'TotalRatings', 'AvgRating', 'StdRating', 'MinRating', 'MaxRating', 'RatedBooks'])
    print(f"User profiles saved to {profiles_file}")
    
    # 2. Save similarity matrix (sample - top 10 similar items for first 50 items)
    print("Saving similarity matrix sample...")
    model_file = f"P2Part2_2Model_Group{7}.csv"
    similarity_sample = []
    
    for item in items[:50]:  # Sample first 50 items
        # Get similarities for this item
        similarities = [(other_item, item_similarity_matrix[item][other_item]) 
                       for other_item in items if other_item != item]
        # Sort and take top 10
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_10_similar = similarities[:10]
        
        for similar_item, similarity in top_10_similar:
            similarity_sample.append({
                'TargetItem': item,
                'SimilarItem': similar_item,
                'Similarity': round(similarity, 4)
            })
    
    save_csv(similarity_sample, model_file, ['TargetItem', 'SimilarItem', 'Similarity'])
    print(f"Model/Similarity data saved to {model_file}")
    
    # 3. Save recommendations
    print("Saving recommendations...")
    recommendations = generate_recommendations(test_data, user_item_matrix, item_similarity_matrix, users, items)
    recommendations_file = f"P2Part2_3Recommendation_Group{7}.csv"
    save_csv(recommendations, recommendations_file, ['UserID', 'BookISBN', 'ModelCalculatedValue', 'PredictedRating', 'Rank'])
    print(f"Recommendations saved to {recommendations_file}")
    
    # 4. Calculate and save RMSE
    print("Saving RMSE results...")
    rmse, actual, predicted = calculate_rmse(test_data, user_item_matrix, item_similarity_matrix, users, items)
    rmse_data = [{
        'RMSE': round(rmse, 4),
        'TotalPredictions': len(actual),
        'MeanActualRating': round(sum(actual) / len(actual), 2),
        'MeanPredictedRating': round(sum(predicted) / len(predicted), 2)
    }]
    
    rmse_file = f"P2Part2_4RMSE_Group{7}.csv"
    save_csv(rmse_data, rmse_file, ['RMSE', 'TotalPredictions', 'MeanActualRating', 'MeanPredictedRating'])
    print(f"RMSE results saved to {rmse_file}")
    
    return recommendations

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
    
    # Build item similarity matrix
    item_similarity_matrix = build_item_similarity_matrix(user_item_matrix, users, items)
    
    # Save all outputs (replace "XX" with your group number)
    recommendations = save_all_outputs(train_data, test_data, user_item_matrix, item_similarity_matrix, users, items, group_no="7")
    
    # Display sample results
    print("\nSample recommendations:")
    for i, rec in enumerate(recommendations[:10]):
        print(f"{i+1}. User {rec['UserID']}: Book {rec['BookISBN']} - Rating {rec['PredictedRating']}")

if __name__ == "__main__":
    main()
