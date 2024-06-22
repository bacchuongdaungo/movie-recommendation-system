from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import GridSearchCV, train_test_split
import pandas as pd
import time

# Step 1: Load Data
users = pd.read_csv('ml-100k/u.user', sep='|', names=['user_id', 'age', 'gender', 'occupation', 'zip_code'], encoding='ISO-8859-1')
items = pd.read_csv('ml-100k/u.item', sep='|', names=['item_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'], encoding='ISO-8859-1')
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'], encoding='ISO-8859-1')

# Prepare the data
items.drop(columns=['video_release_date', 'IMDb_URL', 'unknown'], inplace=True)
items['release_date'].fillna('Unknown', inplace=True)

reader = Reader(line_format='user item rating timestamp', sep='\t', rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['user_id', 'item_id', 'rating']], reader)

# Step 2: Set up the parameter grid for SVD
param_grid = {
    'n_factors': [50, 100, 150],  # number of factors
    'n_epochs': [20, 30],         # number of iterations
    'lr_all': [0.005, 0.01],      # learning rate
    'reg_all': [0.02, 0.1]        # regularization term
}

# Step 3: Run grid search
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=5)
gs.fit(data)

# Step 4: Best RMSE score and parameters
print(f"Best RMSE score obtained: {gs.best_score['rmse']}")
print(f"Best parameters: {gs.best_params['rmse']}")

# Use the best algorithm
algo = gs.best_estimator['rmse']

# Re-train on the full dataset
trainset = data.build_full_trainset()
algo.fit(trainset)

# Function to evaluate the dataset
def evaluate_model(testset, algo):
    start_test = time.time()
    predictions = algo.test(testset)
    test_time = time.time() - start_test
    rmse = accuracy.rmse(predictions, verbose=True)
    mae = accuracy.mae(predictions, verbose=True)
    return rmse, mae, test_time

# Prepare and evaluate test sets
def read_test_set(file_path):
    test_data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 4:
                test_data.append(tuple(map(int, parts[:3])))
    return test_data

testset_ua = read_test_set('ml-100k/ua.test')
testset_ub = read_test_set('ml-100k/ub.test')

rmse_ua, mae_ua, test_time_ua = evaluate_model(testset_ua, algo)
print(f'UA Dataset - RMSE: {rmse_ua:.3f}, MAE: {mae_ua:.3f}, Test Time: {test_time_ua:.2f} s')

rmse_ub, mae_ub, test_time_ub = evaluate_model(testset_ub, algo)
print(f'UB Dataset - RMSE: {rmse_ub:.3f}, MAE: {mae_ub:.3f}, Test Time: {test_time_ub:.2f} s')
