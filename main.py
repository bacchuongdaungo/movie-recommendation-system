from surprise import Dataset
import pandas as pd
from surprise import Reader, SVD, accuracy
from surprise.model_selection import PredefinedKFold
import time
import numpy as np
# Step 1: Load Data
# Loading user data
users = pd.read_csv('ml-100k/u.user', sep='|', names=[
                    'user_id', 'age', 'gender', 'occupation', 'zip_code'],
                    encoding='ISO-8859-1')

# Loading item data
items = pd.read_csv('ml-100k/u.item', sep='|',
                    names=['item_id', 'title', 'release_date',
                           'video_release_date', 'IMDb_URL', 'unknown',
                           'Action', 'Adventure', 'Animation', "Children's",
                           'Comedy', 'Crime', 'Documentary', 'Drama',
                           'Fantasy', 'Film-Noir', 'Horror', 'Musical',
                           'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War',
                           'Western'],
                    encoding='ISO-8859-1')

# Loading ratings data
ratings = pd.read_csv('ml-100k/u.data', sep='\t',
                      names=['user_id', 'item_id', 'rating', 'timestamp'],
                      encoding='ISO-8859-1')

# Loading genre and occupation data (Optional for display purposes)
genres = pd.read_csv('ml-100k/u.genre', sep='|',
                     names=['genre', 'id'], encoding='ISO-8859-1')
occupations = pd.read_csv('ml-100k/u.occupation',
                          names=['occupation'], encoding='ISO-8859-1')

# Checking for missing values in each dataset
missing_users = users.isnull().sum()
missing_items = items.isnull().sum()
missing_ratings = ratings.isnull().sum()

# Check and handling missing data in 'release_date' by examining some rows
missing_release_dates = items[items['release_date'].isnull()]
# print(missing_items, missing_ratings, missing_users, missing_release_dates)
# Dropping unnecessary columns
items.drop(columns=['video_release_date', 'IMDb_URL', 'unknown'], inplace=True)

# Handling missing data
# Assuming 'release_date' should be handled if missing:
items['release_date'].fillna('Unknown', inplace=True)
# Removing rows with essential missing information
items.dropna(subset=['title'], inplace=True)

# Check missing data post-cleanup
# print(items.isnull().sum(), users.isnull().sum(), ratings.isnull().sum())

# Step 2: Prepare the dataset for Surprise
reader = Reader(line_format='user item rating timestamp',
                sep='\t', rating_scale=(1, 5))
# data = Dataset.load_from_df(ratings[['user_id', 'item_id', 'rating']], reader)

train_files = ['ml-100k/u1.base', 'ml-100k/u2.base', 'ml-100k/u3.base',
               'ml-100k/u4.base', 'ml-100k/u5.base']
test_files = ['ml-100k/u1.test', 'ml-100k/u2.test', 'ml-100k/u3.test',
              'ml-100k/u4.test', 'ml-100k/u5.test']


data = Dataset.load_from_folds(list(zip(train_files, test_files)),
                               reader=reader)

# Use PredefinedKFold for splitting
pkf = PredefinedKFold()

algo = SVD()

# Variables to store cumulative metrics
cumulative_rmse = 0
cumulative_mae = 0
fold_results = []

# Iterate over each train-test pair
for i, (trainset, testset) in enumerate(pkf.split(data), start=1):
    # Measure fit time
    start_fit = time.time()
    algo.fit(trainset)
    fit_time = time.time() - start_fit

    # Measure test time
    start_test = time.time()
    predictions = algo.test(testset)
    test_time = time.time() - start_test

    # Calculate RMSE and MAE
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)

    # Store results for final calculation
    fold_results.append((rmse, mae, fit_time, test_time))

    # Print the results for this fold
    print(f'Fold {i}:')
    print(f'  RMSE: {rmse:.4f}')
    print(f'  MAE: {mae:.4f}')
    print(f'  Fit time: {fit_time:.2f} seconds')
    print(f'  Test time: {test_time:.2f} seconds')

    # Update cumulative metrics
    cumulative_rmse += rmse
    cumulative_mae += mae

# Calculate mean and standard deviation of RMSE and MAE across folds
rmse_mean = np.mean([fr[0] for fr in fold_results])
rmse_std = np.std([fr[0] for fr in fold_results])
mae_mean = np.mean([fr[1] for fr in fold_results])
mae_std = np.std([fr[1] for fr in fold_results])

# Print final cross-validation results
print(f'\nFinal RMSE across all folds: {rmse_mean:.3f} ± {rmse_std:.3f}')
print(f'Final MAE across all folds: {mae_mean:.3f} ± {mae_std:.3f}')


# Define function to evaluate the dataset
def evaluate_dataset(trainset, testset, algo):
    # Measure fit time
    start_fit = time.time()
    algo.fit(trainset.build_full_trainset())
    fit_time = time.time() - start_fit

    # Extract original user ID, item ID, and rating
    testset_without_info = [(uid, iid, r_ui_trans) for (
        uid, iid, r_ui_trans, additional_info) in testset]

    # Measure test time
    start_test = time.time()
    predictions = algo.test(testset_without_info)
    test_time = time.time() - start_test

    # Calculate RMSE and MAE
    rmse = accuracy.rmse(predictions, verbose=True)
    mae = accuracy.mae(predictions, verbose=True)

    return rmse, mae, fit_time, test_time


# Prepare test sets correctly
testset_ua = [tuple(map(int, line.strip().split('\t')))
              for line in open('ml-100k/ua.test').readlines()]
testset_ub = [tuple(map(int, line.strip().split('\t')))
              for line in open('ml-100k/ub.test').readlines()]

# Load and prepare train sets
trainset_ua = Dataset.load_from_file('ml-100k/ua.base', reader=reader)
trainset_ub = Dataset.load_from_file('ml-100k/ub.base', reader=reader)

# Evaluate ua dataset
rmse_ua, mae_ua, fit_time_ua, test_time_ua = evaluate_dataset(
    trainset_ua, testset_ua, algo)
print(
    f'UA Dataset - RMSE: {rmse_ua:.3f}, MAE: {mae_ua:.3f}, Fit Time: {fit_time_ua:.2f} s, Test Time: {test_time_ua:.2f} s')

# Evaluate ub dataset
rmse_ub, mae_ub, fit_time_ub, test_time_ub = evaluate_dataset(
    trainset_ub, testset_ub, algo)
print(
    f'UB Dataset - RMSE: {rmse_ub:.3f}, MAE: {mae_ub:.3f}, Fit Time: {fit_time_ub:.2f} s, Test Time: {test_time_ub:.2f} s')
