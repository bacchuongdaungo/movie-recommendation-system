Recommendation System for MovieLens Dataset
This project includes two versions of a recommendation system using the MovieLens dataset:

-main.py - The standard version using the Surprise library's SVD algorithm.
-hyperparameter.py - The improved version that includes hyperparameter tuning of the SVD algorithm.

Prerequisites:
Before running the project, ensure you have the following installed:
-Python 3.6 or higher
-Pandas library
-NumPy library
-Surprise library
-Installation

Install the required Python libraries using pip:

pip install pandas numpy scikit-surprise
Running the Code:
-Standard Version: To run the standard version of the recommendation system, navigate to the project directory and execute the following command "python main.py".
This script will load the MovieLens dataset, perform basic data preparation, train the SVD model, and output the RMSE and MAE of the model.

-Improved Version with Hyperparameter Tuning: To run the improved version with hyperparameter tuning, use the following command "python hyperparameter.py".
This script will perform a grid search to find the best hyperparameters for the SVD model and will output the best RMSE score, the best parameters, and the evaluation metrics on the test set.

Outputs:
Both scripts will display the model's performance in terms of RMSE (Root Mean Square Error) and MAE (Mean Absolute Error). Additionally, the improved version will report the best hyperparameters found during the grid search.

Additional Information:
Ensure that the MovieLens dataset files are located in the ml-100k directory within the project directory, as the scripts expect this file structure for loading the data.

