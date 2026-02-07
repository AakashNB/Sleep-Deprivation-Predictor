
This project is a machine learning pipeline designed to predict whether a person is sleep deprived (getting less than 6 hours of sleep) based on their lifestyle and health metrics.

The system uses Logistic Regression and Random Forest models to analyze factors such as Age, Physical Activity Level, Stress Level, Heart Rate, and Daily Steps.

Code Structure
data_analysis.py This file acts as the inspector for the project. It handles loading the raw CSV data into a pandas DataFrame. It also performs data cleaning, which includes removing rows with missing values and dropping columns that are not useful for prediction, such as Person ID.

feature_engineering.py This file prepares the data for the machine learning models. It performs two main tasks:

Creating the Label: It generates a new column called Sleep_Deprived. If a person sleeps less than 6 hours, they are marked as 1; otherwise, they are marked as 0.

Feature Selection: It selects the specific numeric columns (like Stress Level and Heart Rate) that the model will use to make predictions, while removing the answer column to prevent cheating.

model_training.py This file contains the core machine learning logic. It splits the data into a training set (80%) and a testing set (20%). It then initializes and trains two models: Logistic Regression and Random Forest. Finally, it evaluates their performance using Accuracy, F1 Score, and ROC-AUC, and generates visualization plots.

main.py This is the main execution script. It imports the functions from the three files above and runs them in the correct order. This is the only file you need to run to execute the entire project.

requirements.txt This file lists all the external Python libraries required to run this project, such as pandas, scikit-learn, and matplotlib.

Setup Instructions
1. Install Dependencies Open your terminal or command prompt, navigate to the project folder, and run the following command to install the necessary libraries:

Bash
pip install -r requirements.txt
2. Configure the Dataset Ensure your dataset file (CSV) is located in the same folder as the Python scripts. Open main.py and confirm that the filename variable matches the exact name of your CSV file.

3. Run the Project To execute the full pipeline, run the main script:

