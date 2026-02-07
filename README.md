Sleep Deprivation Predictor is a machine learning project that predicts if a person is sleep deprived (like less than 6 hours of sleep) based on lifestyle factors like stress, physical activity, and age.

Project Structure:

data_analysis.py: Loads the raw dataset, inspects columns/types, and handles missing values.

feature_engineering.py: Creates the target label (Sleep_Deprived) and selects relevant numeric features for training.

model_training.py: Splits data (80/20), trains Logistic Regression and Random Forest models, and evaluates performance (Accuracy, ROC-AUC, Confusion Matrix).

main.py: The entry point that ties everything together and runs the full pipeline.

requirements.txt: List of Python libraries required to run the project.

Setup & Installation
1. Navigate to the project folder Open your terminal and run:

cd ~/Downloads/Sleep-Deprivation-Tracker

2. Install dependencies Install the necessary Python libraries:

pip3 install pandas matplotlib seaborn scikit-learn

3. Run the Project Execute the main script to train the models and see results:

python3 main.py
