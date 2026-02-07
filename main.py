from data_analysis import load_and_inspect, clean_data
from feature_engineering import create_label, select_features
from model_training import train_and_evaluate

# Define your dataset filename here
# Make sure this matches the actual name of your uploaded CSV file
dataset_file = 'Sleep_health_and_lifestyle_dataset.csv'

# 1. Load & Inspect
df = load_and_inspect(dataset_file)

# 2. Clean
df = clean_data(df)

# 3. Create Label
df = create_label(df)

# 4. Feature Selection
# This function returns the features (X), the label (y), and the list of feature names
X, y, feature_names = select_features(df)

# 5-9. Train, Evaluate, Visualize
train_and_evaluate(X, y, feature_names)
