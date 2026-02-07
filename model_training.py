from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def train_and_evaluate(X, y, feature_names):
    # --- Step 5: Train/Test Split ---
    print("\n--- Step 5: Train/Test Split ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

    # --- Step 6: Train Models ---
    print("\n--- Step 6: Train Models ---")
    # Model 1: Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    
    # Model 2: Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    models = {'Logistic Regression': lr, 'Random Forest': rf}
    
    # --- Step 7: Evaluation ---
    print("\n--- Step 7: Evaluation ---")
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)
        
        print(f"\nModel: {name}")
        print(f"Accuracy: {acc:.2f}")
        print(f"F1 Score: {f1:.2f}")
        print(f"ROC-AUC: {roc:.2f}")
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # --- Step 8: Feature Importance ---
    print("\n--- Step 8: Feature Importance (Random Forest) ---")
    importances = rf.feature_importances_
    feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)
    print(feature_imp_df)
    
    # --- Step 9: Visualization ---
    # Plot Feature Importance
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Importance', y='Feature', data=feature_imp_df, palette='viridis')
    plt.title('Feature Importance (Random Forest)')
    plt.tight_layout()
    plt.show() # In a notebook this shows inline. In script, it might pop up.
    
    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        plt.plot(fpr, tpr, label=name)
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
