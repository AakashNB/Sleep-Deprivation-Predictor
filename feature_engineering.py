def create_label(df):
    print("\n--- Step 3: Create Label ---")
    # Sleep Deprivation = Sleep Duration < 6 hours
    # 1 = Deprived, 0 = Healthy
    df['Sleep_Deprived'] = (df['Sleep Duration'] < 6).astype(int)
    
    print("Label Distribution:")
    print(df['Sleep_Deprived'].value_counts())
    return df

def select_features(df):
    print("\n--- Step 4: Feature Selection ---")
    # We choose numeric features that likely impact sleep
    # We EXCLUDE 'Sleep Duration' (because that IS the label) and 'Sleep Disorder' (too easy/outcome)
    features = ['Age', 'Quality of Sleep', 'Physical Activity Level',
                'Stress Level', 'Heart Rate', 'Daily Steps']
    
    # Check if these columns exist
    available_features = [f for f in features if f in df.columns]
    
    print(f"Selected Features: {available_features}")
    return df[available_features], df['Sleep_Deprived'], available_features
