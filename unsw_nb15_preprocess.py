import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle

def load_unsw_nb15(train_path, test_path, client_id=0, num_clients=3):
    # Load datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Shuffle data
    train_df = shuffle(train_df, random_state=42)
    test_df = shuffle(test_df, random_state=42)

    # Identify categorical columns (edit as per actual columns)
    cat_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    if 'id' in train_df.columns:
        train_df = train_df.drop('id', axis=1)
    if 'id' in test_df.columns:
        test_df = test_df.drop('id', axis=1)

    # Label column (usually 'label' or 'attack_cat')
    if 'label' in train_df.columns:
        label_col = 'label'
    elif 'attack_cat' in train_df.columns:
        label_col = 'attack_cat'
    else:
        raise Exception("No label/attack_cat column found!")

    # Save original multiclass labels for analysis
    original_labels = train_df[label_col].values
    attack_names = sorted(train_df[label_col].unique(), key=lambda x: str(x))
    print("Attack label mapping (index : name):")
    for idx, name in enumerate(attack_names):
        print(f"{idx}: {name}")
    attack_label_mapping = {idx: name for idx, name in enumerate(attack_names)}
    attack_name_to_idx = {name: idx for idx, name in enumerate(attack_names)}
    train_df['attack_label_idx'] = train_df[label_col].map(attack_name_to_idx)
    original_label_indices = train_df['attack_label_idx'].values

    # Convert to binary: 0 = normal, 1 = attack (edit if your label is different)
    # In UNSW-NB15, label column is 0 (normal) and 1 (attack)
    train_df['label'] = train_df['label'].apply(lambda x: 0 if x == 0 else 1)
    test_df['label'] = test_df['label'].apply(lambda x: 0 if x == 0 else 1)

    # Encode categorical columns
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        encoders[col] = le
        test_df[col] = test_df[col].map(lambda s: "<UNK>" if s not in le.classes_ else s)
        le_classes = np.append(le.classes_, "<UNK>")
        le.classes_ = le_classes
        test_df[col] = le.transform(test_df[col].astype(str))

    # Stratified split for clients
    X = train_df.drop(['label', 'attack_label_idx'], axis=1).values
    y = train_df['label'].values
    skf = StratifiedKFold(n_splits=num_clients, shuffle=True, random_state=42)
    splits = list(skf.split(X, y))
    train_idx, _ = splits[client_id]
    X_train = X[train_idx]
    y_train = y[train_idx]
    client_original_labels = original_label_indices[train_idx]

    # Prepare test set
    X_test = test_df.drop(['label'], axis=1).values
    y_test = test_df['label'].values

    print(f"Client {client_id} sample count: {len(X_train)}")
    print(f"Client {client_id} class breakdown: {np.bincount(y_train)}")

    return X_train, X_test, y_train, y_test, client_original_labels, attack_label_mapping

# Example standalone run:
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, client_original_labels, attack_label_mapping = load_unsw_nb15(
        "/home/gul/USNW/UNSW_NB15_training-set.csv",
        "/home/gul/USNW/UNSW_NB15_testing-set.csv", 
        client_id=0, num_clients=3)
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
