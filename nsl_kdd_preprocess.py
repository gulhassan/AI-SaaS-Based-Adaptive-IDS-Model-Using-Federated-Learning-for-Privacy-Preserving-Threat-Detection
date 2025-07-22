import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import numpy as np

def load_nsl_kdd(train_path, test_path, client_id=0, num_clients=3):
    # Load column names
    col_names = [line.strip() for line in open("/home/gul/nsl_kdd/KDDFeatureNames.txt").readlines()]
    train_df = pd.read_csv(train_path, names=col_names)
    test_df = pd.read_csv(test_path, names=col_names)

    # Drop duplicates
    train_df.drop_duplicates(inplace=True)

    # === SAVE ORIGINAL LABELS (strings) ===
    original_labels = train_df["label"].values  # Save before modification
    attack_names = sorted(train_df["label"].unique(), key=lambda x: str(x))

    # Print the mapping: name to index
    print("Attack label mapping (index : name):")
    for idx, name in enumerate(attack_names):
        print(f"{idx}: {name}")

    # Build a mapping (dict) for future use
    attack_label_mapping = {idx: name for idx, name in enumerate(attack_names)}
    attack_name_to_idx = {name: idx for idx, name in enumerate(attack_names)}

    # Optionally encode labels as integer indices for analysis/plotting
    train_df["attack_label_idx"] = train_df["label"].map(attack_name_to_idx)
    original_label_indices = train_df["attack_label_idx"].values  # integer-encoded original labels

    # Convert original multiclass labels to binary: 0 = normal (label == 1), 1 = attack
    normal_label = 1
    train_df["label"] = train_df["label"].apply(lambda x: 0 if x == normal_label else 1)
    test_df["label"] = test_df["label"].apply(lambda x: 0 if x == normal_label else 1)

    print("Binary label distribution after conversion:\n", train_df["label"].value_counts())

    # Encode object (categorical) columns
    encoders = {}
    for col in train_df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])
        encoders[col] = le

        test_col = test_df[col].map(lambda s: "<UNK>" if s not in le.classes_ else s)
        le_classes = np.append(le.classes_, "<UNK>")
        le.classes_ = le_classes
        test_df[col] = le.transform(test_col)

    # Stratified split across clients
    X = train_df.drop(["label", "attack_label_idx"], axis=1).values
    y = train_df["label"].values

    skf = StratifiedKFold(n_splits=num_clients, shuffle=True, random_state=42)
    splits = list(skf.split(X, y))

    # Get the partition for this client
    train_idx, _ = splits[client_id]
    X_train = X[train_idx]
    y_train = y[train_idx]
    # For plotting multi-class distribution (by integer index)
    client_original_labels = original_label_indices[train_idx]

    # Use full test set (optional: also split this if needed)
    X_test = test_df.drop("label", axis=1).values
    y_test = test_df["label"].values

    print(f"Client {client_id} sample count: {len(X_train)}")
    print(f"Client {client_id} class breakdown: {np.bincount(y_train)}")

    # Return the label mapping as well (for use in plots)
    return X_train, X_test, y_train, y_test, client_original_labels, attack_label_mapping

# Optional: test run
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, client_original_labels, attack_label_mapping = load_nsl_kdd(
        "/home/gul/nsl_kdd/KDDTrain+_balanced.txt",
        "/home/gul/nsl_kdd/KDDTest+.txt",
        client_id=0,
        num_clients=3
    )
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    # Example: print the mapping for your plots
    print(attack_label_mapping)

