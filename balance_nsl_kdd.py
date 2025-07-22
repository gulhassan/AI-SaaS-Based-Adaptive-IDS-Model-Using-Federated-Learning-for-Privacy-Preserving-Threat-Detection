import pandas as pd
import numpy as np

# Paths to your original NSL-KDD files
TRAIN_PATH = "/home/gul/nsl_kdd/KDDTrain+.txt"
TEST_PATH = "/home/gul/nsl_kdd/KDDTest+.txt"
FEATURE_NAMES_PATH = "/home/gul/nsl_kdd/KDDFeatureNames.txt"

# Load feature names
col_names = [line.strip() for line in open(FEATURE_NAMES_PATH).readlines()]

# Load training data
train_df = pd.read_csv(TRAIN_PATH, names=col_names)

# Print current class distribution
print("Original label distribution:\n", train_df["label"].value_counts())

# Choose your normal label (usually "normal" or 1, depending on your version)
normal_label = 1

# Create binary labels (0=normal, 1=attack)
train_df["binary_label"] = train_df["label"].apply(lambda x: 0 if x == normal_label else 1)

# Separate normal and attack
normal_df = train_df[train_df["binary_label"] == 0]
attack_df = train_df[train_df["binary_label"] == 1]

# Find smaller class size for balanced dataset
min_size = min(len(normal_df), len(attack_df))

# Downsample both classes to min_size
normal_sample = normal_df.sample(n=min_size, random_state=42)
attack_sample = attack_df.sample(n=min_size, random_state=42)

# Combine and shuffle
balanced_train_df = pd.concat([normal_sample, attack_sample]).sample(frac=1, random_state=42)

# Drop the binary label if you want to keep original labels
balanced_train_df = balanced_train_df.drop(columns=["binary_label"])

# Save the new balanced training data
balanced_train_df.to_csv("/home/gul/nsl_kdd/KDDTrain+_balanced.txt", header=False, index=False)

# Print balanced class distribution
print("Balanced label distribution:\n", balanced_train_df["label"].value_counts())
