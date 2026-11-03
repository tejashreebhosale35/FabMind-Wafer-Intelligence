import os
import gc
import psutil
import pickle
import sys
import pandas as pd
import numpy as np

# 1. Core compatibility hook for the pickle loader
import pandas.core.indexes
sys.modules['pandas.indexes'] = pandas.core.indexes

from sklearn.model_selection import train_test_split

# File locations matching your structure
INPUT_FILE = r"C:\Users\suraj\Documents\Tejashree_Python\FabMind\data\raw\LSWMD.pkl"
OUTPUT_DIR = r"C:\Users\suraj\Documents\Tejashree_Python\FabMind\data\processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# RAM Check
memory = psutil.virtual_memory()
print("Available RAM:", round(memory.available / (1024**3), 2), "GB")

if memory.available < 6 * (1024**3):
    raise MemoryError("Insufficient RAM available to process the large LSWMD structure.")

print("\nLoading original dataset...")
with open(INPUT_FILE, 'rb') as f:
    df = pickle.load(f, encoding='latin1')

# Clean columns
df.columns = [col.decode('utf-8') if isinstance(col, bytes) else str(col) for col in df.columns]

# Extract nested defect labels if failureType isn't a direct column name
if 'failureType' not in df.columns:
    print("Extracting nested defect labels from target variables...")
    for possible_col in df.columns:
        first_val = df[possible_col].iloc[0]
        if isinstance(first_val, (list, np.ndarray)) and len(first_val) > 0:
            test_str = str(first_val)
            if any(cls in test_str for cls in ['Center', 'Donut', 'Edge', 'Scratch', 'None']):
                df['failureType'] = df[possible_col]
                break

label_column = "failureType"

def extract_clean_class(val):
    if isinstance(val, (list, np.ndarray)):
        if len(val) > 0 and len(val[0]) > 0:
            return str(val[0][0]).strip()
        return "nan"
    return str(val).strip()

df[label_column] = df[label_column].apply(extract_clean_class)

# Drop missing or unmapped rows
labelled = df[
    df[label_column].notna()
    & ~df[label_column].isin(["", "nan", "None", "[]", "None found"])
].copy()

print("\n--- Validated 9-Class Distribution ---")
print(labelled[label_column].value_counts())

# FIX: Replaced .apply() grouping with list comprehension to cleanly protect columns
max_samples_per_class = 2000
sampled_groups = []

for name, group in labelled.groupby(label_column):
    n_samples = min(len(group), max_samples_per_class)
    sampled_groups.append(group.sample(n=n_samples, random_state=42))

subset = pd.concat(sampled_groups, ignore_index=True)

# Explicitly guarantee columns are standard strings right before splitting
subset.columns = [str(col) for col in subset.columns]

print(f"\nSelected balanced subset size: {len(subset)}")
print("Verified final columns:", subset.columns.tolist())

# Split matching config: train_val_split: 0.85
train, temporary = train_test_split(
    subset,
    test_size=0.15,
    random_state=42,
    stratify=subset[label_column]
)

# Splitting remaining 15% between Validation and Test equally
validation, test = train_test_split(
    temporary,
    test_size=0.50,
    random_state=42,
    stratify=temporary[label_column]
)

# Save as high-performance Python 3 Pickles
train.to_pickle(os.path.join(OUTPUT_DIR, "train.pkl"), protocol=pickle.HIGHEST_PROTOCOL)
validation.to_pickle(os.path.join(OUTPUT_DIR, "validation.pkl"), protocol=pickle.HIGHEST_PROTOCOL)
test.to_pickle(os.path.join(OUTPUT_DIR, "test.pkl"), protocol=pickle.HIGHEST_PROTOCOL)

print("\nDataset preparation completed successfully!")
print(f"Training samples (85%): {len(train)}")
print(f"Validation samples: {len(validation)}")
print(f"Testing samples: {len(test)}")
print("Saved files in:", OUTPUT_DIR)

del df, labelled, subset, train, validation, test, temporary
gc.collect()
