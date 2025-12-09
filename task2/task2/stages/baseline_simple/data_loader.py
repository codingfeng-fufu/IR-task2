"""Simple data loader for stage1 - no preprocessing optimization"""

import pandas as pd
from typing import List, Tuple


def load_training_data(positive_file: str, negative_file: str) -> Tuple[List[str], List[int]]:
    """
    Load training data from positive and negative files.

    Args:
        positive_file: Path to positive titles file
        negative_file: Path to negative titles file

    Returns:
        titles: List of title strings
        labels: List of labels (1 for positive, 0 for negative)
    """
    titles = []
    labels = []

    # Load positive titles
    try:
        with open(positive_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    titles.append(line)
                    labels.append(1)
    except Exception as e:
        print(f"Error loading positive file: {e}")
        return [], []

    # Load negative titles
    try:
        with open(negative_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    titles.append(line)
                    labels.append(0)
    except Exception as e:
        print(f"Error loading negative file: {e}")
        return [], []

    print(f"Loaded {len([l for l in labels if l == 1])} positive titles")
    print(f"Loaded {len([l for l in labels if l == 0])} negative titles")
    print(f"Total training samples: {len(titles)}")

    return titles, labels


def load_test_data(test_file: str) -> Tuple[List[str], List[int]]:
    """
    Load test data from Excel file.

    Args:
        test_file: Path to test Excel file

    Returns:
        titles: List of title strings
        labels: List of labels (1 for Y, 0 for N)
    """
    try:
        df = pd.read_excel(test_file)

        # Get titles from 'title given by manchine' column
        titles = df['title given by manchine'].tolist()

        # Convert Y/N to 1/0
        labels = [1 if label == 'Y' else 0 for label in df['Y/N'].tolist()]

        # Filter out NaN values
        valid_data = [(t, l) for t, l in zip(titles, labels) if pd.notna(t) and isinstance(t, str)]
        titles = [t for t, l in valid_data]
        labels = [l for t, l in valid_data]

        print(f"Loaded {len(titles)} test samples")
        print(f"Positive: {sum(labels)}, Negative: {len(labels) - sum(labels)}")

        return titles, labels
    except Exception as e:
        print(f"Error loading test file: {e}")
        return [], []
