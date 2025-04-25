import numpy as np

def calculate_rmse(actual: np.array, predicted: np.array) -> float:
    """Calculate Root Mean Square Error"""
    return round(np.sqrt(np.mean((actual - predicted) ** 2)), 2) 