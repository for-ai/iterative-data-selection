import numpy as np
import pickle

def cosine_similarity(a, b):
    """Calculate the cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def compute_diversity(data, subset):
    """Compute the diversity score of the subset with respect to the full dataset.
    The diversity is calculated as the sum of the maximum similarities of each data point in the 
    entire dataset to the closest point in the subset."""
    diversity_score = 0
    for v in data:
        max_sim = max(cosine_similarity(v, a) for a in subset)
        diversity_score += max_sim
    return diversity_score

def qdit_selection(data, data_quality, K, alpha):
    """Select a subset of data points using the QDIT algorithm.
    This function optimizes a quality-diversity score using a greedy approach."""
    N = len(data)
    subset_indices = []
    remaining_indices = list(range(N))

    while len(subset_indices) < K:
        best_score = -np.inf
        best_index = None

        for i in remaining_indices:
            candidate_subset = subset_indices + [i]
            # Compute the diversity with respect to the candidate subset including the new point
            diversity_score = compute_diversity(data, [data[j] for j in candidate_subset])
            # Only consider the quality score of the candidate data point (not the average of subset)
            quality_score = data_quality[i]
            qd_score = (1 - alpha) * diversity_score + alpha * quality_score
            
            if qd_score > best_score:
                best_score = qd_score
                best_index = i

        subset_indices.append(best_index)
        remaining_indices.remove(best_index)

    return subset_indices

# # Example usage
# data = np.random.rand(100, 10)  # 100 data points, each is a 10-dimensional vector
# data_quality = np.random.rand(100)  # Quality scores for each data point
# K = 10  # Number of points to select
# alpha = 0.5  # Trade-off parameter between quality and diversity

# selected_indices = qdit_selection(data, data_quality, K, alpha)
# print("Selected indices:", selected_indices)

# with open('qdit_selected_indices.pkl', 'wb') as f:
#     pickle.dump(selected_indices, f)