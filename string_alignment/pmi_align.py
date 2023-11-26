import numpy as np
from typing import List, Iterable, Dict, Tuple


def compute_ppmi_matrix(datapoint_pred_counts: Iterable[Dict[str, int]], datapoint_ref_counts: Iterable[Dict[str, int]]) -> Tuple[np.array, List[str], List[str]]:
    """
    Computes the pointwise mutual information matrix for a given datapoint
    :param datapoint_pred_counts: The counts of each token (e.g., word or character) in the predicted sentence
    :param datapoint_ref_counts: The counts of each word in the reference sentence
    :return: The pointwise mutual information matrix
    """
    vocab_preds = sorted(list(set([token for pred_counts in datapoint_pred_counts for token in pred_counts.keys()])))
    vocab_refs = sorted(list(set([token for ref_counts in datapoint_ref_counts for token in ref_counts.keys()])))
    co_occurrence_matrix = np.zeros((len(vocab_preds), len(vocab_refs))) 
    # the co-occurrence matrix is a matrix of size |V_pred| x |V_ref|, where |V_pred| is the size of the vocabulary of the predicted sentence and |V_ref| is the size of the vocabulary of the reference sentence

    for pred_counts, ref_counts in zip(datapoint_pred_counts, datapoint_ref_counts):
        for pred_token in pred_counts.keys():
            for ref_token in ref_counts.keys():
                co_occurrence_matrix[vocab_preds.index(pred_token), vocab_refs.index(ref_token)] += 1
    arr = co_occurrence_matrix
    row_totals = arr.sum(axis=1).astype(float)
    prob_cols_given_row = (arr.T / row_totals).T
    col_totals = arr.sum(axis=0).astype(float)
    prob_of_cols = col_totals / sum(col_totals)
    ratio = prob_cols_given_row / prob_of_cols
    ratio[ratio==0] = 0.00001
    _pmi = np.log(ratio)
    _pmi[_pmi<0] = 0
    return _pmi, vocab_preds, vocab_refs

def dtw_alignment(distance_matrix: np.array): # assumption: the columns hold the reference tokens and the rows hold the predicted tokens
    dtw_cost_matrix = np.full(distance_matrix.shape, np.inf)
    dtw_cost_matrix[0, 0] = 0

    for i in range(1, distance_matrix.shape[0]):
        for j in range(1, distance_matrix.shape[1]):
            cost = distance_matrix[i, j]
            dtw_cost_matrix[i, j] = cost + min(dtw_cost_matrix[i-1, j], # the prediction is too short here
                                               dtw_cost_matrix[i, j-1], # the prediction is too long here
                                               dtw_cost_matrix[i-1, j-1]) # correct prediction

    dtw_cost_matrix = dtw_cost_matrix.T

    i = len(dtw_cost_matrix) - 1  # Index for the last row
    j = len(dtw_cost_matrix[0]) - 1  # Index for the last column

    # Initialize the alignment indices list
    alignment_indices = [0] * len(dtw_cost_matrix)

    # Traverse the DTW matrix from bottom-right to top-left
    while i > 0 or j > 0:
        # Store the current alignment index
        alignment_indices[i] = j

        # Determine the next cell in the alignment path
        if i > 0 and (j == 0 or dtw_cost_matrix[i-1][j] <= min(dtw_cost_matrix[i-1][j-1], dtw_cost_matrix[i][j-1])):
            i = i - 1  # Move up (deletion or prediction too short)
        elif j > 0 and (i == 0 or dtw_cost_matrix[i][j-1] <= min(dtw_cost_matrix[i-1][j-1], dtw_cost_matrix[i-1][j])):
            j = j - 1  # Move left (insertion or observed too long)
        else:
            i = i - 1  # Move diagonally (substitution)

    # Set the alignment index for the starting point (top-left corner)
    alignment_indices[0] = 0

    return alignment_indices