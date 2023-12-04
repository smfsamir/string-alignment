import numpy as np
from typing import List, Iterable, Dict, Tuple


def compute_ppmi_matrix(co_occurrence_counts: Dict[Tuple[str, str], int]) -> Tuple[np.array, List[str], List[str]]:
    """
    Computes the pointwise mutual information matrix. 
    :param co_occurrence_counts: A dictionary of the number of times two tokens co-occur. Assumed that first
    token is source and second token is target.
    :return: The pointwise mutual information matrix
    """
    # vocab_preds = sorted(list(set([token for pred_counts in datapoint_pred_counts for token in pred_counts.keys()])))
    # vocab_refs = sorted(list(set([token for ref_counts in datapoint_ref_counts for token in ref_counts.keys()])))

    # populate vocab_preds using the  
    vocab_src = sorted(list(set([token[0] for token in co_occurrence_counts.keys()])))
    vocab_tgt = sorted(list(set([token[1] for token in co_occurrence_counts.keys()])))
    co_occurrence_matrix = np.zeros((len(vocab_src), len(vocab_tgt))) 
    # the co-occurrence matrix is a matrix of size |V_src| x |V_tgt|, where |V_src| is the size of the vocabulary of the predicted sentence and |V_tgt| is the size of the vocabulary of the reference sentence
    # populate the co-occurrence matrix using the co-occurrence counts
    for (src, tgt), count in co_occurrence_counts.items():
        co_occurrence_matrix[vocab_src.index(src), vocab_tgt.index(tgt)] = count
    arr = co_occurrence_matrix
    row_totals = arr.sum(axis=1).astype(float)
    prob_cols_given_row = (arr.T / row_totals).T
    col_totals = arr.sum(axis=0).astype(float)
    prob_of_cols = col_totals / sum(col_totals)
    ratio = prob_cols_given_row / prob_of_cols
    ratio[ratio==0] = 0.00001
    _pmi = np.log(ratio)
    _pmi[_pmi<0] = 0
    return _pmi, vocab_src, vocab_tgt

def compute_npmi_matrix(co_occurrence_counts: Dict[Tuple[str, str], int]) -> Tuple[np.array, List[str], List[str]]:
    """
    Computes the normalized pointwise mutual information matrix. 
    :param co_occurrence_counts: A dictionary of the number of times two tokens co-occur. Assumed that first
    token is source and second token is target.
    :return: The normalized pointwise mutual information matrix
    """
    vocab_src = sorted(list(set([token[0] for token in co_occurrence_counts.keys()])))
    vocab_tgt = sorted(list(set([token[1] for token in co_occurrence_counts.keys()])))
    co_occurrence_matrix = np.zeros((len(vocab_src), len(vocab_tgt))) 

    for (src, tgt), count in co_occurrence_counts.items():
        co_occurrence_matrix[vocab_src.index(src), vocab_tgt.index(tgt)] = count

    arr = co_occurrence_matrix
    row_totals = arr.sum(axis=1).astype(float)
    prob_cols_given_row = (arr.T / row_totals).T
    col_totals = arr.sum(axis=0).astype(float)
    prob_of_cols = col_totals / sum(col_totals)

    pmi_ratio = np.log(prob_cols_given_row / np.outer(row_totals, prob_of_cols) + 1e-10)
    pmi_ratio[pmi_ratio < 0] = 0

    npmi_matrix = pmi_ratio / (-np.log(prob_cols_given_row) + 1e-10)
    npmi_matrix[np.isnan(npmi_matrix)] = 0

    return npmi_matrix, vocab_src, vocab_tgt


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