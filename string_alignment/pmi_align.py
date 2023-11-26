import numpy as np
from typing import List, Iterable, Dict


def compute_ppmi_matrix(datapoint_pred_counts: Iterable[Dict[str, int]], datapoint_ref_counts: Iterable[Dict[str, int]]) -> List[List[float]]:
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
    return _pmi