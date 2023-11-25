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
    # set 0 entries to a value smaller than 1 to avoid division by 0.
    co_occurrence_matrix[co_occurrence_matrix == 0] = 0.00000001
    # compute the PPMI matrix
    ppmi_matrix = np.log(co_occurrence_matrix / np.outer(np.sum(co_occurrence_matrix, axis=1), np.sum(co_occurrence_matrix, axis=0)))
    # set negative values to 0
    ppmi_matrix[ppmi_matrix < 0] = 0
    return ppmi_matrix