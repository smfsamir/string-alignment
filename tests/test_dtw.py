import numpy as np
import panphon
import pytest
from string_alignment.pmi_align import dtw_alignment

def test_phonetic_forced_alignment(): 
    ipa_str_one = ['n', 'ɪ', 'k']
    ipa_str_two = ['n','ɛ','k']

    ft = panphon.FeatureTable()
    embeddings_one = [ft.word_fts(segment)[0] for segment in ipa_str_one]
    assert len(embeddings_one) == 3
    embeddings_two = [ft.word_fts(segment)[0] for segment in ipa_str_two]
    assert len(embeddings_two) == 3
    distance_matrix = np.zeros((len(embeddings_one), len(embeddings_two)))
    for i, embedding_one in enumerate(embeddings_one):
        for j, embedding_two in enumerate(embeddings_two):
            distance_matrix[i, j] = embedding_one - embedding_two
    alignment_inds = dtw_alignment(distance_matrix)
    assert alignment_inds == [0, 1, 2]

# def test_phonetic_forced_alignment_two():
#     ipa_str_one = ['n', 'ɪ', '', 'k']
#     ipa_str_two = ['n','ɛ','k']

#     ft = panphon.FeatureTable()
#     embeddings_one = [ft.word_fts(segment)[0] for segment in ipa_str_one]
#     assert len(embeddings_one) == 3
#     embeddings_two = [ft.word_fts(segment)[0] for segment in ipa_str_two]
#     assert len(embeddings_two) == 3
#     distance_matrix = np.zeros((len(embeddings_one), len(embeddings_two)))
#     for i, embedding_one in enumerate(embeddings_one):
#         for j, embedding_two in enumerate(embeddings_two):
#             distance_matrix[i, j] = embedding_one - embedding_two
#     alignment_inds = dtw_alignment(distance_matrix)
#     assert alignment_inds == [0, 1, 2]

def test_phonetic_forced_alignment_three():
    ipa_str_one = ['n', 'ɪ', 'k']
    ipa_str_two = ['n','ɛ','k', 'k']

    ft = panphon.FeatureTable()
    embeddings_one = [ft.word_fts(segment)[0] for segment in ipa_str_one]
    assert len(embeddings_one) == 3
    embeddings_two = [ft.word_fts(segment)[0] for segment in ipa_str_two]
    assert len(embeddings_two) == 4
    distance_matrix = np.zeros((len(embeddings_one), len(embeddings_two)))
    for i, embedding_one in enumerate(embeddings_one):
        for j, embedding_two in enumerate(embeddings_two):
            distance_matrix[i, j] = embedding_one - embedding_two
    alignment_inds = dtw_alignment(distance_matrix)
    assert alignment_inds == [0, 1, 2, 2]

def test_phonetic_forced_alignment_four():
    ipa_str_one = ['n', 'ɪ', 'ɪ',  'k']
    ipa_str_two = ['n','ɛ', 'k']

    ft = panphon.FeatureTable()
    embeddings_one = [ft.word_fts(segment)[0] for segment in ipa_str_one]
    assert len(embeddings_one) == 4
    embeddings_two = [ft.word_fts(segment)[0] for segment in ipa_str_two]
    assert len(embeddings_two) == 3
    distance_matrix = np.zeros((len(embeddings_one), len(embeddings_two)))
    for i, embedding_one in enumerate(embeddings_one):
        for j, embedding_two in enumerate(embeddings_two):
            distance_matrix[i, j] = embedding_one - embedding_two
    alignment_inds = dtw_alignment(distance_matrix)
    assert alignment_inds == [0, 1, 3]


def test_phonetic_forced_alignment_five(): # TODO: this is currently failing. Need to fix
    ipa_str_one = ['n', 'ɪ', 'k']
    ipa_str_two = ['n','ɛ','ɪ', 'k']

    ft = panphon.FeatureTable()
    embeddings_one = [ft.word_fts(segment)[0] for segment in ipa_str_one]
    assert len(embeddings_one) == 3
    embeddings_two = [ft.word_fts(segment)[0] for segment in ipa_str_two]
    assert len(embeddings_two) == 4
    distance_matrix = np.zeros((len(embeddings_one), len(embeddings_two)))
    for i, embedding_one in enumerate(embeddings_one):
        for j, embedding_two in enumerate(embeddings_two):
            distance_matrix[i, j] = embedding_one - embedding_two
    alignment_inds = dtw_alignment(distance_matrix)
    assert alignment_inds == [0, 1, 1, 2]