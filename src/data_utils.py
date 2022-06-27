import json
from operator import itemgetter
from itertools import chain
import torch
import numpy as np
from torch.utils.data import TensorDataset



def get_queries(location):
    with open(location, 'r') as query_file:
        all_queries = json.load(query_file)
        return [q['query'].replace('\n', ' ').replace('.', '').lower() for q in all_queries]


def get_corpus(location):
    with open(location, 'r') as document_file:
        all_sentences = json.load(document_file)
        return [s['sentences'].replace('\n', ' ').replace('.', '').lower() for s in all_sentences]


def get_judgments(location):
    with open(location, 'r') as relevance:
        rel_fed = [rel.strip().replace('-1', '1').split(' ') for rel in relevance.readlines()]
        return [list(map(int, rf)) for rf in rel_fed if '' not in rf]


def load_fold(fold_number):
    train_index = np.load('/vinai/quannla/bert-meets-cranfield/data/folds/train_index_fold' + str(fold_number) + '.npy')
    test_index = np.load('/vinai/quannla/bert-meets-cranfield/data/folds/test_index_fold' + str(fold_number) + '.npy')
    return train_index, test_index


def get_tensor_dataset(index, padded, attention_mask, token_type_ids, temp_feedback):
    padded_list = list(itemgetter(*index)(padded))
    padded = list(chain.from_iterable(padded_list))

    attention_mask_list = list(itemgetter(*index)(attention_mask))
    attention_mask = list(chain.from_iterable(attention_mask_list))

    token_type_ids_list = list(itemgetter(*index)(token_type_ids))
    token_type_ids = list(chain.from_iterable(token_type_ids_list))

    labels_list = list(itemgetter(*index)(temp_feedback))
    labels = list(chain.from_iterable(labels_list))

    padded = torch.cat(padded, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(padded, attention_mask, token_type_ids, labels)
