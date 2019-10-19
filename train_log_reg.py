import os
import math
import nltk
import json
import torch
import logging
import deepmatcher as dm
import py_stringmatching as sm
import torch.optim as optim
from torch import nn
from os import path
from tqdm import tqdm
from itertools import chain
from pandas import pandas
from torch.utils.data import Dataset, DataLoader
from ceneje_prodmatch import DATA_DIR, DEEPMATCH_DIR, RESULTS_DIR, CACHE_DIR, CONFIG_DIR
from ceneje_prodmatch.src.matching.similarity import Similarity, SimilarityDataset, LogisticRegressionModel
from ceneje_prodmatch.src.matching.runner import Runner

logging.getLogger('deepmatcher.core')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_match_predictions(
        results: pandas.DataFrame,
        threshold=0.5,
        match_pred_attr='match_prediction'):

    # Check if match_score column is present
    assert('match_score' in results.columns)
    # Check threshold range
    assert(threshold >= 0 and threshold <= 1)
    results[match_pred_attr] = results['match_score']\
        .apply(lambda score: 1 if score >= threshold else 0)
    # Reorder columns to avoid scrolling
    results = results[['match_score', match_pred_attr] +
                      results.columns.values[1:-1].tolist()]
    return results


def get_statistics(
        results: pandas.DataFrame,
        label_attr='label',
        match_pred_attr='match_prediction'):

    assert(match_pred_attr in results.columns)
    assert(label_attr in results.columns)
    TP = FP = TN = FN = 0
    wrong_preds = []
    for i, row in results.iterrows():
        if row[match_pred_attr] == 1:
            if row[label_attr] == 1:
                TP += 1
            else:
                FP += 1
                wrong_preds += [i]
        else:
            if row[label_attr] == 1:
                FN += 1
                wrong_preds += [i]
            else:
                TN += 1
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accurcy = (TP + TN) / (TP + FP + TN + FN)
    F1 = 2 * (precision * recall) / (precision + recall)
    return {'precision': precision,
            'recall': recall,
            'accuracy': accurcy,
            'F1': F1,
            'wrong': results.loc[wrong_preds]}


def get_pos_neg_ratio(dataset: pandas.DataFrame, label_attr='label'):
    pos = len(dataset[dataset[label_attr] == 1])
    neg = len(dataset) - pos
    return neg / pos


if __name__ == "__main__":

    # Import config

    with open(os.path.join(CONFIG_DIR, 'config.json')) as f:
        cfg = json.load(f)
    deepmatcher_cfg = cfg['deepmatcher']

    columns = deepmatcher_cfg['train']['left_right_ignore_cols']
    ignore_columns  = [deepmatcher_cfg['create']['left_prefix'] + col for col in columns]
    ignore_columns += [deepmatcher_cfg['create']['right_prefix'] + col for col in columns]
    if deepmatcher_cfg['create']['create_nm_mode'] == 'similarity':
        ignore_columns += ['similarity']
    train = pandas.read_csv(path.join(DEEPMATCH_DIR, cfg['split']['train_data_name'] + '.csv'))
    pos_neg_ratio = get_pos_neg_ratio(train)

    # Run a similarity matching algorithm based on manual weight of the attributes

    """ unlabeled = pandas.read_csv(path.join(DEEPMATCH_DIR, cfg['split']['unlabeled_data_name'] + '.csv'))
    simil = Similarity(data=unlabeled, ignore_columns=ignore_columns, na_value='')
    predictions = simil.get_scores(tokenizer=sm.QgramTokenizer(qval=5))
    # predictions = pandas.read_csv(path.join(RESULTS_DIR, 'predictions_no_name_prod_hybrid.csv'))
    predictions = get_match_predictions(predictions)
    # predictions.to_csv(path.join(RESULTS_DIR, 'predictions_jaccard_5.csv'))
    print(get_statistics(predictions)) """

    # Run a similarity matching algorithm based on simple logistic regression

    train = pandas.read_csv(path.join(DEEPMATCH_DIR, cfg['split']['train_data_name'] + '.csv'))
    val = pandas.read_csv(path.join(DEEPMATCH_DIR, cfg['split']['val_data_name'] + '.csv'))
    unlabeled = pandas.read_csv(path.join(DEEPMATCH_DIR, cfg['split']['unlabeled_data_name'] + '.csv'))

    train_dataset = SimilarityDataset(train, ignore_columns=ignore_columns)
    val_dataset = SimilarityDataset(val, ignore_columns=ignore_columns)
    test_dataset = SimilarityDataset(unlabeled, ignore_columns=ignore_columns)

    # pos_weight = 2 * pos_neg_ratio / (1 + pos_neg_ratio)
    # neg_weight = 2 - pos_weight

    pos_neg_ratio = get_pos_neg_ratio(train)
    model = LogisticRegressionModel(input_dim=3)

    print(pos_neg_ratio)
    model.run_train(
        train_dataset,
        val_dataset,
        model,
        resume=False,
        train_epochs=5,
        pos_neg_ratio=pos_neg_ratio,
        log_freq=16
    )
    predictions = model.run_predict(test_dataset)
    unlabeled.insert(0, 'match_score', predictions)
    # unlabeled.to_csv(path.join(RESULTS_DIR, 'predictions_logistic_desc.csv'))
    predictions = get_match_predictions(unlabeled)

    print(get_statistics(predictions))

    # predictions = pandas.read_csv(path.join(RESULTS_DIR, deepmatcher_cfg['train']['predictions_data_name'] + '.csv'))
    """ predictions = get_match_predictions(predictions)
    print(get_statistics(predictions)) """
