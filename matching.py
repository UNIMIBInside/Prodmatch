import math
import nltk
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
from ceneje_prodmatch import DATA_DIR, DEEPMATCH_DIR, RESULTS_DIR, CACHE_DIR
from ceneje_prodmatch.scripts.matching.similarity import Similarity, SimilarityDataset, LogisticRegressionModel
from ceneje_prodmatch.scripts.matching.runner import Runner

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
            'wrong': results.iloc[wrong_preds]}


def get_pos_neg_ratio(dataset: pandas.DataFrame, label_attr='label'):
    pos = len(dataset[dataset[label_attr] == 1])
    neg = len(dataset) - pos
    return neg / pos


if __name__ == "__main__":
    columns = ['idProduct']
    ignore_columns = ['left_' + col for col in columns]
    ignore_columns += ['right_' + col for col in columns]
    ignore_columns += ['idProduct', 'similarity']
    train = pandas.read_csv(path.join(DEEPMATCH_DIR, 'train.csv'))
    pos_neg_ratio = get_pos_neg_ratio(train)

    # Run deepmatcher algorithm

    train, validation, test = dm.data.process(
        path=DEEPMATCH_DIR, cache=path.join(
            CACHE_DIR, 'rnn_pos_neg_fasttext_jaccard_desc_cache.pth'),
        train='train_desc.csv', validation='validation_desc.csv', test='test_desc.csv',
        ignore_columns=ignore_columns, lowercase=False,
        embeddings='fasttext.sl.bin', id_attr='id', label_attr='label',
        left_prefix='left_', right_prefix='right_', pca=False, device=device)
    model = dm.MatchingModel(
        attr_summarizer=dm.attr_summarizers.RNN(),
        attr_comparator='abs-diff'
    )
    model.initialize(train, device=device)
    model.run_train(
        train,
        validation,
        epochs=10,
        batch_size=16,
        pos_neg_ratio=pos_neg_ratio,
        best_save_path=path.join(RESULTS_DIR, 'models',
                                 'rnn_pos_neg_fasttext_jaccard_desc_model.pth'),
        device=device
    )
    model.run_eval(test, device=device)
    model.load_state(
        path.join(
            RESULTS_DIR, 'models',
            'rnn_pos_neg_fasttext_jaccard_desc_model.pth'),
        device=device)
    candidate = dm.data.process_unlabeled(
        path.join(DEEPMATCH_DIR, 'unlabeled_desc.csv'),
        trained_model=model,
        ignore_columns=ignore_columns + ['label'])
    predictions = model.run_prediction(candidate, output_attributes=True, device=device)
    predictions.to_csv(
        path.join(
            RESULTS_DIR, 'predictions_rnn_pos_neg_fasttext_jaccard_desc.csv'))

    # Run a similarity matching algorithm based on manual weight of the attributes

    """ unlabeled = pandas.read_csv(path.join(DEEPMATCH_DIR, 'unlabeled.csv'))
	simil = Similarity(data=unlabeled, ignore_columns=ignore_columns, na_value='na')
	predictions = simil.get_scores(metric=sm.Jaccard())
	# predictions = pandas.read_csv(path.join(RESULTS_DIR, 'predictions_no_name_prod_hybrid.csv'))
	predictions = get_match_predictions(predictions)
	# predictions.to_csv(path.join(RESULTS_DIR, 'predictions_jaccard_5.csv'))
	print(get_statistics(predictions))
	"""

    # Run a similarity matching algorithm based on simple logistic regression

    """ train = pandas.read_csv(path.join(DEEPMATCH_DIR, 'train.csv'))
    val = pandas.read_csv(path.join(DEEPMATCH_DIR, 'validation.csv'))
    unlabeled = pandas.read_csv(path.join(DEEPMATCH_DIR, 'unlabeled.csv'))

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
        resume=True,
        train_epochs=10, 
        pos_neg_ratio=pos_neg_ratio,
        log_freq=16
    )
    predictions = model.run_predict(test_dataset)
    unlabeled.insert(0, 'match_score', predictions)
    predictions = get_match_predictions(unlabeled)
    print(get_statistics(predictions)) """

    """ predictions = pandas.read_csv(path.join(RESULTS_DIR, 'predictions_rnn_pos_neg_fasttext_jaccard_name.csv'))
    predictions = get_match_predictions(predictions)
    print(get_statistics(predictions)) """
