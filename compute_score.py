import os
import json
import logging
from os import path
from pandas import pandas
from ceneje_prodmatch import DATA_DIR, DEEPMATCH_DIR, RESULTS_DIR, CACHE_DIR, CONFIG_DIR

logging.getLogger('deepmatcher.core')

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


if __name__ == '__main__':

    # Import cfg
    with open(os.path.join(CONFIG_DIR, 'config.json')) as f:
        cfg = json.load(f)
    deepmatcher_cfg = cfg['deepmatcher']

    stats = {}
    for prediction in os.listdir(RESULTS_DIR):
        if prediction.startswith('rnn'):
            predictions = pandas.read_csv(os.path.join(RESULTS_DIR, prediction))
            stat = get_match_predictions(predictions)
            stat = get_statistics(stat)
            stats[prediction] = {
                'precision': stat['precision'],
                'recall': stat['recall'],
                'accuracy': stat['accuracy'],
                'F1': stat['F1']
            }
    stats = sorted(stats.items(), key=lambda x: x['F1'])
    print(stats)