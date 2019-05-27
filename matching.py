import torch
import logging
import deepmatcher as dm
from os import path
from pandas import pandas
from ceneje_prodmatch import DATA_DIR, DEEPMATCH_DIR, RESULTS_DIR, CACHE_DIR
from ceneje_prodmatch.scripts.helpers import preprocess
from ceneje_prodmatch.scripts.helpers.deepmatcherdata import deepmatcherdata

logging.getLogger('deepmatcher.core')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_match_predictions(results: pandas.DataFrame, threshold: float, match_pred_attr: str):
	# Check if match_score column is present
	assert('match_score' in results.columns)
	# Check threshold range
	assert(threshold >= 0 and threshold <= 1)
	results['match_prediction'] = results['match_score']\
		.apply(lambda score: 1 if score >= threshold else 0)
	# Reorder columns to avoid scrolling...
	results =  results[['match_score', 'match_prediction'] + results.columns.values[1:-1].tolist()]
	return results

def get_statistics(results: pandas.DataFrame, label_attr: str, match_pred_attr: str):
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
	accurcy = (TP + TN) / len(results)
	F1 = 2 * (precision * recall) / (precision + recall)
	return {'precision': precision, 'recall': recall, 'accuracy': accurcy, 'F1': F1, 'wrong': results.iloc[wrong_preds]}

if __name__ == "__main__":
	predictions = pandas.read_csv('/home/belerico/Desktop/predictions_no_name_prod.csv')
	predictions = get_match_predictions(predictions, 0.9, 'match_prediction')
	print(get_statistics(predictions, 'label', 'match_prediction'))
	# columns = ['idProduct']
	# ignore_columns = ['ltable_' + col for col in columns]
	# ignore_columns += ['rtable_' + col for col in columns]
	# ignore_columns += ['idProduct']
	# train, validation, test = dm.data.process(
	#     path=DEEPMATCH_DIR,
	#     cache=path.join(CACHE_DIR, 'rnn_lstm_fasttext_cache.pth'),
	#     train='train.csv',
	#     validation='validation.csv',
	#     test='test.csv',
	#     ignore_columns=ignore_columns,
	#     lowercase=False,
	#     embeddings='fasttext.sl.bin',
	#     id_attr='_id',
	#     label_attr='label',
	#     left_prefix='ltable_',
	#     right_prefix='rtable_',
	#     pca=False,
	#     device=device
	# )
	# model = dm.MatchingModel(
	#     attr_summarizer=dm.attr_summarizers.RNN(
	#         word_contextualizer='lstm'
	#     ),
	#     attr_comparator='abs-diff'
	# )
	# model.initialize(train, device=device)
	# model.run_train(
	#     train,
	#     validation,
	#     epochs=10,
	#     batch_size=16,
	#     best_save_path=path.join(RESULTS_DIR, 'models', 'rnn_lstm_fasttext_model.pth'),
	#     device=device
	# )
	# model.run_eval(test, device=device)
	# model.load_state(path.join(RESULTS_DIR, 'models', 'rnn_lstm_fasttext_model.pth'), device=device)
	# candidate = dm.data.process_unlabeled(
	#     path=path.join(DEEPMATCH_DIR, 'unlabeled.csv'),
	#     trained_model=model     ,
	#     ignore_columns=ignore_columns + ['label'])
	# predictions = model.run_prediction(candidate, output_attributes=list(candidate.get_raw_table().columns), device=device)
	# predictions.to_csv(path.join(RESULTS_DIR, 'predictions.csv'))