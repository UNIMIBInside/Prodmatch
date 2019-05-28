from tqdm import tqdm
import nltk
import torch
import logging
import deepmatcher as dm
from os import path
from itertools import chain
from pandas import pandas
from ceneje_prodmatch import DATA_DIR, DEEPMATCH_DIR, RESULTS_DIR, CACHE_DIR
from ceneje_prodmatch.scripts.helpers import preprocess
from ceneje_prodmatch.scripts.helpers.deepmatcherdata import deepmatcherdata

logging.getLogger('deepmatcher.core')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_similarity_scores(
	unlabeled: pandas.DataFrame,
	distance='jaccard_distance', 
	ngrams=3, 
	left_attr='ltable_', 
	right_attr='rtable_', 
	ignore_columns=[],
	na_value=''):
	
	def similarity(row_left, row_right):
		# Distance word based
		if ngrams == 0:
			left_prod = chain(*map(lambda s: s.split(' '), row_left.tolist()))
			right_prod = chain(*map(lambda s: s.split(' '), row_right.tolist()))
		else: 
			left_prod = nltk.ngrams(' '.join(row_left.tolist()), n=ngrams)
			right_prod = nltk.ngrams(' '.join(row_right.tolist()), n=ngrams)
		if distance == 'jaccard_distance':
			left_prod = set(left_prod)
			right_prod = set(right_prod)
		elif distance == 'edit_distance':
			left_prod = list(left_prod)
			right_prod = list(right_prod)
		# Get similarity
		return 1 - getattr(nltk, distance)(left_prod, right_prod) / max([len(left_prod), len(right_prod)])

	assert(ngrams >= 0)

	unlabeled = unlabeled.fillna(na_value)
	left_cols = [
		col for col in unlabeled 
		if col.startswith(left_attr) and col not in ignore_columns
	]
	right_cols = [
		col for col in unlabeled 
		if col.startswith(right_attr) and col not in ignore_columns
	]
	tqdm.pandas()
	match_score = unlabeled[left_cols + right_cols]\
					.progress_apply(lambda row: similarity(row[left_cols], row[right_cols]), axis=1)
	unlabeled.insert(0, 'match_score', match_score)
	return unlabeled

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
	results =  results[['match_score', match_pred_attr] + results.columns.values[1:-1].tolist()]
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

def get_pos_neg_ratio(path:str, **kwargs):
	data = pandas.read_csv(path, **kwargs)
	pos = len(data[data[['label']] == 1])
	neg = len(data) - pos
	print(neg / pos)


if __name__ == "__main__":
	columns = ['idProduct']
	ignore_columns = ['ltable_' + col for col in columns]
	ignore_columns += ['rtable_' + col for col in columns]
	ignore_columns += ['idProduct']
	get_pos_neg_ratio(path.join(DEEPMATCH_DIR, 'train.csv'))
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
	#     attr_summarizer=dm.attr_summarizers.SIF(),
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

	# unlabeled = pandas.read_csv(path.join(DEEPMATCH_DIR, 'unlabeled.csv'))
	# predictions = get_similarity_scores(unlabeled, ngrams=0, distance='edit_distance', ignore_columns=ignore_columns, na_value='na')
	# predictions = get_match_predictions(predictions)
	# predictions.to_csv(path.join(RESULTS_DIR, 'predictions_jaccard_3.csv'))
	# print(get_statistics(predictions))