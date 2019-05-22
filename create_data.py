import os
import numpy
import time
from os import path
from pandas import pandas
from ceneje_prodmatch import DATA_DIR, DEEPMATCH_DIR
from ceneje_prodmatch.scripts.helpers import preprocess
from ceneje_prodmatch.scripts.helpers.deepmatcherdata import deepmatcherdata

def fillCols(row, col1, col2):
	if row[col1] == '':
		row[col1] = row[col2]
	else:
		row[col2] = row[col1]
	return row

def read_files_start_with(folder: str, prefix: str, products=None, **kwargs):
	"""
	Function that reads csv files, starting with a user defined prefix, from a specified folder

	Parameters
	----------
	folder (str): folder from which read files
	prefix (str): pick files only starting with prefix
	kwargs: keyword arguments to pass to pandas.read_csv() function

	Returns
	-------
	List of pandas.DataFrame object, sorted by filename
	"""
	if products is None:
		prefixed = sorted(
			[filename for filename in os.listdir(folder) if filename.startswith(prefix)]
		)
	else:
		prefixed = sorted(
			[filename for filename in os.listdir(folder) 
				if filename.startswith(prefix) and 
				any(product for product in products if product in filename)
			]
		)
	return 	[
				pandas.read_csv(
					path.join(folder, prefixed[i]),  
					dtype={'idProduct':object, 'idSeller':object, 'idSellerProduct':object},
					**kwargs
				) 
				for i in range(len(prefixed))
	]

def read_files(folder: str, prefixes: str, products=None, **kwargs):
	"""
	Function that reads csv files, starting with a user defined prefixes, from a specified folder

	Parameters
	----------
	folder (str): folder from which read files
	prefixes (str): pick files only starting with prefixes
	products (list or None): pick only files that contains products after prefix
	kwargs: keyword arguments to pass to pandas.read_csv() function

	Returns
	-------
	List of tuples of pandas.DataFrame object. Every tuple contains n pandas.DataFrame objects, where n=len(prefixes),
	following the order specified by prefixes, ready for future ordered automatic joins 
	"""
	files = [
				read_files_start_with(folder, prefixes[i], products, **kwargs)
				for i in range(len(prefixes))
	]
	return list(zip(*files))

def join_datasets(datasets: list):
	"""
	Since every join will merge SellerProductsData x SellerProductsMapping x Products (if i'm not wrong),
	this function executes those joins given a list of tuple of datasets, where every tuple contains dataset
	following SellerProductsData_, SellerProductsMapping_, Products_ order

	Parameters
	----------
	datasets (list of tuples of pandas.DataFrame): list that contains tuples of pandas.DataFrame.
		Every dataset in a tuples will be joined from left to right
	"""
	return [
		# SellerProductsData_ dataset
		datasets[i][0][['idSeller', 'idSellerProduct', 'brandSeller', 'nameSeller', 'descriptionSeller']]\
			.merge(
				right=datasets[i][1],  # SellerProductsMapping_ dataset
				how='inner', 
				on=['idSellerProduct', 'idSeller']
			).merge(
				right=datasets[i][2][['idProduct', 'nameProduct', 'brand']],  # Products_ dataset
				how='inner',
				on='idProduct'
			)
		for i in range(len(datasets))
	]

def get_normalized_matching(integrated_data: list):
	return pandas.concat(
		[
			preprocess.normalize(
				integrated_data[i][integrated_data[i].duplicated(subset='idProduct', keep=False)]
			)
			for i in range(len(integrated_data))
		]
	)

if __name__ == '__main__':
	init = time.time()
	files = read_files(
		folder=DATA_DIR, 
		prefixes=['SellerProductsData', 'SellerProductsMapping', 'Products'], 
		products=['Monitor'],
		sep='\t', 
		encoding='utf-8'
	)
	integrated_data = join_datasets(files)
	matching = get_normalized_matching(integrated_data) 
	matching.to_csv(path.join(DATA_DIR, 'matching.csv'))
	# Set seed for reproducible results
	# numpy.random.seed(42)
	# Read data
	# sellerProdData = pandas.read_csv(path.join(DATA_DIR, 'SellerProductsData_LedTv_20190426.csv'), 
	# 				sep='\t',
	# 				encoding='utf-8')
	# sellerProdMapping = pandas.read_csv(path.join(DATA_DIR, 'SellerProductsMapping_LedTv_20190426.csv'), 
	# 				sep='\t',
	# 				encoding='utf-8')
	# prodData = pandas.read_csv(path.join(DATA_DIR, 'Products_LedTv_20190426.csv'), 
	# 				sep='\t',
	# 				encoding='utf-8')
	# cenejeAttributes = pandas.read_csv(path.join(DATA_DIR, 'CenejeAttributes_LedTv_20190426.csv'), 
	# 				sep='\t',
	# 				encoding='utf-8')
	# # Join sellerProdData and sellerProdMapping
	# integrated = pandas.merge(
	# 	left=sellerProdData[['idSeller', 'idSellerProduct', 'brandSeller', 'nameSeller', 'descriptionSeller']],
	# 	right=sellerProdMapping,
	# 	how='inner', 
	# 	on=['idSellerProduct', 'idSeller']
	# )
	# # Join result of previous join with prodData
	# integrated = pandas.merge(
	# 	left=integrated,
	# 	right=prodData[['idProduct', 'nameProduct', 'brand']], 
	# 	how='inner',
	# 	on='idProduct'
	# )
	# # Fill empty brand and brandSeller from one another
	# # since there're mapping
	# # integrated[['brand', 'brandSeller']] = integrated[['brand', 'brandSeller']].apply(
	# # 	lambda row: fillCols(row, 'brand', 'brandSeller'),
	# # 	axis=1
	# # )
	
	# # Integration with ceneje attributes data
	# # integrated_attr = pandas.merge(
	# # 	left=integrated,
	# # 	right=cenejeAttributes.drop_duplicates(['idProduct', 'idAtt'], keep='first')[['idProduct', 'nameAtt', 'attValue']],
	# # 	how='left',
	# # 	on='idProduct'
	# # )
	# # integrated_attr.to_csv(path.join(DATA_DIR, 'AttributesIntegratedProducts.tsv'), sep='\t')

	# # Take only those records that have idProduct duplicated (as they match)
	# matching = integrated[integrated.duplicated(subset='idProduct', keep=False)]
	# # Clean rubbish HTML from text
	# matching = preprocess.normalize(matching, na_value='not_available')
	# integrated.to_csv(path.join(DATA_DIR, 'IntegratedProducts.csv'))
	# matching.to_csv(path.join(DATA_DIR, 'Matching.csv'))
	keys = ['idProduct', 'brandSeller', 'nameSeller', 'descriptionSeller', 'nameProduct', 'brand']
	deepdata = deepmatcherdata(
		matching, 
		group_cols=['idProduct'], 
		keys=keys,
		id_attr='_id',
		left_attr='ltable_',
		right_attr='rtable_',
		label_attr='label',
		normalize=False, 
		perc=.001
	)
	data = deepdata.deepdata
	print(time.time() - init)
	data.to_csv(path.join(DEEPMATCH_DIR, 'deepmatcher.csv'))
	train, val, test = deepdata.train_val_test_split([0.6, 0.2, 0.2])
	unlabeled_data = train[:int(len(train) * 0.2)]
	train = train[int(len(train) * 0.2):]
	train.to_csv(path.join(DEEPMATCH_DIR, 'train.csv'))
	val.to_csv(path.join(DEEPMATCH_DIR, 'validation.csv'))
	test.to_csv(path.join(DEEPMATCH_DIR, 'test.csv'))
	unlabeled_data.to_csv(path.join(DEEPMATCH_DIR, 'unlabeled.csv'))