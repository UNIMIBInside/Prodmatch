import numpy
import time
from os import path
from pandas import pandas
from ceneje_prodmatch import DATA_DIR
from ceneje_prodmatch.scripts.helpers import preprocess
from ceneje_prodmatch.scripts.helpers.deepmatcherdata import deepmatcherdata

def fillCols(row, col1, col2):
	if row[col1] == '':
		row[col1] = row[col2]
	else:
		row[col2] = row[col1]
	return row

if __name__ == '__main__':
	# Set seed for reproducible results
	numpy.random.seed(42)
	# Read data
	sellerProdData = pandas.read_csv(path.join(DATA_DIR, 'SellerProductsData_LedTv_20190426.csv'), 
					sep='\t',
					encoding='utf-8')
	sellerProdMapping = pandas.read_csv(path.join(DATA_DIR, 'SellerProductsMapping_LedTv_20190426.csv'), 
					sep='\t',
					encoding='utf-8')
	prodData = pandas.read_csv(path.join(DATA_DIR, 'Products_LedTv_20190426.csv'), 
					sep='\t',
					encoding='utf-8')
	cenejeAttributes = pandas.read_csv(path.join(DATA_DIR, 'CenejeAttributes_LedTv_20190426.csv'), 
					sep='\t',
					encoding='utf-8')
	# Join sellerProdData and sellerProdMapping
	integrated = pandas.merge(
		left=sellerProdData[['idSeller', 'idSellerProduct', 'brandSeller', 'nameSeller', 'descriptionSeller']],
		right=sellerProdMapping,
		how='inner', 
		on=['idSellerProduct', 'idSeller']
	)
	# Join result of previous join with prodData
	integrated = pandas.merge(
		left=integrated,
		right=prodData[['idProduct', 'nameProduct', 'brand']], 
		how='inner',
		on='idProduct'
	)
	# Fill empty brand and brandSeller from one another
	# since there're mapping
	# integrated[['brand', 'brandSeller']] = integrated[['brand', 'brandSeller']].apply(
	# 	lambda row: fillCols(row, 'brand', 'brandSeller'),
	# 	axis=1
	# )
	
	# Integration with ceneje attributes data
	# integrated_attr = pandas.merge(
	# 	left=integrated,
	# 	right=cenejeAttributes.drop_duplicates(['idProduct', 'idAtt'], keep='first')[['idProduct', 'nameAtt', 'attValue']],
	# 	how='left',
	# 	on='idProduct'
	# )
	# integrated_attr.to_csv(path.join(DATA_DIR, 'AttributesIntegratedProducts.tsv'), sep='\t')

	# Take only those records that have idProduct duplicated (as they match)
	matching = integrated[integrated.duplicated(subset='idProduct', keep=False)]
	# Clean rubbish HTML from text
	matching = preprocess.cleanHtml(matching, na_value='not_available')
	integrated.to_csv(path.join(DATA_DIR, 'IntegratedProducts.csv'), sep='\t')
	matching.to_csv(path.join(DATA_DIR, 'Matching.csv'), sep='\t')
	keys = ['idProduct', 'brandSeller', 'nameSeller', 'descriptionSeller', 'nameProduct', 'brand']
	init = time.time()
	deepdata = deepmatcherdata(
		matching, 
		group_cols=['idProduct'], 
		keys=keys,
		id_attr='id',
		left_attr='ltable',
		right_attr='rtable',
		label_attr='label',
		clean_html=False, 
		perc=.002
	)
	data = deepdata.deepdata
	print(time.time() - init)
	data.to_csv(path.join(DATA_DIR, 'deepmatcher.csv'))
	train, val, test = deepdata.train_val_test_split([0.6, 0.2, 0.2])
	unlabeled_data = train.iloc[len(train) * 0.2:]
	train = train.iloc[len(train) * 0.2:]
	train.to_csv(path.join(DATA_DIR, 'train.csv'))
	val.to_csv(path.join(DATA_DIR, 'validation.csv'))
	test.to_csv(path.join(DATA_DIR, 'test.csv'))
	unlabeled_data.to_csv(path.join(DATA_DIR, 'unlabeled.csv'))