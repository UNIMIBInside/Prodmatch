import html
import numpy
import re
import time
from pandas import pandas
from bs4 import BeautifulSoup
from itertools import combinations, chain, product


def strip(df: pandas.DataFrame, fillna=True, fillna_value=''):
	""" 
	Apply colStrip function to all object columns, i.e. all columns containg string values
	
	Parameters
	----------
	df (pandas.DataFrame): DataFrame to clean\n
	fillna (bool): wheter or not call pandas.Series.fillna(fillna_value)\n
	fillna_value (str): value to replace na

	Returns
	-------
	pandas.DataFrame: DataFrame with all object columns cleaned
	"""
	# Get the columns containing object values (strings)
	df_obj_cols = df.dtypes[df.dtypes == 'object'].index.tolist()
	if df_obj_cols == []:
		return df
	if fillna:
		df[df_obj_cols] = df[df_obj_cols].fillna(fillna_value)
	# Apply to all object columns col_strip_clean function
	# df[df_obj_cols] = df[df_obj_cols].apply(lambda col: colStrip(col, not(fillna), fillna_value), axis=1)
	# This version speed up performance using numpy.vectorize
	df[df_obj_cols] = df[df_obj_cols].apply(numpy.vectorize(lambda x: ' '.join(x.strip())))
	return df


def colStrip(col: pandas.Series, fillna=True, fillna_value=''):
	""" 
	Strip leading and trailing whitespaces and remove exceeding ones

	Parameters
	----------
	col (pandas.Series): dataframe column to clean\n
	fillna (bool): wheter or not call pandas.Series.fillna(fillna_value)\n
	fillna_value (str): value to replace na

	Returns
	-------
	pandas.Series: cleaned column
	"""
	if col.dtype != 'object':
		raise Exception('Strip and clean whitespaces works only with object dtype')
	if fillna:
		col = col.fillna(fillna_value)
	col = col.apply(lambda row: ' '.join(row.split()))
	return col

def strCleanHtml(s: str, strip: bool):
	"""
	Helper function, it cleans up HTML rubbish from string s

	Parameters
	----------
	s (str): string to be cleaned\n
	strip (bool): wheter or not strip leading and trailing whitespaces and remove exceeding ones\n

	Returns
	-------
	cleaned string
	"""
	s = re.sub(r'(&)(\d+)', r'\1#\2', s)
	# I don't know why it has to be called two times
	s = html.unescape(html.unescape(s))
	if strip:
		s = ' '.join(BeautifulSoup(s, 'lxml').get_text(separator=u' ').split())
	else:
		s = BeautifulSoup(s, 'lxml').get_text(separator=u' ')
	return s

def colCleanHtml(col: pandas.Series, strip=True, fillna=True, fillna_value=''):
	"""
	Generically clean HTML text from an object column

	Parameters
	----------
	col (pandas.Series): column to be cleaned\n
	strip (bool): wheter or not strip leading and trailing whitespaces and remove exceeding ones\n
	fillna (bool): wheter or not call pandas.Series.fillna(fillna_value)\n
	fillna_value (str): value to replace na

	Returns
	-------
	pandas.Series: HTML-cleaned column
	"""
	if fillna:
		col = col.fillna(fillna_value)
	# Insert # into strings like &189; otherwise html.escape does't work properly
	col = col.str.replace(r'(&)(\d+)', r'\1#\2')
	# Unescape two times (I don't know why exactly two times)
	# Convert strings like &lt; into <
	col = html.unescape(html.unescape(col))
	# Retrieve text inside html tags and separate it with a space
	# Remove exceeding whitespaces, since:
	# split() splits string by whitespaces, tabs, ... and ' '.join() concatenates them
	if strip:
		col = col.apply(lambda row: ' '.join(BeautifulSoup(row, 'lxml').get_text(separator=u' ').split()))
	else:
		col = col.apply(lambda row: BeautifulSoup(row, 'lxml').get_text(separator=u' '))
	# col = col.apply(numpy.vectorize(lambda x: strCleanHtml(x, strip)))
	return col

def cleanHtml(df: pandas.DataFrame, strip=True, fillna=True, fillna_value=''):
	""" 
	Apply strCleanHtml function to all object columns, i.e. all columns containg string values
	
	Parameters
	----------
	df (pandas.DataFrame): DataFrame to clean\n
	strip (bool): wheter or not strip leading and trailing whitespaces and remove exceeding ones\n
	fillna (bool): wheter or not call pandas.Series.fillna(fillna_value)\n
	fillna_value (str): value to replace na

	Returns
	-------
	pandas.DataFrame: DataFrame with all object columns cleaned
	"""
	# Get the columns containing object values (strings)
	df_obj_cols = df.dtypes[df.dtypes == 'object'].index.tolist()
	if df_obj_cols == []:
		return df
	if fillna:
		df[df_obj_cols] = df[df_obj_cols].fillna(fillna_value)
	# Apply to all object columns colCleanHtml function
	# df[df_obj_cols] = df[df_obj_cols].apply(lambda col: colCleanHtml(col, not(fillna), fillna_value), axis=1)
	df[df_obj_cols] = df[df_obj_cols].apply(numpy.vectorize(lambda x: strCleanHtml(x, strip)))
	return df

def fillCols(row, col1, col2):
	if row[col1] == '':
		row[col1] = row[col2]
	else:
		row[col2] = row[col1]
	return row

def pairUp(row, df, keys, perc):
	x = []
	x.append(row[keys].values)
	# Retrieve all products not matching with the one in row['idProduct']
	non_match = df.loc[ df.idProduct != row['idProduct']]
	how_many = int(len(non_match) * perc)
	if how_many == 0 or how_many > len(non_match):
		raise Exception('Can\'t sample items. Requested ' + str(how_many) + ', sampleable: ' + str(len(non_match)))
	# print(len(non_match), how_many)
	# Create pair (prod, prod non matching) for every product sampled from non_match DataFrame
	return product(x, non_match.sample(how_many)[keys].values)

def prepareDeepmatcherData(df: pandas.DataFrame, group_cols, keys: list, perc=.75): 
	"""
	Deepmatcher needs data in a particular way, such as:
	|Label|Left product attributes|Right product attributes|
	where Label is 'Match' or 'Not match' and the same attributes for both products

	Parameters
	----------
	df: dataset from which data will be create\n
	group_cols: column(s) to group by on
	keys: list with attributes name
	perc: how many non-matching tuples will be created for each product in percentage (0;1)

	Returns
	-------
	DataFrame in the form accepted by Deepmatcher
	"""
	if perc <= 0 or perc >= 1:
		raise Exception('Percentage must be between 1 and 99')
	left = ['left_' + key for key in keys]
	right = ['right_' + key for key in keys]
	# data = pandas.DataFrame(columns=['label'] + left + right)
	# matching = matching.sort_values(by='idProduct').reset_index()
	# matching.index = numpy.arange(0, len(matching))
	# print(matching[38:43])
	# last_row = -1
	# index_in_group = 0
	# init = time.time()
	# for i_row, _ in matching.iterrows():
	# 	if i_row == len(matching) - 1:
	# 		pass
	# 	else:
	# 		how_many_in_group = idProductGroupedSizes[matching.loc[i_row, 'idProduct']] - index_in_group
	# 		for j in range(1, how_many_in_group):
	# 			# data.loc[last_row + j, 'label'] = 'match'
	# 			data.loc[last_row + j, left] = matching.loc[i_row, keys].values
	# 			data.loc[last_row + j, right] = matching.loc[i_row + j, keys].values
	# 		last_row = last_row + how_many_in_group - 1
	# 		# print(data)
	# 		if matching.loc[i_row, 'idProduct'] == matching.loc[i_row + 1, 'idProduct']:
	# 			index_in_group += 1
	# 		else:
	# 			index_in_group = 0
	"""
	To create non-matching tuples, every product p will be paired up with a subset 
	sample at random from products different from p. That's essentially what pairUp method do
	"""
	non_match = df[[group_cols] + keys]\
		.apply(lambda row: pandas.Series(pairUp(row, df, keys, perc)), axis=1)\
		.stack()\
		.apply(lambda x: list(chain.from_iterable(x)))\
		.apply(pandas.Series)\
		.set_axis(labels=left + right, axis=1, inplace=False)\
		.reset_index(level=0, drop=True)
	non_match['label'] = 'non_match'
	"""
	To create matching tuples i group the dataframe by idProduct, and for every group
	i pair up products two by two (combinations)
	"""
	match = df.groupby(group_cols)[keys].apply(lambda x : combinations(x.values, 2))\
			.apply(pandas.Series)\
			.stack()\
			.apply(lambda x: list(chain.from_iterable(x)))\
			.apply(pandas.Series)\
			.set_axis(labels=left + right, axis=1, inplace=False)\
			.reset_index(level=0, drop=True)
	match['label'] = 'match'
	return pandas.concat([match, non_match])


if __name__ == '__main__':
	# Set seed for reproducible results
	numpy.random.seed(42)
	sellerProdData = pandas.read_csv('ceneje_data/SellerProductsData_LedTv_20190426.csv', 
					sep='\t',
					encoding='utf-8')
	sellerProdMapping = pandas.read_csv('ceneje_data/SellerProductsMapping_LedTv_20190426.csv', 
					sep='\t',
					encoding='utf-8')
	prodData = pandas.read_csv('ceneje_data/Products_LedTv_20190426.csv', 
					sep='\t',
					encoding='utf-8')
	cenejeAttributes = pandas.read_csv('ceneje_data/CenejeAttributes_LedTv_20190426.csv', 
					sep='\t',
					encoding='utf-8')
	# cenejeProdData = cleanHtml(cenejeProdData)
	# Join sellerProdData and sellerProdMapping
	integrated = pandas.merge(
		left=sellerProdData[['idSeller', 'idSellerProduct', 'brandSeller', 'nameSeller', 'descriptionSeller']],
		right=sellerProdMapping,
		how='inner', 
		on=['idSellerProduct', 'idSeller']
	)
	# print(integrated)
	# Join result of previous join with prodData
	integrated = pandas.merge(
		left=integrated,
		right=prodData[['idProduct', 'nameProduct', 'brand']], 
		how='inner',
		on='idProduct'
	)
	# Fill empty brand and brandSeller from one another
	# since there're mapping
	integrated[['brand', 'brandSeller']] = integrated[['brand', 'brandSeller']].apply(
		lambda row: fillCols(row, 'brand', 'brandSeller'),
		axis=1
	)
	
	# Integration with ceneje attributes data
	# integrated_attr = pandas.merge(
	# 	left=integrated,
	# 	right=cenejeAttributes.drop_duplicates(['idProduct', 'idAtt'], keep='first')[['idProduct', 'nameAtt', 'attValue']],
	# 	how='left',
	# 	on='idProduct'
	# )

	# Take only those records that have idProduct duplicated (as they match)
	matching = integrated[integrated.duplicated(subset='idProduct', keep=False)]
	# Clean rubbish HTML from text
	matching = cleanHtml(matching, fillna_value='not available')
	# print(matching.loc[matching.idProduct.isin(['9631030'])])
	# integrated.to_csv('ceneje_data/IntegratedProducts.csv', sep='\t')
	# matching.to_csv('ceneje_data/Matching.csv', sep='\t')
	keys = ['brandSeller', 'nameSeller', 'descriptionSeller', 'nameProduct', 'brand']	
	init = time.time()
	data = prepareDeepmatcherData(matching, 'idProduct', keys, .002)
	print(time.time() - init)
	data.to_csv('ceneje_data/deepmatcher.csv', sep='\t')
	# data.to_csv('ceneje_data/deppmatcher_data.csv', sep='\t')
	print(len(data))
	# data[['label']] = 'match'
	# print(data.head(40))
	# integrated_attr.to_csv('ceneje_data/AttributesIntegratedProducts.csv', sep='\t')
	# ofamostrano.to_csv('ceneje_data/AttributesAsColsIntegratedProducts.csv', sep='\t')
	# sellerProdData.get