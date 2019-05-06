import html
import numpy
import re
import time
from pandas import pandas
from bs4 import BeautifulSoup
from itertools import combinations, chain


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

# def mapping(sellerProdMapping: pandas.DataFrame):
# 	prodDict = {}
# 	uniqProd = {}
# 	for i, row in sellerProdMapping.iterrows():
# 		prodDict[str(row['idSellerProduct'])+":"+str(row['idSeller'])] = row['idProduct']
# 		if (row['idProduct'] in uniqProd):
# 			uniqProd[row['idProduct']].append(str(row['idSellerProduct'])+":"+str(row['idSeller']))
# 		else:
# 			uniqProd[row['idProduct']] = []
# 			uniqProd[row['idProduct']].append(str(row['idSellerProduct'])+":"+str(row['idSeller']))
# 	print(uniqProd)

if __name__ == '__main__':
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
	# mapping(sellerProdMapping)
	# cenejeProdData = cleanHtml(cenejeProdData)
	# Join prodData and sellerProdMapping
	integrated = pandas.merge(
		left=sellerProdData[['idSellerProduct', 'brandSeller', 'nameSeller', 'descriptionSeller']],
		right=sellerProdMapping,
		how='inner', 
		on='idSellerProduct'
	)
	# print(integrated)
	# Join pd_spm and sellerProdData
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
	integrated_attr = pandas.merge(
		left=integrated,
		right=cenejeAttributes.drop_duplicates(['idProduct', 'idAtt'], keep='first')[['idProduct', 'nameAtt', 'attValue']],
		how='left',
		on='idProduct'
	)
	# integrated = cleanHtml(integrated)
	# print(sellerProdData.descriptionSeller)
	# ofamostrano = pandas.concat([pd_spm_spd_attr, pandas.get_dummies(pd_spm_spd_attr[['nameAtt']])], 1).groupby(['idProduct', 'nameProduct', 'brand', 'idSellerProduct', 'brandSeller', 'descriptionSeller', 'attValue']).sum().reset_index()
	# print(len(integrated), len(integrated_attr))
	matching = integrated[integrated.duplicated(subset='idProduct', keep=False)]
	# matching.to_csv('ceneje_data/matching.csv', sep='\t')
	# idProductGrouped = matching.groupby(['idProduct'])
	# idProductGroupedSizes = idProductGrouped.size()
	# Get the exact same products 
	# I think that if there are some, maybe there could be an error in the mapping dataset
	# sameSeller = idProductGrouped.filter(lambda group: (group.idSeller.nunique() == 1 & group.idSellerProduct.nunique() == 1))
	# print(sameSeller)
	# matching = matching.drop(sameSeller.idProduct.index.tolist(), axis=0)
	matching = cleanHtml(matching, fillna_value='not available')
	# integrated.to_csv('ceneje_data/IntegratedProducts.csv', sep='\t')
	# matching.to_csv('ceneje_data/Matching.csv', sep='\t')
	keys = ['brandSeller', 'nameSeller', 'descriptionSeller', 'nameProduct', 'brand']
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
	# print(time.time() - init)
	init = time.time()
	data = matching.groupby('idProduct')[keys]\
			.apply(lambda x : combinations(x.values, 2))\
			.apply(pandas.Series)\
			.stack()\
			.apply(lambda x: list(chain.from_iterable(x)))\
			.apply(pandas.Series)\
			.set_axis(labels=left + right, axis=1, inplace=False)\
			.reset_index(level=0, drop=True)
	print(time.time() - init)
	print(data.head(50))
	# data[['label']] = 'match'
	# print(data.head(40))
	# integrated_attr.to_csv('ceneje_data/AttributesIntegratedProducts.csv', sep='\t')
	# ofamostrano.to_csv('ceneje_data/AttributesAsColsIntegratedProducts.csv', sep='\t')
	# sellerProdData.get