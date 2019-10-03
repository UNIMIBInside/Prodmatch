import os
import numpy
import pandas
from os import path
from collections import Counter
from ceneje_prodmatch import DATA_DIR, UNSPLITTED_DATA_DIR, DEEPMATCH_DIR, CONFIG_DIR
from ceneje_prodmatch.src.helper import preprocess

def get_null_values_num_by_col(dataset: pandas.DataFrame, col: str, na_value=''):
    """
        Get the number of null values in the specified column

        Parameters
        ----------

        dataset (pandas.DataFrame): the dataset to search in
        col (str): a string specifing the column 

        Returns
        -------

        number of null values
    """
    assert(col in dataset.columns)
    return dataset[col].isna().sum()

def get_attr_avg_var_len(dataset: pandas.DataFrame, col: str):
    """
        Get the avarage and variance length of the elements in dataset specified column

        Parameters
        ----------

        dataset (pandas.DataFrame): the dataset to search in
        col (str): a string specifing the column 

        Returns
        -------

        tuple containing (average, variance)
    """
    assert(col in dataset.columns)
    lengths = dataset[col].apply(lambda x: numpy.nan if pandas.isna(x) else len(x))
    # mean = lengths.sum() / (len(dataset) - dataset[col].isna().sum())
    # print(mean, numpy.nanmean(lengths))
    # var = lengths.apply(lambda x: (x - mean) ** 2).sum() / (len(dataset) - dataset[col].isna().sum())
    # print(var, numpy.nanvar(lengths))
    return numpy.nanmean(lengths), numpy.nanvar(lengths)

def get_most_frequent_words(dataset: pandas.DataFrame, col: str, top_n=10):
    """
        Get the most frequent word in the specified column

        Parameters
        ----------

        dataset (pandas.DataFrame): the dataset to search in
        col (str): a string specifing the column
        top_n (int or str): get at most `top_n` results.
            If top_n is `all` then all the words will be returned

        Returns
        -------

        The top_n frequent words
    """
    if isinstance(top_n, int) and top_n > 0:
        return pandas.DataFrame(   
            pandas.Series(' '.join(dataset[col]).split()).value_counts(),
            columns=['Count']
        ).rename_axis('Word').iloc[:top_n, :]
    elif isinstance(top_n, str) and top_n == 'all':
       return pandas.DataFrame(   
            pandas.Series(' '.join(dataset[col]).split()).value_counts(),
            columns=['Count']
        ).rename_axis('Word')
    else:
        raise Exception('top_n must be an integer greater than 0 or a the string \'all\'') 



if __name__ == '__main__':
    dataset = pandas.read_csv(
        path.join(DATA_DIR, 'SellerProductsData_LedTv_20190426.csv'), 
        sep='\t', 
        encoding='utf-8', 
        na_values='')
    print(get_null_values_num_by_col(dataset, 'brandSeller'))
    print(get_attr_avg_var_len(dataset, 'brandSeller'))
    dataset = preprocess.normalize(dataset)
    print(get_most_frequent_words(dataset, 'descriptionSeller'))