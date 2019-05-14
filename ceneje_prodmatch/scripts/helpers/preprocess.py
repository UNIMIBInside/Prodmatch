import html
import numpy
import re
import time
import string
from os import path
from pandas import pandas
from bs4 import BeautifulSoup
from ceneje_prodmatch import DATA_DIR

with open(path.join(DATA_DIR, 'slovenian-stopwords.txt'), 'r') as f:
    stop_words = f.read().split()

"""
Helper functions used to preprocess data.
For now it only works with objects (textual) data, but it can be enriched
with methods to preprocess whatever dtypes
"""

def strip(df: pandas.DataFrame, fillna=True, na_value=''):
    """ 
    Apply strip function to all object columns in DataFrame, i.e. all columns containg string values
    
    Parameters
    ----------
    df (pandas.DataFrame): DataFrame to clean\n
    fillna (bool): wheter or not call pandas.Series.fillna(na_value)\n
    na_value (str): value to replace na

    Returns
    -------
    pandas.DataFrame: DataFrame with all object columns cleaned
    """
    # Get the columns containing object values (strings)
    df_obj_cols = df.dtypes[df.dtypes == 'object'].index.tolist()
    if df_obj_cols == []:
        return df
    if fillna:
        df[df_obj_cols] = df[df_obj_cols].fillna(na_value)
    # Apply to all object columns col_strip_clean function
    # df[df_obj_cols] = df[df_obj_cols].apply(lambda col: colStrip(col, not(fillna), na_value), axis=1)
    # This version speed up performance using numpy.vectorize
    df[df_obj_cols] = df[df_obj_cols].apply(numpy.vectorize(lambda x: ' '.join(x.split())))
    return df


def strip(col: pandas.Series, fillna=True, na_value=''):
    """ 
    Strip leading and trailing whitespaces and remove exceeding ones, i.e. 
    between two strings, all but one whitespace will be cleaned

    Parameters
    ----------
    col (pandas.Series): dataframe column to clean\n
    fillna (bool): wheter or not call pandas.Series.fillna(na_value)\n
    na_value (str): value to replace na

    Returns
    -------
    pandas.Series: cleaned column
    """
    if col.dtype != 'object':
        raise Exception('Strip and clean whitespaces works only with object dtype')
    if fillna:
        col = col.fillna(na_value)
    col = col.apply(lambda row: ' '.join(row.split()))
    return col

def tolower(col: pandas.Series, fillna=True, na_value=''):
    """ 
    Strip leading and trailing whitespaces and remove exceeding ones, i.e. 
    between two strings, all but one whitespace will be cleaned

    Parameters
    ----------
    col (pandas.Series): dataframe column to clean\n
    fillna (bool): wheter or not call pandas.Series.fillna(na_value)\n
    na_value (str): value to replace na

    Returns
    -------
    pandas.Series: cleaned column
    """
    if col.dtype != 'object':
        raise Exception('tolower works only with object dtype')
    if fillna:
        col = col.fillna(na_value)
    col = col.apply(numpy.vectorize(str.lower))
    return col

def tolower(df: pandas.DataFrame, fillna=True, na_value=''):
    """ 
    Apply tolower function to all object columns in DataFrame, i.e. all columns containg string values
    
    Parameters
    ----------
    df (pandas.DataFrame): DataFrame to be lowered\n
    fillna (bool): wheter or not call pandas.Series.fillna(na_value)\n
    na_value (str): value to replace na

    Returns
    -------
    pandas.DataFrame: DataFrame with all object columns cleaned
    """
    # Get the columns containing object values (strings)
    df_obj_cols = df.dtypes[df.dtypes == 'object'].index.tolist()
    if df_obj_cols == []:
        return df
    if fillna:
        df[df_obj_cols] = df[df_obj_cols].fillna(na_value)
    # Apply to all object columns col_strip_clean function
    # df[df_obj_cols] = df[df_obj_cols].apply(lambda col: colStrip(col, not(fillna), na_value), axis=1)
    # This version speed up performance using numpy.vectorize
    df[df_obj_cols] = df[df_obj_cols].apply(numpy.vectorize(str.lower))
    return df

def __normalize(s: str):
    """
    Helper function, it cleans up HTML rubbish from string s, lower the case, 
    remove unnecessary whitespaces, stopwords and punctuation

    Parameters
    ----------
    s (str): string to be cleaned\n
 
    Returns
    -------
    cleaned string
    """
    s = re.sub(r'(&)(\d+)', r'\1#\2', s)
    # I don't know why it has to be called two times
    s = html.unescape(html.unescape(s))
    rules = str.maketrans('', '', string.punctuation)
    # s = ' '.join(BeautifulSoup(s, 'lxml').get_text(separator=u' ').split())
    s = ' '.join(
        [
            word for word in BeautifulSoup(s, 'lxml')\
                                .get_text(separator=u' ')\
                                .translate(rules)\
                                .lower()\
                                .split()
            if not word in stop_words
        ]
    )
    return s

def normalize(
        col: pandas.Series, 
        fillna=True, 
        na_value=''
    ):
    """
    Generically clean HTML text from an object column, lower the case, 
    remove unnecessary whitespaces, stopwords and punctuation

    Parameters
    ----------
    col (pandas.Series): column to be cleaned\n
    fillna (bool): wheter or not call pandas.Series.fillna(na_value)\n
    na_value (str): value to replace na

    Returns
    -------
    pandas.Series: HTML-cleaned column
    """
    if fillna:
        col = col.fillna(na_value)
    # Insert # into strings like &189; otherwise html.escape does't work properly
    col = col.str.replace(r'(&)(\d+)', r'\1#\2')
    # Unescape two times (I don't know why exactly two times)
    # Convert strings like &lt; into <
    col = html.unescape(html.unescape(col))
    # Retrieve text inside html tags and separate it with a space
    # Remove exceeding whitespaces, since:
    # split() splits string by whitespaces, tabs, ... and ' '.join() concatenates them
    rules = str.maketrans('', '', string.punctuation)
    # col = col.apply(lambda row: ' '.join(BeautifulSoup(row, 'lxml').get_text(separator=u' ')
    #                                 .split()))
    col = col.apply(lambda row: ' '.join(BeautifulSoup(row, 'lxml').get_text(separator=u' ')
                                    .translate(rules).split()))
    return col


def normalize(
        df: pandas.DataFrame,
        fillna=True, 
        na_value=''
    ):
    """ 
    Apply __noralize function to all object columns, i.e. all columns containg string values.
    Generically clean HTML text from a DataFrame, lower the case, 
    remove unnecessary whitespaces, stopwords and punctuation


    Parameters
    ----------
    df (pandas.DataFrame): DataFrame to clean\n
    fillna (bool): wheter or not call pandas.Series.fillna(na_value)\n
    na_value (str): value to replace na

    Returns
    -------
    pandas.DataFrame: DataFrame with all object columns cleaned
    """
    # Get the columns containing object values (strings)
    df_obj_cols = df.dtypes[df.dtypes == 'object'].index.tolist()
    if df_obj_cols == []:
        return df
    if fillna:
        df[df_obj_cols] = df[df_obj_cols].fillna(na_value)
    # Apply to all object columns colCleanHtml function
    # df[df_obj_cols] = df[df_obj_cols].apply(lambda col: colCleanHtml(col, not(fillna), na_value), axis=1)
    df[df_obj_cols] = df[df_obj_cols].apply(numpy.vectorize(lambda x: __normalize(x)))
    return df