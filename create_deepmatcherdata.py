import os
import math
import json
import numpy
import time
from os import path
from pandas import pandas
from ceneje_prodmatch import DATA_DIR, UNSPLITTED_DATA_DIR, DEEPMATCH_DIR, CONFIG_DIR
from ceneje_prodmatch.src.helper import preprocess
from ceneje_prodmatch.src.helper.deepmatcherdata import DeepmatcherData, train_val_test_split

# def fillCols(row, col1, col2):
# 	if row[col1] == '':
# 		row[col1] = row[col2]
# 	else:
# 		row[col2] = row[col1]
# 	return row


def read_files_start_with(folder: str, prefix: str, contains=None, **kwargs):
    """
    Function that reads csv files, starting with a user defined prefix, from a specified folder
    If contains is not None, then the selected files are all that starts with prefix and 
    contains one string from contains

    Parameters
    ----------
    folder (str): 
        folder from which read files
    prefix (str): 
        pick files only starting with prefix
    contains (list): 
        list of strings that a file may contains in its name
    kwargs: 
        keyword arguments to pass to pandas.read_csv() function

    Returns
    -------
    List of pandas.DataFrame object, sorted by filename
    """

    if contains is None:
        prefixed = sorted([
            filename for filename in os.listdir(
            folder) if filename.startswith(prefix)
        ])
    else:
        prefixed = sorted([
            filename for filename in os.listdir(folder)
            if filename.startswith(prefix) and
            any(product for product in contains if product in filename)
        ])
    return [
        pandas.read_csv(
            path.join(folder, prefixed[i]),
            dtype={'idProduct': object, 'idSeller': object,
                   'idSellerProduct': object},
            **kwargs
        )
        for i in range(len(prefixed))
    ]


def read_files(
    folder: str, 
    prefixes: str, 
    contains=None, 
    **kwargs
):
    """
    Function that reads csv files, starting with a user defined prefixes, from a specified folder

    Parameters
    ----------
    folder (str): 
        folder from which read files
    prefixes (str): 
        pick files only starting with prefixes
    contains (list or None): 
        pick only files that contains `contains` after prefix.
        If is None, then I suppose files are splitted by category, and for each category
        there're three datasets: SellerProductsData_, SellerProductsMapping_ and Products_
    kwargs: 
        keyword arguments to pass to pandas.read_csv() function

    Returns
    -------
    List of tuples of pandas.DataFrame object. Every tuple contains n pandas.DataFrame objects, where n=len(prefixes),
    following the order specified by prefixes, ready for future ordered automatic joins 
    """

    files = [
        read_files_start_with(folder, prefixes[i], contains, **kwargs)
        for i in range(len(prefixes))
    ]
    return list(zip(*files))


def join_datasets(
    datasets: list, 
    seller_prod_data_attrs: list, 
    ceneje_prod_data_attrs: list,
    unsplitted=False, 
    L3=None
):
    """
    Since every join will merge SellerProductsData x SellerProductsMapping x Products (if i'm not wrong),
    this function executes those joins given a list of tuple of datasets, where every tuple contains dataset
    following SellerProductsData_, SellerProductsMapping_, Products_ order

    Parameters
    ----------
    datasets (list of tuples of pandas.DataFrame): 
        list that contains tuples of pandas.DataFrame.
        Every dataset in a tuples will be joined from left to right
    seller_prod_data_attrs (list):
        columns in output from seller products data dataset
    ceneje_prod_data_attrs (list):
        columns in output from ceneje products data dataset
    unsplitted (bool):
        Are the datasets in the folder splitted by category? 
        If not, then one must specify the L3 category, in order to automatize the process
        of dataset join
    L3 (list):
        list of L3 ids

    Returns
    -------
    integrated datasets, one per category
    """

    if unsplitted:
        assert(L3 is not None and isinstance(L3, list))
        integrated_dataset = []
        for category in L3:
            integrated_dataset.append(
                # SellerProductsData_ dataset
                datasets[0][0][['idSeller', 'idSellerProduct'] + seller_prod_data_attrs]\
                .merge(
                    right=datasets[0][1],  # SellerProductsMapping_ dataset
                    how='inner',
                    on=['idSellerProduct', 'idSeller']
                ).merge(
                    right=datasets[0][2][datasets[0][2]['L3'] == category][['idProduct'] + ceneje_prod_data_attrs],  # Products_ dataset
                    how='inner',
                    on='idProduct'
                ).reset_index(drop=True)
            )
        return integrated_dataset
    return [
        # SellerProductsData_ dataset
        datasets[i][0][['idSeller', 'idSellerProduct'] + seller_prod_data_attrs]\
        .merge(
            right=datasets[i][1],  # SellerProductsMapping_ dataset
            how='inner',
            on=['idSellerProduct', 'idSeller']
        ).merge(
            right=datasets[i][2][['idProduct'] + ceneje_prod_data_attrs],  # Products_ dataset
            how='inner',
            on='idProduct'
        ).reset_index(drop=True)
        for i in range(len(datasets))
    ]

def get_matching(integrated_data: list, normalize=True, **kwargs):
    """
    kwargs: 
        keyword arguments to pass to normalize function
    """
    return [
        preprocess.normalize(
            # For every integrated dataset, keep only those products that are duplicates (matching)
            integrated_data[i][integrated_data[i].duplicated(
                subset='idProduct', keep=False)],
            **kwargs
        )
        if normalize
        else integrated_data[i][integrated_data[i].duplicated(
                subset='idProduct', keep=False)]
        for i in range(len(integrated_data))
    ]


def get_deepmatcher_data(matching_datasets: list, *args, **kwargs):
    """
    Get the final deepmatcher data, ready to be processed by Deepmatcher framework

    Parameters
    ----------
    matching_datasets (list): 
        a list of matching products per category
    args:
        positional arguments to DeepmatcherData class
    kwargs:
        keyword arguments to DeepmatcherData class

    Returns:
    --------
    pandas.Dataframe containing the data to be processed by Deepmatcher
    """
    
    return pandas.concat([
        DeepmatcherData(
            matching,
            *args,
            **kwargs
        ).deepdata
        for matching in matching_datasets
    ]).reset_index(drop=True).rename_axis('id', axis=0, copy=False)


if __name__ == '__main__':

    # Import config
    with open(os.path.join(CONFIG_DIR, 'config.json')) as f:
        cfg = json.load(f)

    default_cfg = cfg['default']
    preprocess_cfg = cfg['preprocess']
    unsplitted_cfg = cfg['unsplitted']
    deepmatcher_cfg = cfg['deepmatcher']['create']
    split_cfg = cfg['split']

    init = time.time()
    # Join datasets
    unsplitted = unsplitted_cfg['unsplitted_data']
    if unsplitted:
        files = read_files(
            folder=UNSPLITTED_DATA_DIR,
            prefixes=default_cfg['prefixes'],
            contains=unsplitted_cfg['unsplitted_contains'],
            sep='\t',
            encoding='utf-8'
        )
        integrated_unsplitted_data = join_datasets(
            files, 
            default_cfg['seller_prod_data_attrs'], 
            default_cfg['ceneje_prod_data_attrs'], 
            unsplitted=unsplitted, 
            L3=unsplitted_cfg['L3_ids']
        )
    files = read_files(
        folder=DATA_DIR,
        prefixes=default_cfg['prefixes'],
        contains=default_cfg['contains'],
        sep='\t',
        encoding='utf-8'
    )
    integrated_data = join_datasets(
        files,
        default_cfg['seller_prod_data_attrs'], 
        default_cfg['ceneje_prod_data_attrs']
    )
    if unsplitted:
        integrated_data += integrated_unsplitted_data
    if default_cfg['integrated_data_to_csv']:
        pandas.concat(integrated_data).to_csv(path.join(DATA_DIR, default_cfg['integrated_data_name'] + '.csv')) 
    
    # Get matching tuples
    matching = get_matching(
        integrated_data, 
        normalize=preprocess_cfg['normalize'], 
        lower=preprocess_cfg['lower'], 
        remove_brackets=preprocess_cfg['remove_brackets']
    )
    if default_cfg['matching_data_to_csv']:
        pandas.concat(matching).to_csv(path.join(DATA_DIR, default_cfg['matching_data_name'] + '.csv'))

    # Create deepmatcher dataset
    deepdata = get_deepmatcher_data(
        matching,
        group_cols=deepmatcher_cfg['group_cols'],
        attributes=deepmatcher_cfg['attributes'],
        id_attr=deepmatcher_cfg['id_attr'],
        left_prefix=deepmatcher_cfg['left_prefix'],
        right_prefix=deepmatcher_cfg['right_prefix'],
        label_attr=deepmatcher_cfg['label_attr'],
        non_match_ratio=deepmatcher_cfg['non_match_ratio'],
        create_nm_mode=deepmatcher_cfg['create_nm_mode'],
        similarity_attr=deepmatcher_cfg['similarity_attr'],
        similarity_thr=deepmatcher_cfg['similarity_thr']
    )
    if deepmatcher_cfg['deepmatcher_data_to_csv']:
        deepdata.to_csv(path.join(DEEPMATCH_DIR, deepmatcher_cfg['deepmatcher_data_name'] + '.csv'))
    print(time.time() - init)
    
    # Split into train, validation, test and unlabeled
    if split_cfg['split']:
        train, val, test = train_val_test_split(deepdata, split_cfg['train_val_test'])
        unlabeled_data = train[:math.ceil(len(train) * split_cfg['unlabeled'])]
        train = train[math.ceil(len(train) * split_cfg['unlabeled']) + 1:]
        
        train.to_csv(path.join(DEEPMATCH_DIR, split_cfg['train_data_name'] + '.csv'))
        val.to_csv(path.join(DEEPMATCH_DIR, split_cfg['val_data_name'] + '.csv'))
        test.to_csv(path.join(DEEPMATCH_DIR, split_cfg['test_data_name'] + '.csv'))
        unlabeled_data.to_csv(path.join(DEEPMATCH_DIR, split_cfg['unlabeled_data_name'] + '.csv'))
