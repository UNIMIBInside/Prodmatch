import os
import math
import json
import numpy
import time
import random
import deepmatcher as dm
from os import path
from pandas import pandas
from ceneje_prodmatch import DATA_DIR, UNSPLITTED_DATA_DIR, DEEPMATCH_DIR, CONFIG_DIR, UNLABELED_DIR
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

    if contains is None or contains == []:
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
                datasets[0][0].loc[:, ['idSeller', 'idSellerProduct'] + seller_prod_data_attrs]\
                .merge(
                    right=datasets[0][1],  # SellerProductsMapping_ dataset
                    how='inner',
                    on=['idSellerProduct', 'idSeller']
                ).merge(
                    right=(datasets[0][2].loc[datasets[0][2]['L3'] == category]).loc[:, ['idProduct'] + ceneje_prod_data_attrs],  # Products_ dataset
                    how='inner',
                    on='idProduct'
                ).reset_index(drop=True)
            )
        return integrated_dataset
    return [
        # SellerProductsData_ dataset
        datasets[i][0].loc[:, ['idSeller', 'idSellerProduct'] + seller_prod_data_attrs]\
        .merge(
            right=datasets[i][1],  # SellerProductsMapping_ dataset
            how='inner',
            on=['idSellerProduct', 'idSeller']
        ).merge(
            right=datasets[i][2].loc[:, ['idProduct'] + ceneje_prod_data_attrs],  # Products_ dataset
            how='inner',
            on='idProduct'
        ).reset_index(drop=True)
        for i in range(len(datasets))
    ]

def get_matching(integrated_data: list, normalize=True, normalize_attributes=None, **kwargs):
    """
    kwargs: 
        keyword arguments to pass to normalize function
    """
    # return [
    #     preprocess.normalize(
    #         # For every integrated dataset, keep only those products that are duplicates (matching)
    #         integrated_data[i].loc[integrated_data[i].duplicated(
    #             subset='idProduct', keep=False), :],
    #         **kwargs
    #     )
    #     if normalize
    #     else integrated_data[i].loc[integrated_data[i].duplicated(
    #             subset='idProduct', keep=False), :]
    #     for i in range(len(integrated_data))
    # ]
    matching = []
    for i in range(len(integrated_data)):
        # The following line is needed because deepmatcher expects tuple to be
        # <left_p, right_p, label>, so I need at least two matching products to create a pair
        duplicated_data = integrated_data[i].loc[integrated_data[i].duplicated(subset='idProduct', keep=False), :]
        if normalize:
            norm_data = preprocess.normalize(
                duplicated_data.loc[:, normalize_attributes], **kwargs
            )
            duplicated_data = duplicated_data.assign(**{normalize_attributes[i]: norm_data.loc[:, normalize_attributes[i]] for i in range(len(normalize_attributes))})
        matching.append(duplicated_data)
    return matching


def get_deepmatcher_data(matching_datasets: list, *args, drop_duplicates=False, drop_attributes=None, **kwargs):
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
    if not drop_duplicates:
        return pandas.concat([
            DeepmatcherData(
                matching,
                *args,
                **kwargs
            ).deepdata
            for matching in matching_datasets
        ]).reset_index(drop=True).rename_axis('id', axis=0, copy=False)
    else:
        return pandas.concat([
            DeepmatcherData(
                matching.drop_duplicates(subset=drop_attributes),
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
        folder=path.join(DATA_DIR, 'splitted' if cfg['default']['splitted_folder'] else ''),
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
        pandas.concat(integrated_data).to_csv(path.join(DEEPMATCH_DIR, 'experiments', '20_12_19', default_cfg['integrated_data_name'] + '.csv')) 
    
    # Remove from integrated data those offers that appears in unlabeled file
    # TODO: remove from all integrated files
    unlabeled_file_name = [file_name for file_name in os.listdir(UNLABELED_DIR) if str.lower(default_cfg['contains'][0]) in file_name]
    unlabeled = pandas.read_csv(path.join(UNLABELED_DIR, unlabeled_file_name[0]),  
                    dtype={'idProduct': object, 'idSeller': object,'idSellerProduct': object})
    integrated_data[0] = integrated_data[0]\
                            .loc[
                                integrated_data[0]\
                                    .merge(unlabeled, on=['idSeller', 'idSellerProduct', 'idProduct'], how='left', indicator=True)\
                                    ['_merge'] == 'left_only'
                            ]

    # Get matching tuples
    matching = get_matching(
        integrated_data, 
        normalize=preprocess_cfg['normalize'], 
        normalize_attributes=default_cfg['seller_prod_data_attrs'],
        lower=preprocess_cfg['lower'], 
        remove_brackets=False,
        remove_duplicated_words=preprocess_cfg['remove_duplicated_words']
    )
    if default_cfg['matching_data_to_csv']:
        pandas.concat(matching).to_csv(path.join(DEEPMATCH_DIR, 'experiments', '20_12_19', default_cfg['matching_data_name'] + '.csv'))

    # Create deepmatcher dataset
    deepdata = get_deepmatcher_data(
        matching,
        group_cols=deepmatcher_cfg['group_cols'],
        attributes=deepmatcher_cfg['attributes'],
        combine_attrs=deepmatcher_cfg['combine_attrs'],
        id_attr=deepmatcher_cfg['id_attr'],
        left_prefix=deepmatcher_cfg['left_prefix'],
        right_prefix=deepmatcher_cfg['right_prefix'],
        label_attr=deepmatcher_cfg['label_attr'],
        non_match_ratio=deepmatcher_cfg['non_match_ratio'],
        create_nm_mode=deepmatcher_cfg['create_nm_mode'],
        similarity_attr=deepmatcher_cfg['similarity_attr'],
        similarity_thr=deepmatcher_cfg['similarity_thr'],
        drop_duplicates=deepmatcher_cfg['drop_duplicates'], 
        drop_attributes=cfg['default']['seller_prod_data_attrs']
    )
    print(time.time() - init)

    # Remove from integrated data those offers that appears in deepmatcher test file
    # TODO: remove from all integrated files
    cols = ['idSeller', 'idSellerProduct', 'idProduct']
    left_cols = [deepmatcher_cfg['left_prefix'] + col for col in cols]
    right_cols = [deepmatcher_cfg['right_prefix'] + col for col in cols]

    test_file_name = [file_name for file_name in os.listdir(path.join(DEEPMATCH_DIR, 'experiments', '20_12_19')) if file_name.startswith('test') and str.lower(default_cfg['contains'][0]) in file_name]
    test = pandas.read_csv(path.join(path.join(DEEPMATCH_DIR, 'experiments', '20_12_19'), test_file_name[0]),  
                    dtype={key: object for key in left_cols+right_cols})
    deepdata = deepdata.loc[
                    deepdata\
                        .merge(test, on=left_cols+right_cols, how='left', indicator=True)\
                        ['_merge'] == 'left_only'
    ]

    if deepmatcher_cfg['deepmatcher_data_to_csv']:
        deepdata.to_csv(path.join(DEEPMATCH_DIR, 'experiments', '20_12_19', deepmatcher_cfg['deepmatcher_data_name'] + '.csv'))

    # Split into train, validation, test and unlabeled
    if split_cfg['split']:
        """ train, val, test = train_val_test_split(deepdata, split_cfg['train_val_test'])
        unlabeled_data = train[:math.ceil(len(train) * split_cfg['unlabeled'])]
        train = train[math.ceil(len(train) * split_cfg['unlabeled']) + 1:]
        
        train.to_csv(path.join(DEEPMATCH_DIR, split_cfg['train_data_name'] + '.csv'))
        val.to_csv(path.join(DEEPMATCH_DIR, split_cfg['val_data_name'] + '.csv'))
        test.to_csv(path.join(DEEPMATCH_DIR, split_cfg['test_data_name'] + '.csv'))
        unlabeled_data.to_csv(path.join(DEEPMATCH_DIR, split_cfg['unlabeled_data_name'] + '.csv')) """
        random.seed(42)
        random_state=42

        deepdata = deepdata.sample(frac=1, random_state=random_state)
        # unlabeled = deepdata.iloc[:math.ceil(len(deepdata) * split_cfg['unlabeled']), :]

        train, val = dm.data.split(
            deepdata, 
            None, None, None, None, 
            split_ratio=split_cfg['train_val'], 
            stratified=True,
            random_state=random.getstate()
        )
        # Drop duplicates
        # merged=pandas.merge(train, test, on=['right_nameSeller', 'left_nameSeller'], how='outer', indicator=True).query('_merge=="left_only"').iloc[:, :len(train.columns)]
        train.sample(frac=1, random_state=random_state).set_index(['id']).to_csv(path.join(DEEPMATCH_DIR, 'experiments', '20_12_19', split_cfg['train_data_name'] + '.csv'))
        val.sample(frac=1, random_state=random_state).set_index(['id']).to_csv(path.join(DEEPMATCH_DIR, 'experiments', '20_12_19', split_cfg['val_data_name'] + '.csv'))
        # test.sample(frac=1, random_state=random_state).set_index(['id']).to_csv(path.join(DEEPMATCH_DIR, 'experiments', '20_12_19', split_cfg['test_data_name'] + '.csv'))
        # unlabeled.sample(frac=1).to_csv(path.join(DEEPMATCH_DIR, 'experiments', '18_12_19', split_cfg['unlabeled_data_name'] + '.csv'))


