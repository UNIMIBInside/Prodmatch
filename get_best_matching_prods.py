import os
import math
import nltk
import json
import torch
import argparse
import logging
import deepmatcher as dm
import py_stringmatching as sm
import torch.optim as optim
from torch import nn
from os import path
from tqdm import tqdm
from itertools import chain, product
from pandas import pandas
from collections import namedtuple
from torch.utils.data import Dataset, DataLoader
from ceneje_prodmatch import DATA_DIR, DEEPMATCH_DIR, RESULTS_DIR, CACHE_DIR, CONFIG_DIR
from ceneje_prodmatch.src.helper import preprocess
from ceneje_prodmatch.src.matching.similarity import Similarity, SimilarityDataset, LogisticRegressionModel
from ceneje_prodmatch.src.matching.runner import Runner

logging.getLogger('deepmatcher.core')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pairUp(row: namedtuple, data: pandas.DataFrame):
    return product(
        [row],
        data.values.tolist()
    )

if __name__ == "__main__":

    seller_offers = pandas.read_csv(
        path.join(DATA_DIR, 'SellerProductsData_WashingMachine_20190515.csv'),
        sep='\t',
        encoding='utf-8',
        dtype={'idProduct': object, 'idSeller': object,'idSellerProduct': object}
    )
    seller_offers = preprocess.normalize(
        seller_offers[['brandSeller', 'nameSeller', 'descriptionSeller']],
        fillna=True,
        na_value='',
        lower=True
    )
    old_seller_offers_cols = seller_offers.columns
    seller_offers.columns = ['left_' + col for col in seller_offers.columns]
    print(seller_offers)

    ceneje_products = pandas.read_csv(
        path.join(DATA_DIR, 'Products_WashingMachine_20190515.csv'),
        sep='\t',
        encoding='utf-8',
        dtype={'idProduct': object, 'idSeller': object,'idSellerProduct': object}
    )
    ceneje_products = preprocess.normalize(
        ceneje_products[['idProduct', 'brand', 'nameProduct']],
        fillna=True,
        na_value='',
        lower=True
    )
    ceneje_products['description'] = ''
    ceneje_products.columns = ['right_idProduct'] + ['right_' + col for col in old_seller_offers_cols]
    print(ceneje_products)

    unlabeled = pandas.DataFrame([
        chain.from_iterable([left_prod, right_prod])
        for row in seller_offers.iloc[:10, :].itertuples(index=False)
        for left_prod, right_prod in pairUp(row, ceneje_products)
    ], columns=seller_offers.columns.tolist() + ceneje_products.columns.tolist())\
        .reset_index(drop=True).rename_axis('id', axis=0, copy=False)
    unlabeled.to_csv(path.join(RESULTS_DIR, 'offers.csv'))

    # print(unlabeled)

    model = dm.MatchingModel(
        attr_summarizer=dm.attr_summarizers.RNN(),
        attr_comparator='abs-diff'
    )
    model.load_state(
        path.join(RESULTS_DIR, 'models', 'rnn_pos_neg_fasttext_jaccard_new_cat_rand_model.pth'),
        device=device)
    candidate = dm.data.process_unlabeled(
        path.join(RESULTS_DIR, 'offers.csv'),
        trained_model=model,
        ignore_columns=['right_idProduct'])
    predictions = model.run_prediction(candidate, output_attributes=True, device=device)
    predictions.to_csv(path.join(RESULTS_DIR, 'offers_preds.csv'))
    print(predictions)

    """ parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', 
        '--model',
        default=os.path.join(RESULTS_DIR, 'model', 'best_model.pth'), 
        help='model path', 
        action='store'
    )
    parser.add_argument(
        '-s',
        '--seller-offers', 
        default='0.05', 
        type=str, 
        help='sampling interval in seconds', 
        action='store'
    )
    parser.add_argument(
        '-o', 
        '--output', 
        default=RESULTS_DIR, 
        type=str, 
        help='path to store results', 
        action='store'
    )
    parser.add_argument(
        '-f', 
        '--flatpak', 
        help='whether run octave from flatpak', 
        action='store_true'
    )
    parser.add_argument(
        '-r', 
        '--remove-old-results', 
        help='whether remove old results folder', 
        action='store_true'
    )
    args = parser.parse_args() """
    
