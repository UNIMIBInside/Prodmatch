import os
import json
import torch
import argparse
import logging
import deepmatcher as dm
from os import path
from itertools import chain, product
from pandas import pandas
from collections import namedtuple
from ceneje_prodmatch import DATA_DIR, RESULTS_DIR, CONFIG_DIR
from ceneje_prodmatch.src.helper import preprocess

logging.getLogger('deepmatcher.core')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pairUp(row: namedtuple, data: pandas.DataFrame):
    return product(
        [row],
        data.values.tolist()
    )

if __name__ == "__main__":

    # Import cfg
    with open(os.path.join(CONFIG_DIR, 'config.json')) as f:
        cfg = json.load(f)

    default_cfg = cfg['default']
    offers_matching_cfg = cfg['offers_matching']

    """
    Reading and normalizing data. Here I rename all the columns as
    the deepmatcher model I trained expect the columns it will process to be named
    as those used to train it
    """
    seller_offers = pandas.read_csv(
        path.join(DATA_DIR, offers_matching_cfg['seller_offers_data_name'] + '.csv'),
        sep='\t',
        encoding='utf-8',
        dtype={'idProduct': object, 'idSeller': object,'idSellerProduct': object}
    )
    seller_offers = preprocess.normalize(
        seller_offers[['idSeller', 'idSellerProduct', 'brandSeller', 'nameSeller', 'descriptionSeller']],
        fillna=True,
        na_value='',
        lower=True
    )
    old_seller_offers_cols = seller_offers.columns
    seller_offers.columns = ['left_' + col for col in seller_offers.columns]
    # print(seller_offers)

    ceneje_products = pandas.read_csv(
        path.join(DATA_DIR, offers_matching_cfg['ceneje_prod_data_name'] + '.csv'),
        sep='\t',
        encoding='utf-8',
        dtype={'idProduct': object, 'idSeller': object,'idSellerProduct': object}
    )
    """
        Uncomment the following line if you want to use also the description
        provided by Ceneje
        ceneje_products[['idProduct', 'brand', 'nameProduct', 'descriptionProduct']]
    """
    ceneje_products = preprocess.normalize(
        ceneje_products[['idProduct', 'brand', 'nameProduct']],
        fillna=True,
        na_value='',
        lower=True
    )
    """
        Comment the following line if you want to use also the description
        provided by Ceneje
    """
    ceneje_products['description'] = ''
    ceneje_products.columns = ['right_idProduct'] + ['right_' + col for col in old_seller_offers_cols[2:]]
    print(ceneje_products.columns)

    """
    Create the data that will be process by deepmatcher.
    Here I choose a subset of offers, namely the first n_offers, and couple each of them
    with all the possible products in its category.
    Unfortunately torchtext, the module deepmatcher is built upon, does not handle pandas dataframe,
    so one have to write that dataset to csv and load it after!  
    """
    unlabeled = pandas.DataFrame([
        chain.from_iterable([left_prod, right_prod])
        for row in seller_offers.iloc[:offers_matching_cfg['n_offers'], :].itertuples(index=False)
        for left_prod, right_prod in pairUp(row, ceneje_products)
    ], columns=seller_offers.columns.tolist() + ceneje_products.columns.tolist())\
        .reset_index(drop=True).rename_axis('id', axis=0, copy=False)
    unlabeled.to_csv(path.join(RESULTS_DIR, 'offers.csv'))
    print(unlabeled.columns)

    model = dm.MatchingModel(
        attr_summarizer=dm.attr_summarizers.RNN(),
        attr_comparator='abs-diff'
    )
    model.load_state(
        path.join(RESULTS_DIR, 'models', cfg['deepmatcher']['train']['best_model_name'] + '.pth'),
        device=device)
    candidate = dm.data.process_unlabeled(
        path.join(RESULTS_DIR, 'offers.csv'),
        trained_model=model,
        ignore_columns=['right_idProduct', 'left_idSeller', 'left_idSellerProduct'])
    predictions = model.run_prediction(candidate, output_attributes=True, device=device)
    # predictions.to_csv(path.join(RESULTS_DIR, 'offers_preds.csv'))

    n_products = len(ceneje_products)
    best_prediction = dict()

    for offer in range(offers_matching_cfg['n_offers']):
        offer_products = predictions.iloc[n_products * offer:n_products * (offer + 1), :]
        offer_products.sort_values(by=['match_score'], ascending=False)
        mask = offer_products['match_score'] >= offers_matching_cfg['match_score_thr']
        possible_matching_products = offer_products[mask]
        """ print('Offer n.' + str(offer))
        print('Name: ', seller_offers.loc[offer, 'left_nameSeller'])
        print('Brand: ', seller_offers.loc[offer, 'left_brandSeller'])
        print('Possible matching products: ', possible_matching_products[['match_score', 'right_idProduct', 'right_nameSeller', 'right_brandSeller']].sort_values(by=['match_score'], ascending=False).values.tolist())
        """
        best_prediction[
            seller_offers.loc[offer, 'left_idSeller'] + ':' + seller_offers.loc[offer, 'left_idSellerProduct']
        ] = {
            'name': seller_offers.loc[offer, 'left_nameSeller'],
            'brand': seller_offers.loc[offer, 'left_brandSeller'],
            'possible_matching_products': possible_matching_products[['match_score', 'right_idProduct', 'right_nameSeller', 'right_brandSeller']].sort_values(by=['match_score'], ascending=False).values.tolist()
        }

    with open(path.join(RESULTS_DIR, offers_matching_cfg['best_predictions_name'] + '.json'), 'w') as fp:
        json.dump(best_prediction, fp)
    
