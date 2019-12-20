import requests
import json
import os
import json
import numpy
import torch
import argparse
import logging
import deepmatcher as dm
from os import path
from itertools import chain, product
from pandas import pandas
from collections import namedtuple, OrderedDict
from ceneje_prodmatch.src.helper import preprocess
from torch.utils.data import Dataset, DataLoader
from ceneje_prodmatch import DATA_DIR, DEEPMATCH_DIR, RESULTS_DIR, CACHE_DIR, CONFIG_DIR
from ceneje_prodmatch.src.matching.similarity import Similarity, SimilarityDataset, LogisticRegressionModel
from ceneje_prodmatch.src.matching.runner import Runner

logging.getLogger('deepmatcher.core')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
requests.get('https://reports.ceneje.si/smatch/FindMatches?idl3=971&input=ASUS%20prenosnik%20E203NA-FD026TS').content
"""

def pairUp(row: namedtuple, data: pandas.DataFrame):
    data = data.loc[data['right_brandSeller'] == getattr(row, 'left_brandSeller'), :]
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
    deepmatcher_cfg = cfg['deepmatcher']

    """
    Reading and normalizing data. Here I rename all the columns as
    the deepmatcher model I trained expect the columns it will process to be named
    as those used to train it
    """
    seller_offers = pandas.read_csv(
        path.join(DATA_DIR, 'experiments', offers_matching_cfg['seller_offers_data_name'] + '.csv'),
        sep=',',
        encoding='utf-8',
        dtype={'idSeller': object,'idSellerProduct': object}
    )
    """ 
    Drop products that not have corresponding unique product in the ground truth
    idSeller    idSellerProduct
    342         736740
    438         1339406
    """
    seller_offers=seller_offers[(seller_offers['idSeller']!='342') & (seller_offers['idSellerProduct']!='736740')]
    seller_offers=seller_offers[(seller_offers['idSeller']!='438') & (seller_offers['idSellerProduct']!='1339406')]

    seller_offers = pandas.concat([seller_offers[['idSeller', 'idSellerProduct',]],preprocess.normalize(
        seller_offers[['brandSeller', 'nameSeller']],
        fillna=True,
        na_value='',
        lower=True
    )], axis=1)
    old_seller_offers_cols = seller_offers.columns
    seller_offers.columns = ['left_' + col for col in seller_offers.columns]
    # print(seller_offers)

    ceneje_products = pandas.read_csv(
        path.join(DATA_DIR, 'experiments', offers_matching_cfg['ceneje_prod_data_name'] + '.csv'),
        sep=',',
        encoding='utf-8',
        dtype={'idProduct': object}
    )
    """
        Uncomment the following line if you want to use also the description
        provided by Ceneje
        ceneje_products[['idProduct', 'brand', 'nameProduct', 'descriptionProduct']]
    """
    """
        Comment the following line if you want to use also the description
        provided by Ceneje
    """
    # ceneje_products['descriptionProduct'] = 'ehnični podatki:<br> Barva: Črna<br> Diagonala zaslona (v cm): 6.8 cm<br> Diagonala zaslona (v palcih 1"=2,54cm)): 2.7 "<br> Goriščna razdalja (LOV): 6 - 18 mm<br> Ločljivost optičnega videa: 1280x720 pix<br> Ločljivost v pikslih: 18 MPix<br> Mera, globina: 23 mm<br> Mera, višina: 58 mm<br> Mera, širina: 95 mm<br> Napajanje: 2xMignon baterija - prosimo, naročite posebej<br> Podprte pomnilniške kartice: SD, SDHC<br> Teža: 105 g<br> Tip kamere: Digitalna kamera<br> Zoom (digitalni): 4 x<br> Zoom (optični): 3 xLastnosti:<br>  Digitalni fotoaparat Polaroid E-126 18 Mio. Pixel Opt. Zoom: 3xčrna<br>'
    # ceneje_products['descriptionProduct'] = ''
    ceneje_products = pandas.concat([ceneje_products[['idProduct']],preprocess.normalize(
        ceneje_products[['brand', 'nameProduct']],
        fillna=True,
        na_value='',
        lower=True
    )], axis=1)

    ceneje_products.rename(columns={'brand': 'brandSeller', 'nameProduct': 'nameSeller'}, inplace=True)
    # Use this if a combined attribute is used
    """ attributes_set = set(ceneje_products.columns)
    combine_attrs_set = set(['brandSeller', 'nameSeller'])
    assert(attributes_set.issuperset(combine_attrs_set))
    combined_attrs_name = 'combined_' + '_'.join(combine_attrs_set)
    attributes = list((attributes_set - combine_attrs_set)) + [combined_attrs_name]
    ceneje_products[combined_attrs_name] = ceneje_products.loc[:, combine_attrs_set].apply(lambda x: ' '.join(x), axis=1)  # Combine attributes
    ceneje_products[combined_attrs_name] = ceneje_products.loc[:, combined_attrs_name].apply(lambda x: ' '.join(list(OrderedDict.fromkeys(x.split()))))
    ceneje_products = ceneje_products.drop(['brandSeller', 'nameSeller'], axis=1)  # Drop columns """

    ceneje_products.columns = ['right_idProduct'] + ['right_' + col for col in old_seller_offers_cols[2:]]
    # print(ceneje_products.columns)

    """
    Create the data that will be process by deepmatcher.
    Here I choose a subset of offers, namely the first n_offers, and couple each of them
    with all the possible products in its category.
    Unfortunately torchtext, the module deepmatcher is built upon, does not handle pandas dataframe,
    so one have to write that dataset to csv and load it after!  
    """
    unlabeled = pandas.DataFrame([
        chain.from_iterable([left_prod, right_prod])
        for row in seller_offers.itertuples(index=False)
        for left_prod, right_prod in pairUp(row, ceneje_products)
    ], columns=seller_offers.columns.tolist() + ceneje_products.columns.tolist())\
        .reset_index(drop=True).rename_axis('id', axis=0, copy=False)
    unlabeled.to_csv(path.join(RESULTS_DIR, 'offers.csv'))
    # unlabeled['label'] = 0
    print(unlabeled.columns)

    model = dm.MatchingModel(
        attr_summarizer=dm.attr_summarizers.RNN(
            word_contextualizer=dm.word_contextualizers.RNN(unit_type='gru')
        ),
        attr_condense_factor=1
    )
    model.load_state(
        path.join(RESULTS_DIR, 'models', cfg['deepmatcher']['train']['best_model_name'] + '.pth'),
        device=device)
    candidate = dm.data.process_unlabeled(
        # path.join(RESULTS_DIR, 'offers.csv'),
        unlabeled,
        trained_model=model,
        ignore_columns=['id','left_idSeller','left_idSellerProduct','right_idProduct','right_idSeller','right_idSellerProduct','right_brandSeller','left_brandSeller']) 
    predictions = model.run_prediction(candidate, output_attributes=True, device=device)
    print(predictions)
    predictions=predictions.astype({'left_idSeller':str,'left_idSellerProduct':str,'right_idProduct':str})
    predictions.to_csv(path.join(RESULTS_DIR, 'offers_preds.csv'))

    """ columns = deepmatcher_cfg['train']['left_right_ignore_cols']
    ignore_columns  = [deepmatcher_cfg['create']['left_prefix'] + col for col in columns]
    ignore_columns += [deepmatcher_cfg['create']['right_prefix'] + col for col in columns]
    if deepmatcher_cfg['create']['create_nm_mode'] == 'similarity':
        ignore_columns += ['similarity']

    unlabeled = SimilarityDataset(unlabeled, ignore_columns=ignore_columns)
    model = LogisticRegressionModel(input_dim=3)    
    predictions = model.run_predict(unlabeled, best_model_name='log_reg_adam_new_ledtv')
    unlabeled.data.insert(0, 'match_score', predictions)
    predictions = unlabeled.data """

    n_products = len(ceneje_products)
    best_prediction = dict()

    for num_offer in range(len(seller_offers)):
        offer=seller_offers.iloc[num_offer, :]
        """ offer_products = predictions.iloc[n_products * offer:n_products * (offer + 1), :]
        offer_products.sort_values(by=['match_score'], ascending=False)
        mask = offer_products['match_score'] >= offers_matching_cfg['match_score_thr']
        possible_matching_products = offer_products[mask] """
        """ print('Offer n.' + str(offer))
        print('Name: ', seller_offers.loc[offer, 'left_nameSeller'])
        print('Brand: ', seller_offers.loc[offer, 'left_brandSeller'])
        print('Possible matching products: ', possible_matching_products[['match_score', 'right_idProduct', 'right_nameSeller', 'right_brandSeller']].sort_values(by=['match_score'], ascending=False).values.tolist())
        """
        predictions_for_offer=predictions[(predictions['left_idSeller']==offer['left_idSeller']) & (predictions['left_idSellerProduct']==offer['left_idSellerProduct'])]
        mask=predictions_for_offer['match_score'] >= offers_matching_cfg['match_score_thr']
        possible_matching_products=predictions_for_offer[mask].sort_values(by=['match_score'], ascending=False).reset_index(drop=True)
        true_idProduct=gt[(gt['idSeller']==offer['left_idSeller']) & (gt['idSellerProduct']==offer['left_idSellerProduct'])]['idProduct'].values.tolist()[0]
        rank=possible_matching_products.reset_index()[(possible_matching_products['left_idSeller']==offer['left_idSeller']) & (possible_matching_products['left_idSellerProduct']==offer['left_idSellerProduct']) & (possible_matching_products['right_idProduct']==true_idProduct)].index.values.tolist()[0]
        best_prediction[
            offer['left_idSeller'] + ':' + offer['left_idSellerProduct']
        ] = {
            'name': offer['left_nameSeller'],
            'brand': offer['left_brandSeller'],
            'rank': rank,
            'possible_matching_products_length': len(possible_matching_products),
            'possible_matching_products': possible_matching_products[['match_score', 'right_idProduct', 'right_brandSeller', 'right_nameSeller']].values.tolist()
        }

    with open(path.join(RESULTS_DIR, offers_matching_cfg['best_predictions_name'] + '.json'), 'w') as fp:
        json.dump(best_prediction, fp)
    
