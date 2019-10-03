import os
import pandas
from os import path
from ceneje_prodmatch import DATA_DIR, UNSPLITTED_DATA_DIR, DEEPMATCH_DIR, CONFIG_DIR

def split_data(
    unsplitted_prods_filename: str,
    unsplitted_maps_filename: str,
    unplitted_sellers_filename: str,
    **kwargs):

    """
        Split data from ceneje into Products, SellerProductsData and SellerProductsMapping
        based on L3 id 

        Parameters
        ----------
        kwargs: 
            keyword arguments to pass to pandas.read_csv() function
    """
    unsplitted_prods = pandas.read_csv(
        path.join(UNSPLITTED_DATA_DIR, unsplitted_prods_filename),
        **kwargs
    )
    unsplitted_maps = pandas.read_csv(
        path.join(UNSPLITTED_DATA_DIR, unsplitted_maps_filename),
        **kwargs
    )
    unsplitted_sellers = pandas.read_csv(
        path.join(UNSPLITTED_DATA_DIR, unsplitted_sellers_filename),
        **kwargs
    )
    prods_len = 0
    sellers_len = 0
    if not path.exists(path.join(DATA_DIR, 'splitted')):
        os.makedirs(path.join(DATA_DIR, 'splitted'))
    for key, value in categories.items():
        prods = unsplitted_prods.loc[unsplitted_prods['L3'] == key]
        maps = prods.loc[:, ['idProduct']].merge(right=unsplitted_maps, how='inner', on='idProduct').loc[:, ['idProduct', 'idSeller', 'idSellerProduct']]
        sellers = maps.loc[:, ['idSeller', 'idSellerProduct']].merge(right=unsplitted_sellers, how='inner', on=['idSeller', 'idSellerProduct'])
        prods.to_csv(path.join(DATA_DIR, 'splitted', 'Products_' + value + '.csv'), sep='\t', encoding='utf-8')
        maps.to_csv(path.join(DATA_DIR, 'splitted', 'SellerProductsMapping_' + value + '.csv'), sep='\t', encoding='utf-8')
        sellers.to_csv(path.join(DATA_DIR, 'splitted', 'SellerProductsData_' + value + '.csv'), sep='\t', encoding='utf-8')
        prods_len += len(prods)
        sellers_len += len(sellers)
    print(sellers_len, len(unsplitted_sellers))

if __name__ == '__main__':
    categories = {
        873: "Digital_camera",
        930: "Summer_tires",
        931: "Winter_tires",
        1666: "Accumulator_drillers",
        971: "Laptops",
        1014: "Men_sneakers",
        174: "Cartridges"
    }
    unsplitted_prods_filename = 'Products_NewCategories_20190605.csv'
    unsplitted_maps_filename = 'SellerProductsMapping_NewCategories_20190605.csv'
    unsplitted_sellers_filename = 'SellerProductsData_NewCategories_20190605.csv'

    split_data(
        unsplitted_prods_filename, 
        unsplitted_maps_filename,
        unsplitted_sellers_filename, 
        sep='\t',
        encoding='utf-8'
    )