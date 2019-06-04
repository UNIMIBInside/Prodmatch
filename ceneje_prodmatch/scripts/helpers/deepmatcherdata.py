import numpy
from pandas import pandas
from tqdm import tqdm
from itertools import combinations, chain, product
from ceneje_prodmatch.scripts.helpers import preprocess

class DeepmatcherData(object):
    
    def __init__(self, 
            matching_tuples: pandas.DataFrame, 
            group_cols, 
            attributes: list,
            id_attr: str,
            label_attr: str,
            left_attr: str,
            right_attr: str,
            normalize=True, 
            create_non_match=True,
            perc=.75
    ):
        """
        Deepmatcher needs data in a particular way, such as:\n
        |Label|Left product attributes|Right product attributes|\n
        where Label is 'Match' or 'Not match' and the same attributes for both products.\n
        I suppose that the data are all the matching tuples, in the sense that every idSellerProduct
        is joined with the idProduct from ceneje, if so the algorithm works like this:
        * For creating the matching ('match', left prod, right prod) tuples, the original tuples will be grouped by
        group_cols (i think this would always be idProduct, but better be general), and for every group 
        there will be created combinations 
        * For creating the non matching ('non match', left prod, right prod) tuples, the original ones
        will be paired up with a subset sample at random from products different from it

        Parameters
        ----------
        matching_tuples: dataset possibly containing all the matching seller products (idSellerProduct) joined with
        the ceneje idProduct from which data will be create\n
        preprocess: wheater or not preprocess matching tuples data. It would be better if the preprocess
        takes place before creating data for deepmatcher, since data will grow on both rows and columns\n
        group_cols: column(s) to group by on\n
        attributes: list with attributes name to include in the output data\n
        create_non_match: wheater or not create non matching tuples\n
        perc: how many non-matching tuples will be created for each product in percentage (0;1)

        Returns
        -------
        DataFrame in the form accepted by Deepmatcher
        """
        if group_cols == []:
            raise Exception('group_cols must be a string or list indicating by which cols the data will be grouped by')
        if attributes == []:
            attributes = self.data.keys().values
        if perc <= 0 or perc >= 1:
            raise Exception('Percentage must be in (0;1) interval')
        if normalize:
            self.__data = preprocess.normalize(matching_tuples)
        self.__data = matching_tuples
        """ A = self.__data
        B = self.__data
        em.set_key(A, 'index')
        em.set_key(B, 'index')
        ab = em.AttrEquivalenceBlocker()
        C1 = ab.block_tables(A, B, 
                    l_block_attr='descriptionSeller', r_block_attr='descriptionSeller',
                    l_output_attrs=attributes,
                    r_output_attrs=attributes,
                    l_output_prefix='l_', r_output_prefix='r_')
        print(C1.head()) """
        self.__deeplabels = [left_attr + key for key in attributes] + [right_attr + key for key in attributes]
        self.matching = self.__getMatchingData(group_cols, attributes, label_attr)
        self.non_matching = None
        if create_non_match:
            self.non_matching = self.__getNonMatchingData(attributes, label_attr, perc)
        self.deepdata = self.__getDeepdata(id_attr)

    def __pairUp(self, row, attributes, perc):
        # Retrieve all products not matching with the one in row['idProduct']
        non_match = self.__data.loc[ self.__data.idProduct != row['idProduct']][attributes]
        how_many = int(len(non_match) * perc)
        if how_many == 0 or how_many > len(non_match):
            raise Exception('Can\'t sample items. Requested ' + str(how_many) + ', sampleable: ' + str(len(non_match)))
        # print(len(non_match), how_many)
        # Create pair (prod, prod non matching) for every product sampled from non_match DataFrame
        return product([row[attributes].values], non_match.sample(how_many).values)

    def __getNonMatchingData(self, attributes: list, label_attr: str, perc=.75):
        """
        To create non-matching tuples, every product p will be paired up with a subset 
        sample at random from products different from p. That's essentially what pairUp method do
        """
        non_match = self.__data[attributes]\
                        .apply(lambda row: pandas.Series(self.__pairUp(row, attributes, perc)), axis=1)\
                        .stack()\
                        .apply(lambda x: list(chain.from_iterable(x)))\
                        .apply(pandas.Series)\
                        .set_axis(labels=self.__deeplabels, axis=1, inplace=False)
        non_match[label_attr] = 0
        return non_match

    def __getMatchingData(self, group_cols, attributes: list, label_attr: str):
        """
        To create matching tuples i group the dataframe by idProduct, and for every group
        i pair up products two by two (combinations)
        """
        match = self.__data.groupby(group_cols)[attributes].apply(lambda x : combinations(x.values, 2))\
                    .apply(pandas.Series)\
                    .stack()\
                    .apply(lambda x: list(chain.from_iterable(x)))\
                    .apply(pandas.Series)\
                    .set_axis(labels=self.__deeplabels, axis=1, inplace=False)
        match[label_attr] = 1
        return match

    def __getDeepdata(self, id_attr):
        deepdata = pandas.concat([self.matching, self.non_matching])\
                    .reset_index(drop=True)
        return deepdata.rename_axis(id_attr, axis=0, copy=False)

    # def getData(self): 
    #     """
    #     Deepmatcher needs data in a particular way, such as:
    #     |Label|Left product attributes|Right product attributes|
    #     where Label is 'Match' or 'Not match' and the same attributes for both products.
    #     I suppose that the data are all the matching tuples, in the sense that every idSellerProduct
    #     is joined with the idProduct from ceneje, if so the algorithm works like this:
    #     * For creating the matching ('match', left prod, right prod) tuples, the original tuples will be grouped by
    #     group_cols (i think this would always be idProduct, but better be general), and for every group 
    #     there will be created combinations 
    #     * For creating the non matching ('non match', left prod, right prod) tuples, the original ones
    #     will be paired up with a subset sample at random from products different from it

    #     Returns
    #     -------
    #     DataFrame in the form accepted by Deepmatcher
    #     """
    #     # data = pandas.DataFrame(columns=['label'] + left + right)
    #     # matching = matching.sort_values(by='idProduct').reset_index()
    #     # matching.index = numpy.arange(0, len(matching))
    #     # print(matching[38:43])
    #     # last_row = -1
    #     # index_in_group = 0
    #     # init = time.time()
    #     # for i_row, _ in matching.iterrows():
    #     # 	if i_row == len(matching) - 1:
    #     # 		pass
    #     # 	else:
    #     # 		how_many_in_group = idProductGroupedSizes[matching.loc[i_row, 'idProduct']] - index_in_group
    #     # 		for j in range(1, how_many_in_group):
    #     # 			# data.loc[last_row + j, 'label'] = 'match'
    #     # 			data.loc[last_row + j, left] = matching.loc[i_row, attributes].values
    #     # 			data.loc[last_row + j, right] = matching.loc[i_row + j, attributes].values
    #     # 		last_row = last_row + how_many_in_group - 1
    #     # 		# print(data)
    #     # 		if matching.loc[i_row, 'idProduct'] == matching.loc[i_row + 1, 'idProduct']:
    #     # 			index_in_group += 1
    #     # 		else:
    #     # 			index_in_group = 0

    #     # deepdata = pandas.concat([self.matching, self.non_matching])\
    #     #             .reset_index(level=0)\
    #     #             .reset_index(level=0, drop=True)
    #     # return deepdata.rename_axis('id', axis=0, copy=False)
    #     return self.deepdata

    def train_val_test_split(self, splits: list, shuffle=True):
        """
        Split data into train, validation and test
        """
        assert(numpy.sum(splits) == 1)
        splits = numpy.asarray(splits)
        index_list = self.deepdata.index.tolist()
        if shuffle:
            numpy.random.shuffle(index_list)
        train, val, test = numpy.array_split(index_list, (splits[:-1].cumsum() * len(index_list)).astype(int))
        return self.deepdata.loc[train, :], self.deepdata.loc[val, :], self.deepdata.loc[test, :]
