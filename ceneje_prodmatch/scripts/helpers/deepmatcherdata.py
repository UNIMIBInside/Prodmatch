import numpy
import py_stringmatching as sm
from pandas import pandas
from tqdm import tqdm
from scipy.special import comb
from collections import namedtuple
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
                 normalize=False,
                 create_nm=True,
                 create_nm_mode='similarity',
                 similarity_attr=None,
                 metric=None,
                 tokenizer=None,
                 na_value='',
                 non_match_ratio=2,
                 similarity_thr=0.7
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
        matching_tuples: 
            dataset possibly containing all the matching seller products (idSellerProduct) joined with
            the ceneje idProduct from which data will be create\n
        group_cols: 
            column(s) to group by on\n
        attributes: 
            list with attributes name to include in the output data\n
        *_attr (str): 
            string that will be used to name columns
        normalize: 
            wheater or not preprocess matching tuples data. It would be better if the preprocess
            takes place before creating data for deepmatcher, since data will grow on both rows and columns\n
        create_nm (bool): wheater or not create non matching tuples\n
        create_nm_mode (str): string representing the non matching tuple creation mode: `similarity`
            uses a similarity function specified by `similarity` argument (default Jaccard with word tokenization);
            `random` picks non matching tuples at random\n
        similarity_attr (str): 
            attribute on which compute similarity
        metric (str): 
            which metric to use: it has to be a py_stringmatching class. 
            For a complete review please look at `py_stringmatching` package\n
        tokenizer (str): 
            which `py_stringmatching` tokenizer use to tokenize text\n
        non_match_ratio (float): 
            how many non-matching tuples will be created for each matching tuple of a product. So if for example
            there're 4 matching products, there will be created 4C2 = 6 combinations; so the number of non-matching tuples
            will be 6 * non_match_ratio
        similarity_thr (float): similarity threshold for creating non-matching tuples using a similarity function
            (to make deepmatcher work harder)

        Returns
        -------
        DataFrame in the form accepted by Deepmatcher
        """
        if group_cols == []:
            raise Exception(
                'group_cols must be a string or list indicating by which cols the data will be grouped by')
        if attributes == []:
            attributes = matching_tuples.keys().values
        if non_match_ratio <= 0:
            raise Exception('Percentage must be positive')
        if normalize:
            self.data = preprocess.normalize(matching_tuples)
        self.data = matching_tuples
        self.attributes = attributes
        """ 
        A = self.data
        B = self.data
        em.set_key(A, 'index')
        em.set_key(B, 'index')
        ab = em.AttrEquivalenceBlocker()
        C1 = ab.block_tables(A, B, 
                    l_block_attr='descriptionSeller', r_block_attr='descriptionSeller',
                    l_output_attrs=attributes,
                    r_output_attrs=attributes,
                    l_output_prefix='l_', r_output_prefix='r_')
        print(C1.head()) """
        self.__deeplabels = [
            left_attr + attr for attr in attributes] + [right_attr + attr for attr in attributes]
        self.matching = self.__getMatchingData(
            group_cols, label_attr)
        self.na_value = na_value
        self.non_match_ratio = non_match_ratio
        assert(create_nm_mode is not None)
        create_nm_mode = create_nm_mode.lower()
        assert(create_nm_mode == 'similarity' or create_nm_mode == 'random')
        self.create_nm_mode = create_nm_mode
        if self.create_nm_mode == 'similarity':
            if similarity_attr is None:
                raise Exception(
                    'You must specify one attribute on which similarity will be computed')
            elif similarity_attr not in self.data.columns:
                raise Exception(
                    'The attribute on which similarity will be computed must be a data column')
            if metric is None:
                metric = sm.Jaccard()
            if tokenizer is None:
                tokenizer = sm.QgramTokenizer()
            self.similarity_attr = similarity_attr
            self.metric = metric
            self.tokenizer = tokenizer
        self.similarity_thr = similarity_thr
        self.non_matching = None
        if create_nm:
            self.non_matching = self.__getNonMatchingData(label_attr)
        self.deepdata = self.__getDeepdata(id_attr)

    def __pairUp(self, row: namedtuple):
        def compute_sim_score(r):
            return self.metric.get_sim_score(
                self.tokenizer.tokenize(r[self.similarity_attr]),
                self.tokenizer.tokenize(getattr(row, self.similarity_attr)))

        # Retrieve all products not matching with the one in getattr(row, 'idProduct')
        match = len(self.data.loc[self.data['idProduct']
                                  == getattr(row, 'idProduct')])

        # How many non matching tuples will be created?
        how_many = int(comb(match, 2, exact=True) *
                       self.non_match_ratio / match)

        # non_match = self.data.loc[self.data['idProduct'] != getattr(row, 'idProduct'), attributes]
        # print('idProduct/Match/Non match: ' + str(getattr(row, 'idProduct')) + '/' + str(match) + '/' + str(how_many))
        # return product(
        #     [row[attributes].values],
        #     non_match.sample(how_many)[attributes].values
        # )

        non_match = self.data.loc[
            (self.data['idProduct'] != getattr(row, 'idProduct')) & (
                self.data[self.similarity_attr] != self.na_value),
            self.attributes
        ]
        if self.create_nm_mode == 'similarity':
            non_match['similarity'] = non_match.loc[:, [
                self.similarity_attr]].apply(compute_sim_score, axis=1)
            non_match = non_match.sort_values(
                by=['similarity'], ascending=False)
            mask = non_match['similarity'] >= self.similarity_thr
            simil = non_match[mask]
            not_simil = non_match[~mask]
            how_many_left = how_many - len(simil)

            if how_many_left > 0:
                return product(
                    [row],
                    pandas.concat(
                        [not_simil.iloc[: how_many_left, :],
                         simil]).values.tolist())
            else:
                return product(
                    [row],
                    simil.iloc[:how_many, :].values.tolist()
                )
        else:
            return product(
                [row],
                non_match.sample(how_many).values.tolist()
            )

    def __getNonMatchingData(self, label_attr: str):
        """
        To create non-matching tuples, every product p will be paired up with a subset 
        sample at random from products different from p. That's essentially what pairUp method do
        """
        print('Create non-matching data...')
        # tokenizer = sm.WhitespaceTokenizer()
        # non_match = self.data[attributes]\
        #                 .apply(lambda row: pandas.Series(self.__pairUp(row, attributes, metric, tokenizer)), axis=1)\
        #                 .stack()\
        #                 .apply(lambda x: list(chain.from_iterable(x)))\
        #                 .apply(pandas.Series)\
        #                 .set_axis(labels=self.__deeplabels, axis=1, inplace=False)
        non_match = pandas.DataFrame(
            [chain.from_iterable([left_prod, right_prod])
             for row in self.data[self.attributes].itertuples(index=False)
             for left_prod, right_prod in self.__pairUp(row)],
            columns=self.__deeplabels + ['similarity'])
        # non_match = pandas.DataFrame([
        #     chain.from_iterable([left_prod, right_prod])
        #     for pairs in self.data[attributes]
        #             .apply(lambda row: self.__pairUp(row, attributes, metric, tokenizer), axis=1)
        #     for left_prod, right_prod in pairs
        # ], columns=self.__deeplabels)
        non_match[label_attr] = 0
        print('Finished')
        return non_match

    def __getMatchingData(self, group_cols, label_attr: str):
        """
        To create matching tuples i group the dataframe by idProduct, and for every group
        i pair up products two by two (combinations)
        """
        print('Create matching data...')
        # match = self.data.groupby(group_cols)[attributes].apply(lambda x : combinations(x.values, 2))\
        #             .apply(pandas.Series)\
        #             .stack()\
        #             .apply(lambda x: list(chain.from_iterable(x)))\
        #             .apply(pandas.Series)\
        #             .set_axis(labels=self.__deeplabels, axis=1, inplace=False)
        match = pandas.DataFrame([
            chain.from_iterable([left_prod, right_prod])
            for idProduct, group in self.data[self.attributes].groupby(group_cols)
            for left_prod, right_prod in combinations(group.values, 2)
        ], columns=self.__deeplabels)
        match[label_attr] = 1
        print('Finished')
        return match

    def __getDeepdata(self, id_attr):
        self.matching.insert(8, 'similarity', 0)
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


def train_val_test_split(data: pandas.DataFrame, splits: list, shuffle=True):
    """
    Split data into train, validation and test
    """
    assert(numpy.sum(splits) == 1)
    splits = numpy.asarray(splits)
    index_list = data.index.tolist()
    if shuffle:
        numpy.random.shuffle(index_list)
    train, val, test = numpy.array_split(
        index_list, (splits[:-1].cumsum() * len(index_list)).astype(int))
    return data.loc[train, :], data.loc[val, :], data.loc[test, :]
