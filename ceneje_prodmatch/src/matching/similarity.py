import time
import torch
import py_stringmatching as sm
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from pandas import pandas
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from os import path
from .runner import Runner
from ... import DATA_DIR, DEEPMATCH_DIR, RESULTS_DIR, CACHE_DIR


class Similarity(object):

    """
        With this class one can compute distance or similarity measure, based on functions
        provided by the `py_stringmatching` package.
        It requires a dataset with a schema similar to the one for deepmatcher:
        |Left attr 1|Left attr 2|...|Right attr 1|Right attr 2|...|Other attrs|

        Parameters
        ----------
        data (pandas.DataFrame): 
            dataset on which compute similarity scores for each tuple
        left_prefix (str): 
            string prefix for recognize the left product attributes
        right_prefix (str): 
            string prefix for recognize the left product attributes
        ignore_columns (list): 
            list of attributes to ignore in the similarity computation
        na_value: 
            value to fill NaN with
    """

    def __init__(
        self,
        data: pandas.DataFrame,
        left_prefix='left_',
        right_prefix='right_',
        ignore_columns=[],
        na_value='',
    ):
        self.data = data
        self.left_prefix = left_prefix
        self.right_prefix = right_prefix
        self.ignore_columns = ignore_columns
        self.na_value = na_value
        self.left_prod_attrs = [
            col for col in data
            if col.startswith(left_prefix) and col not in ignore_columns
        ]
        assert(len(data[self.left_prod_attrs]) > 0)
        self.right_prod_attrs = [
            col for col in data
            if col.startswith(right_prefix) and col not in ignore_columns
        ]
        assert(len(data[self.right_prod_attrs]) > 0)
        assert(len(data[self.left_prod_attrs].keys()) ==
               len(data[self.right_prod_attrs].keys()))
        self.data[self.left_prod_attrs + self.right_prod_attrs] = self.data[self.
                                                                            left_prod_attrs + self.right_prod_attrs].fillna(value=na_value)

    def get_scores(
        self,
        metric=sm.Jaccard(),
        tokenizer=sm.QgramTokenizer(return_set=True),
        similarity=True,
        undefined_scores=None,
        weights=None
    ):
        """
        Method to compute scores: for every left and right product in every tuple,
        it calls on the metric object the method get_sim_score(left, right), where 
        left and right get tokenized by tokenizer.tokenize() method.
        It only support similarity measure.
        For a complete review take a look at https://anhaidgroup.github.io/py_stringmatching/v0.4.1/index.html 

        Parameters
        ----------

        metric (str): 
            which metric to use: it has to be a py_stringmatching class. 
            For a complete review please look at `py_stringmatching` package
        tokenizer (str): 
            which `py_stringmatching` tokenizer use to tokenize text
        similarity (bool): 
            compute similarity score if True, disteance otherwise
        undefined_scores (list): 
            a list that contains, for each positions, the default scores to give to a pair of
            attributes if one of them contains `na_value`. If None every scores will be set
            to 1/2 (better not decide)
        weights (list): 
            list of weights for each of the attributes. 
            If None, every attribute will be equally weighted

        Returns
        -------
        A new copy of the data with a new column 'match_score' with the computed scores
        """
        def scores(row_left, row_right):
            scores = []
            for i in range(len(row_left)):
                if row_left[i] == self.na_value or row_right[i] == self.na_value:
                    scores.append(undefined_scores[i])
                else:
                    if tokenizer is None:
                        scores.append(metric.get_sim_score(
                            row_left[i], row_right[i]))
                    else:
                        scores.append(metric.get_sim_score(tokenizer.tokenize(
                            row_left[i]), tokenizer.tokenize(row_right[i])))
            return sum([scores[i] * weights[i] for i in range(len(scores))]) / len(scores)

        left_prefixs_num = len(self.data[self.left_prod_attrs].keys())
        right_prefixs_num = len(self.data[self.right_prod_attrs].keys())
        if metric is None:
            metric = sm.Jaccard()
        # if tokenizer is None:
        #     tokenizer = sm.QgramTokenizer(return_set=True)
        if weights is None:
            weights = [1] * left_prefixs_num
        if undefined_scores is None:
            undefined_scores = [1/2] * left_prefixs_num
        assert(left_prefixs_num == right_prefixs_num ==
               len(weights) == len(undefined_scores))

        tqdm.pandas()
        match_score = self.data[self.left_prod_attrs + self.right_prod_attrs]\
            .progress_apply(
            lambda row: scores(row[self.left_prod_attrs],
                               row[self.right_prod_attrs]),
            axis=1
        )
        new_data = self.data.copy()
        new_data.insert(0, 'match_score', match_score)
        return new_data


class SimilarityDataset(Dataset):

    """
        With this class one can encapsulate a torch.Dataset. It will be used to batch 
        attributes and compute a similarity vector between pair of attributes.
        It requires a dataset with a schema similar to the one for deepmatcher:
        |Left attr 1|Left attr 2|...|Right attr 1|Right attr 2|...|Other attrs|

        Parameters
        ----------
        data (pandas.DataFrame): 
            dataset on which compute similarity scores for each tuple
        metric (str): 
            which metric to use: it has to be a py_stringmatching class.
            For a complete review please look at `py_stringmatching` package
        tokenizer (str): 
            which `py_stringmatching` tokenizer use to tokenize text
        label_attr (str):
            string for recognize the label attribute
        left_prefix (str): 
        string prefix for recognize the left product attributes
        right_prefix (str):
            string prefix for recognize the left product attributes
        undefined_scores (list): 
            a list that contains, for each positions, the default scores to give to a pair of attributes if one of them contains `na_value`. 
            If None every scores will be set to 1/2 (better not decide)
        ignore_columns (list):
            list of attributes to ignore in the similarity computation
        na_value:
            value to fill NaN with
    """

    def __init__(
        self,
        data: pandas.DataFrame,
        metric=sm.Jaccard(),
        tokenizer=sm.QgramTokenizer(),
        label_attr='label',
        left_prefix='left_',
        right_prefix='right_',
        undefined_scores=None,
        ignore_columns=[],
        na_value='',
    ):
        self.data = data
        self.metric = metric
        self.tokenizer = tokenizer
        self.label_attr = label_attr
        self.left_prefix = left_prefix
        self.right_prefix = right_prefix
        self.ignore_columns = ignore_columns
        self.na_value = na_value
        self.left_prod_attrs = [
            col for col in data
            if col.startswith(left_prefix) and col not in ignore_columns
        ]
        assert(len(data[self.left_prod_attrs]) > 0)
        self.right_prod_attrs = [
            col for col in data
            if col.startswith(right_prefix) and col not in ignore_columns
        ]
        assert(len(self.right_prod_attrs) > 0)
        assert(len(self.left_prod_attrs) == len(self.right_prod_attrs))
        self.undefined_scores = undefined_scores
        if self.undefined_scores is None:
            self.undefined_scores = [1/2] * len(self.left_prod_attrs)
        assert(len(self.undefined_scores) == len(self.left_prod_attrs))
        self.data[self.left_prod_attrs + self.right_prod_attrs] = self.data[self.
                                                                            left_prod_attrs + self.right_prod_attrs].fillna(value=na_value)

    def __scores(self, row_left, row_right):
        scores = []
        for i in range(len(row_left)):
            if row_left[i] == self.na_value or row_right[i] == self.na_value:
                scores.append(self.undefined_scores[i])
            else:
                if self.tokenizer is None:
                    scores.append(self.metric.get_sim_score(
                        row_left[i], row_right[i]))
                else:
                    scores.append(
                        self.metric.get_sim_score(
                            self.tokenizer.tokenize(row_left[i]),
                            self.tokenizer.tokenize(row_right[i])))
        return scores

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row_left = self.data.iloc[index, [
            self.data.columns.get_loc(col) for col in self.left_prod_attrs]]
        row_right = self.data.iloc[index, [
            self.data.columns.get_loc(col) for col in self.right_prod_attrs]]
        label = self.data.iloc[index,
                               self.data.columns.get_loc(self.label_attr)]
        scores = torch.tensor(self.__scores(row_left, row_right))
        return (scores, label)


class LogisticRegressionModel(nn.Module):

    def __init__(self, input_dim, output_dim=2, bias=True):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.linear(x)
        out = self.log_softmax(out)
        return out

    def run_train(self, *args, **kwargs):
        """
        Train the LogisticRegressionModule model on the train_dataset; val_dataset is used
        to evaluate the module and save the best one based on one of the following statistics: 
        accuracy, recall, precision, F1.
        It's recommended using NLLLoss as criterion, the module used to compute loss, since
        the LogisticRegressionModule has a LogSoftmax layer as non linear output layer.
        pos_neg_ratio specifies how imbalanced train_dataset is, and could be calculated as follows:
        weight_pos * #pos = weight_neg * #neg,
        weight_pos / weight_neg = #neg / #pos,
        so it represents the positive weight wrt negative weight

        Parameters
        ----------

        train_dataset: 
            train dataset
        val_dataset: 
            dataset used to validate the model and save the best one
        model: 
            model to train
        resume=False: 
            whether to resume training on the last best model saving
        criterion=None: 
            which type of loss will be computed. By default it's used NLLLoss with weight 
            specified by the pos_neg_ratio (weight=[1/pos_neg_ratio, 1] if pos_neg_ratio >= 1; weight=[1, 1/pos_neg_ratio],
            where the first element is the weight of the '0' class)
            optimizer=None: how to compute backpropagation. By default will be used SDG with a learning rate equals to 0.01
            and a 0.9 momentum
        scheduler=None: 
            a scheduler to drop learning rate. By default the learning rate will be dropped
            based on the F1 measure on validation, if for 5 epochs won't upgrade
        train_epochs=10: 
            train epochs
        pos_neg_ratio=1: 
            ratio of positive examples weight wrt negative examples weight
        best_model_name='best_model': 
            name of the best model
        best_save_on='F1': 
            statistics on which the best model will be saved. It must be one of the following:
            'accuracy', 'precision', 'recall', 'f1'
        best_save_path=None: 
            path where the best model will be saved.
        device=None: 
            device to run train and validation.
        batch_size=32: 
            batch size
        **kwargs
        In the **kwargs you can specify the log frequency of the statistics

        Returns
        -------

        The best statistics
        """
        return Runner.train(*args, **kwargs)

    def run_predict(self, *args, **kwargs):
        """
        Run model on a dataset, return the scores predictions

        Parameters
        ----------

        dataset: 
            dataset that will be evaluated
        model: 
            model evaluate
        load_best_model=True: 
            whether to load the best evaluated model.
        best_model_name='best_model': 
            name of the best model.
        best_save_path=None: 
            path where the best model will be saved.
        device=None:
            device to run train and validation.
        batch_size=32: 
            batch size
        **kwargs
        In the **kwargs you can specify the log frequency of the statistics

        Returns
        -------

        Return predictions scores on dataset
        """
        return Runner.predict(model=self, *args, **kwargs)
