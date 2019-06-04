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

    def __init__(
        self, 
        data: pandas.DataFrame,
        left_attr='ltable_', 
        right_attr='rtable_', 
        ignore_columns=[],
        na_value='',
        ):
        """
        With this class one can compute distance or similarity measure, based on functions
        provided by the `py_stringmatching` package.
        It requires a dataset with a schema similar to the one for deepmatcher:
        |Left attr 1|Left attr 2|...|Right attr 1|Right attr 2|...|Other attrs|

        Parameters
        ----------
        data (pandas.DataFrame): dataset on which compute similarity scores for each tuple\n
        left_attr (str): string prefix for recognize the left product attributes\n
        right_attr (str): string prefix for recognize the left product attributes\n
        ignore_columns (list): list of attributes to ignore in the similarity computation\n
        na_value: value to fill NaN with
        """
        self.data = data
        self.left_attr = left_attr
        self.right_attr = right_attr
        self.ignore_columns = ignore_columns
        self.na_value = na_value
        self.left_prod_attrs = [
            col for col in data 
            if col.startswith(left_attr) and col not in ignore_columns
        ]
        assert(len(data[self.left_prod_attrs]) > 0)
        self.right_prod_attrs = [
            col for col in data 
            if col.startswith(right_attr) and col not in ignore_columns
        ]
        assert(len(data[self.right_prod_attrs]) > 0)
        assert(len(data[self.left_prod_attrs].keys()) == len(data[self.right_prod_attrs].keys()))
        self.data[self.left_prod_attrs + self.right_prod_attrs] = self.data[self.left_prod_attrs + self.right_prod_attrs]\
                                                                        .fillna(value=na_value)

    def get_scores(
        self,
        metric=sm.Jaccard(),
        tokenizer=sm.QgramTokenizer(),
        similarity=True,
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
            For a complete review please look at `py_stringmatching` package\n
        tokenizer (str): which `py_stringmatching` tokenizer use to tokenize text\n
        similarity (bool): compute similarity score if True, disteance otherwise
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
                    scores.append(0)
                else:
                    if tokenizer is None:
                        scores.append(metric.get_sim_score(row_left[i], row_right[i]))
                    else:
                        scores.append(metric.get_sim_score(tokenizer.tokenize(row_left[i]), tokenizer.tokenize(row_right[i])))
            return sum([scores[i] * weights[i] for i in range(len(scores))])

        left_attrs_num = len(self.data[self.left_prod_attrs].keys())
        right_attrs_num = len(self.data[self.right_prod_attrs].keys())
        if weights is not None:
            assert(left_attrs_num == right_attrs_num == len(weights))
            # assert(sum(weights) == 1)
        else:
            # Equal weight for every attributes
            weights = [1] * left_attrs_num
            assert(left_attrs_num == right_attrs_num == len(weights))
       
        tqdm.pandas()
        match_score = self.data[self.left_prod_attrs + self.right_prod_attrs]\
                        .progress_apply(
                            lambda row: scores(row[self.left_prod_attrs], row[self.right_prod_attrs]), 
                            axis=1
                        )
        new_data = self.data.copy()
        new_data.insert(0, 'match_score', match_score)
        return new_data


class SimilarityDataset(Dataset):

    def __init__(
        self, 
        data: pandas.DataFrame,
        metric=sm.Jaccard(),
        tokenizer=sm.QgramTokenizer(),
        label_attr='label',
        left_attr='ltable_', 
        right_attr='rtable_', 
        ignore_columns=[],
        na_value='',
        ):
        """
        With this class one can compute distance or similarity measure, based on functions
        provided by the `py_stringmatching` package.
        It requires a dataset with a schema similar to the one for deepmatcher:
        |Left attr 1|Left attr 2|...|Right attr 1|Right attr 2|...|Other attrs|

        Parameters
        ----------
        data (pandas.DataFrame): dataset on which compute similarity scores for each tuple\n
        metric (str): which metric to use: it has to be a py_stringmatching class. 
            For a complete review please look at `py_stringmatching` package\n
        tokenizer (str): which `py_stringmatching` tokenizer use to tokenize text\n
        label_attr (str): string for recognize the label attribute\n
        left_attr (str): string prefix for recognize the left product attributes\n
        right_attr (str): string prefix for recognize the left product attributes\n
        ignore_columns (list): list of attributes to ignore in the similarity computation\n
        na_value: value to fill NaN with
        """
        self.data = data
        self.metric = metric
        self.tokenizer = tokenizer
        self.label_attr = label_attr
        self.left_attr = left_attr
        self.right_attr = right_attr
        self.ignore_columns = ignore_columns
        self.na_value = na_value
        self.left_prod_attrs = [
            col for col in data 
            if col.startswith(left_attr) and col not in ignore_columns
        ]
        assert(len(data[self.left_prod_attrs]) > 0)
        self.right_prod_attrs = [
            col for col in data 
            if col.startswith(right_attr) and col not in ignore_columns
        ]
        assert(len(data[self.right_prod_attrs]) > 0)
        assert(len(data[self.left_prod_attrs].keys()) == len(data[self.right_prod_attrs].keys()))
        self.data[self.left_prod_attrs + self.right_prod_attrs] = self.data[self.left_prod_attrs + self.right_prod_attrs]\
                                                                        .fillna(value=na_value)
    
    def __scores(self, row_left, row_right):
            scores = []
            for i in range(len(row_left)):
                if row_left[i] == self.na_value or row_right[i] == self.na_value:
                    scores.append(0)
                else:
                    if self.tokenizer is None:
                        scores.append(self.metric.get_sim_score(row_left[i], row_right[i]))
                    else:
                        scores.append(
                            self.metric.get_sim_score(
                                self.tokenizer.tokenize(row_left[i]), self.tokenizer.tokenize(row_right[i])
                            )
                        )
            return scores

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row_left = self.data.iloc[index, [self.data.columns.get_loc(col) for col in self.left_prod_attrs]]
        row_right = self.data.iloc[index, [self.data.columns.get_loc(col) for col in self.right_prod_attrs]]
        label = self.data.iloc[index, self.data.columns.get_loc(self.label_attr)]
        scores = torch.FloatTensor(self.__scores(row_left, row_right))
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

        train_dataset: train dataset\n
        val_dataset: dataset used to validate the model and save the best one\n
        model: model to train\n
        resume=False: wheater to resume training on the last best model saving\n
        criterion=None: which type of loss will be computed. By default it's used NLLLoss with weight 
        specified by the pos_neg_ratio (weight=[1/pos_neg_ratio, 1] if pos_neg_ratio >= 1; weight=[1, 1/pos_neg_ratio],
        where the first element is the weight of the '0' class)\n
        optimizer=None: how to compute backpropagation. By default will be used SDG with a learning rate equals to 0.01
        and a 0.9 momentum\n
        scheduler=None: a scheduler to drop learning rate. By default the learning rate will be dropped
        based on the F1 measure on validation, if for 5 epochs won't upgrade\n
        train_epochs=10: train epochs\n
        pos_neg_ratio=1: ratio of positive examples weight wrt negative examples weight\n
        best_model_name='best_model': name of the best model
        best_save_on='F1': statistics on which the best model will be saved. It must be one of the following:
        'accuracy', 'precision', 'recall', 'f1'\n
        best_save_path=None: path where the best model will be saved.\n
        device=None: device to run train and validation.\n
        batch_size=32: batch size\n
        **kwargs\n
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

        dataset: dataset that will be evaluated\n
        model: model evaluate\n
        load_best_model=True: wheater to load the best evaluated model.\n
        best_model_name='best_model': name of the best model.\n
        best_save_path=None: path where the best model will be saved.\n
        device=None: device to run train and validation.\n
        batch_size=32: batch size\n
        **kwargs\n
        In the **kwargs you can specify the log frequency of the statistics

        Returns
        -------

        Return predictions scores on dataset
        """
        return Runner.predict(model=self, *args, **kwargs)
