import numpy
import time
import deepmatcher as dm
from os import path
from pandas import pandas
from ceneje_prodmatch import DATA_DIR, RESULTS_DIR, CACHE_DIR
from ceneje_prodmatch.scripts.helpers import preprocess
from ceneje_prodmatch.scripts.helpers.deepmatcherdata import deepmatcherdata
import io
from torchtext.utils import unicode_csv_reader

if __name__ == "__main__":
    # with io.open(path.join(DATA_DIR, 'train.csv'), encoding="utf8") as f:
    #     header = next(unicode_csv_reader(f, delimiter='\t'))
    # print(header)
    # train = pandas.read_csv(path.join(DATA_DIR, 'train.csv'))
    # validation = pandas.read_csv(path.join(DATA_DIR, 'validation.csv'))
    # test = pandas.read_csv(path.join(DATA_DIR, 'test.csv'))
    columns = ['idProduct']
    ignore_columns = ['ltable_' + col for col in columns]
    ignore_columns += ['rtable_' + col for col in columns]
    ignore_columns += ['idProduct']
    train, validation, test = dm.data.process(
        path=path.join('ceneje_prodmatch/ceneje_data'),
        cache=path.join(CACHE_DIR, 'cache', 'rnn_lstm_fasttext_model.pth'),
        train='train.csv',
        validation='validation.csv',
        test='test.csv',
        sep=',',
        ignore_columns=ignore_columns,
        lowercase=True,
        embeddings='fasttext.wiki.vec',
        id_attr='id',
        label_attr='label',
        left_prefix='ltable_',
        right_prefix='rtable_',
        pca=False
    )
    model = dm.MatchingModel(
        attr_summarizer=dm.attr_summarizers.RNN(
            word_contextualizer='lstm'
        ),
        attr_comparator='abs-diff'
    )
    model.initialize(train)
    model.run_train(
        train,
        validation,
        epochs=10,
        batch_size=16,
        best_save_path=path.join(RESULTS_DIR, 'rnn_lstm_fasttext_model.pth'),
    )
    model.run_eval(test)