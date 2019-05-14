import logging
import deepmatcher as dm
from os import path
from pandas import pandas
from ceneje_prodmatch import DATA_DIR, RESULTS_DIR, CACHE_DIR
from ceneje_prodmatch.scripts.helpers import preprocess
from ceneje_prodmatch.scripts.helpers.deepmatcherdata import deepmatcherdata

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    columns = ['idProduct']
    ignore_columns = ['ltable_' + col for col in columns]
    ignore_columns += ['rtable_' + col for col in columns]
    ignore_columns += ['idProduct']
    train, validation, test = dm.data.process(
        path=path.join('ceneje_prodmatch/ceneje_data'),
        cache=path.join(CACHE_DIR, 'rnn_lstm_fasttext_cache.pth'),
        train='train.csv',
        validation='validation.csv',
        test='test.csv',
        ignore_columns=ignore_columns,
        lowercase=True,
        embeddings='fasttext.sl.bin',
        id_attr='_id',
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
    model.load_state(path.join(RESULTS_DIR, 'rnn_lstm_fasttext_model.pth'))
    candidate = dm.data.process_unlabeled(
        path=path.join('ceneje_prodmatch/ceneje_data/unlabeled.csv'),
        trained_model=model     ,
        ignore_columns=ignore_columns + ['label'])
    predictions = model.run_prediction(candidate, output_attributes=list(candidate.get_raw_table().columns))
    predictions.to_csv('ceneje_prodmatch/results/predictions.csv')
