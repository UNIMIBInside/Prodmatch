import logging
import deepmatcher as dm
from os import path
from pandas import pandas
from ceneje_prodmatch import DATA_DIR, DEEPMATCH_DIR, RESULTS_DIR, CACHE_DIR
from ceneje_prodmatch.scripts.helpers import preprocess
from ceneje_prodmatch.scripts.helpers.deepmatcherdata import deepmatcherdata

logging.getLogger('deepmatcher.core')
logging.getLogger('deepmatcher.modules')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    columns = ['idProduct']
    ignore_columns = ['ltable_' + col for col in columns]
    ignore_columns += ['rtable_' + col for col in columns]
    ignore_columns += ['idProduct']
    train, validation, test = dm.data.process(
        path=DEEPMATCH_DIR,
        cache=path.join(CACHE_DIR, 'rnn_lstm_fasttext_cache.pth'),
        train='train.csv',
        validation='validation.csv',
        test='test.csv',
        device=device,
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
        device=device, 
        attr_summarizer=dm.attr_summarizers.RNN(
            word_contextualizer='lstm'
        ),
        attr_comparator='abs-diff'
    )
    model.initialize(train, device)
    model.run_train(
        train,
        validation,
        device=device,
        epochs=10,
        batch_size=16,
        best_save_path=path.join(RESULTS_DIR, 'models', 'rnn_lstm_fasttext_model.pth'),
    )
    model.run_eval(test, device=device)
    model.load_state(path.join(RESULTS_DIR, 'models', 'rnn_lstm_fasttext_model.pth'), device=device)
    candidate = dm.data.process_unlabeled(
        path=path.join(DEEPMATCH_DIR, 'unlabeled.csv'),
        trained_model=model     ,
        ignore_columns=ignore_columns + ['label'])
    predictions = model.run_prediction(candidate, device=device, output_attributes=list(candidate.get_raw_table().columns))
    predictions.to_csv(RESULTS_DIR, 'predictions.csv')
