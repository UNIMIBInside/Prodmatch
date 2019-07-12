# ceneje_prodmatch

## Requirements

* At least **python 3.6**. I didn't run test with prior versions
* **Pipenv** for the management of the virtual env. One can install it with `pip install pipenv` or `pip3 install pipenv`
* All test were performed using **Ubuntu 18.04** or **19.04**

## Setup

1. `git clone git@bitbucket.org:ceneje/deepmatcher.git`
2. Move to the project root and run `pipenv install`: this will create the virtual env and install all the needed packages
3. `pipenv shell` to activate the env
4. In order to use the deepmatcher model I trained, you have to download it from https://drive.google.com/file/d/1bAg_90ITxOn9GvauhH2LJ3Y31-NqLE60/view?usp=sharing and place it in `ceneje_prodmatch/results/models` folder
5. `python get_best_matching_prods.py` will generate a json file in `ceneje_prodmatch/results/best_predictions.json` which contains, for every offer in a specific category the possible matching Ceneje products in that category (as we assume that the categorization step can be performed in some way: black magic). TODO: handle unsplitted data


## Tree structure:
```bash 
├── ceneje_prodmatch
│   ├── cache (Cached files by deepmatcher)
│   ├── ceneje_data
│   │   ├── Products_LedTv_20190426.csv
│   │   ├── Products_Monitor_20190515.csv
│   │   ├── ...
│   │   ├── SellerProductsData_LedTv_20190426.csv
│   │   ├── SellerProductsData_Monitor_20190515.csv
│   │   ├── ...
│   │   ├── SellerProductsMapping_LedTv_20190426.csv
│   │   ├── SellerProductsMapping_Monitor_20190515.csv
│   │   ├── ...
│   │   ├── slovenian-stopwords.txt
│   │   └── unsplitted
│   │       ├── Products_NewCategories_20190605.csv
│   │       ├── SellerProductsData_NewCategories_20190605.csv
│   │       └── SellerProductsMapping_NewCategories_20190605.csv
│   ├── deepmatcher_data (Datasets needed to train deepmatcher)
│   ├── __init__.py
│   ├── results (Deepmacther predictions)
│   │   ├── models (Deepmatcher best models)
│   │   │   └── rnn_pos_neg_fasttext_new_cat_rand_model.pth
│   │   ├── offers.csv
│   │   ├── offers_preds.csv
│   └── src
│       ├── config
│       │   ├── config.example.json
│       │   ├── config.json
│       │   └── __init__.py
│       ├── helper
│       │   ├── deepmatcherdata.py
│       │   ├── __init__.py
│       │   ├── preprocess.py
│       ├── __init__.py
│       ├── matching
│           ├── __init__.py
│           ├── runner.py
│           └── similarity.py
├── create_deepmatcherdata.py
├── get_best_matching_prods.py
├── Pipfile
├── Pipfile.lock
├── README.md
├── train_deepmatcher.py
├── yes
└── yes.pub
```