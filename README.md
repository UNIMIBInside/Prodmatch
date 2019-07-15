# ceneje_prodmatch

## Requirements

* At least **python 3.6**. I didn't run test with prior versions
* **Pipenv** for the management of the virtual env. One can install it with `pip install pipenv` or `pip3 install pipenv`
* All test were performed using **Ubuntu 18.04** or **19.04**

## Setup

1. `git clone git@bitbucket.org:ceneje/deepmatcher.git`
2. Change in `Pipfile` the python version
3. Move to the project root and run `pipenv install`: this will create the virtual env and install all the needed packages
4. `pipenv shell` to activate the env
5. In order to use the deepmatcher model I trained, you have to download it from https://drive.google.com/file/d/1bAg_90ITxOn9GvauhH2LJ3Y31-NqLE60/view?usp=sharing and place it in `ceneje_prodmatch/results/models` folder.  This model has been trained on the following 11 categories: 
    * Led tv
    * Washing machine
    * Refrigerator
    * Monitor
    * Summer/Winter tires
    * Digital camera
    * Accumulator drillers
    * Laptops
    * Men sneakers
    * Cartridges
6. `python get_best_matching_prods.py` will generate a json file in `ceneje_prodmatch/results/best_predictions.json` which contains, for every offer in a specific category the possible matching Ceneje products in that category (as we assume that the categorization step can be performed in some way: black magic). **P.S.** If you also want to include the description into the matching process follow the instruction in [get_best_matching_prods.py](get_best_matching_prods.py) file

## Config

One can tune all the configurable parameters in [ceneje_prodmatch/src/config/config.json](ceneje_prodmatch/src/config/config.json). There're six top keys:

* **default**: default configuration parameters for the deepmatcher data creation. See [create_deepmatcherdata.py](create_deepmatcherdata.py)
* **preprocess**: configuration for the preprocess step. See [ceneje_prodmatch/src/helper/preprocess.py](ceneje_prodmatch/src/helper/preprocess.py)
* **unsplitted**: whether to deal with unsplitted data or not. See [create_deepmatcherdata.py](create_deepmatcherdata.py)
* **deepmatcher**: configuration for deepmatcher
    * **creation**: creation of the datasets needed to train it. See also [ceneje_prodmatch/src/helper/deepmatcherdata.py](ceneje_prodmatch/src/helper/deepmatcherdata.py)
    * **train**: parameters for the training phase. See also [train_deepmatcher.py](train_deepmatcher.py)
* **split**: split the deepmatcher data into training, validation and test dataset
* **offers_matching**: get the best matching Ceneje products for some offers

## Useful links

* **deepmatcher**: since the deepmatcher framework is highly tunable, one may want have a look to its repository page at https://github.com/belerico/deepmatcher/tree/torch_1.0.1
* **py_stringmatching**: string similarity and distance https://github.com/anhaidgroup/py_stringmatching/tree/rel_0_4_1

## Tree structure

```bash 
├── ceneje_prodmatch
│   ├── cache (CACHED FILES BY DEEPMATCHER)
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
│   ├── deepmatcher_data (DATASETS NEEDED TO TRAIN DEEPMATCHER)
│   │   ├── deepmatcher.csv
│   │   ├── test.csv
│   │   ├── train.csv
│   │   ├── unlabeled.csv
│   │   └── validation.csv
│   ├── __init__.py
│   ├── results (DEEPMACTHER PREDICTIONS)
│   │   ├── models (DEEPMATCHER BEST MODELS)
│   │   │   └── rnn_pos_neg_fasttext_new_cat_rand_model.pth
│   │   ├── best_predictions.json
│   │   ├── offers.csv
│   │   └── offers_preds.csv
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
│       └── matching
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