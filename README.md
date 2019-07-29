# GDP-LightFM
***Last edit : 29 July 2019***

Recommender system using [LightFM](https://lyst.github.io/lightfm/docs/home.html) (without knowledge graph), trained using custom [MovieLens-20M](https://grouplens.org/datasets/movielens/20m/) dataset.
<br>

# Domain of problems
*Given a user and a list of items, find the top k items that user might like.*

# Contents
- `/data` : contains dataset to use in training
    - ***`/intersect-20m`*** : custom ml-20m dataset where only movies shows up in Ripple-Net's knowledge graph used.    
    - `/ml-1m` : original movielens 1m dataset
- **`/log`** : contains training result stored in single folder named after training timestamp.
- **`/test`** : contains jupyter notebook used in testing the trained models
<!-- ---------------------------------------- -->
- `logger.py` : log training result and save models
- `main.py` : script to run training 
<!-- ---------------------------------------- -->
- `LightFM.ipynb` : jupyter notebook to make LightFM model (no longer used)

### Note
    *italic* means this folder is ommited from git, but necessary if you need to run experiments
    **bold** means this folder has it's own README, check it for detailed information :)

# Preparing 
## Installing dependencies 

    pip3 install -r requirements.txt

## Where to download the dataset
You can download the intersect-20m dataset [here](https://github.com/Jessinra/GDP-KG-Dataset). 

*Note : Dataset is put on separate repository because it's shared among models.*

## Missing component that are required 
- `data/intersect-20m/ratings_re2.csv` : After download the dataset, unzip the `ratings_re2.zip` and put inside the same folder as other things downloaded.
- `data/intersect-20m/ratings.csr` : csr matrix (row = user, col = items, value = ratings) created with intersect-20m ratings, this file can be created using this [jupyter notebook](https://github.com/Jessinra/GDP-KG-Dataset/blob/master/Preprocess.ipynb) (Preprocess.ipynb inside the dataset).

## How to prepare data
Simply provide `data/intersect-20m/ratings.csr` and run `main.py`, the script will preprocess it before training begin.

# How to run
1. Prepare the dataset (check section below this)
2. Run the training script
    ~~~
    python3 main.py
    ~~~

# Training
## How to change hyper parameter
There are several ways to do this :
1. Open `main.py` and change the args parser default value
2. run `main.py` with arguments required.

# Testing / Evaluation
## How to check training result
1. Find the training result folder inside `/log` (find the latest), copy the folder name.
2. Create copy of latest jupyter notebook inside `/test` folder.
3. Rename folder to match a folder in `/log` (for traceability purpose).
4. Replace `TESTING_CODE` at the top of the notebook.
5. Run the notebook

# Final result
| Evaluated on  |  Prec@10   |
|---------------|------------|
|    500 user   |   0.09100  |
|   1000 user   |   0.09330  |
|   5000 user   |   0.09902  |
|  13850 user   |   0.09864  |
|  25000 user   |   0.09764  |

| Evaluated on  | Distinct@10   | Unique items |
|---------------|---------------|--------------|
|     10 user   |    0.23000    |    23        |
|     30 user   |    0.04333    |    13        |
|    100 user   |    0.01300    |    13        |
|   1000 user   |    0.00190    |    19        |
|   3000 user   |    0.00043    |    13        |

# Other findings
- Models tends to suggest generic items that are rated high by a large number of users.
- LightFM model is much slower to train & test (compared to Autorec) since it doesn't support GPU
- It's user centric, when predicting, the model require input (users, items) and will output score for each user-items pairing.
- Require re-train if model want to predict for new user
- Model can be improved using user's features and item's features (also handle cold-start problem)

# Pros
- Model can handle cold-start problem; improved using user's features and item's features (eg: user genre preferences / other features).
- The model uses suitable loss and metric (Prec@k)

# Cons
- Models tends to suggest generic items that are rated high by a large number of users.
- LightFM model is much slower to train & test (compared to Autorec) since it doesn't support GPU
- It's user centric, when predicting, the model require input (users, items) and will output score for each user-items pairing
- Require re-train if model want to predict for new user

# Experiment notes
- The model doesn't learn when dimension set to 64
  
# Author
- Jessin Donnyson - jessinra@gmail.com

# Contributors
- Benedict Tobias Henokh Wahyudi - tobi8800@gmail.com
- Michael Julio - michael.julio@gdplabs.id
- Fallon Candra - fallon.candra@gdplabs.id