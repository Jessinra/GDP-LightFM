# Intersect-20m
Custom ml-20m dataset derived from [MovieLens-20M](https://grouplens.org/datasets/movielens/20m/) dataset, where only movies shows up in Ripple-Net's knowledge graph used.

## Where to download the dataset
You can download the intersect-20m dataset [here](https://github.com/Jessinra/GDP-KG-Dataset). 

*Note : Dataset is put on separate repository because it's shared among models.*

## Missing component that are required 
- `data/intersect-20m/ratings_re2.csv` : After download the dataset, unzip the `ratings_re2.zip` and put inside the same folder as other things downloaded.
- `data/intersect-20m/ratings.csr` : csr matrix (row = user, col = items, value = ratings) created with intersect-20m ratings, this file can be created using this [jupyter notebook](https://github.com/Jessinra/GDP-KG-Dataset/blob/master/Preprocess.ipynb) (Preprocess.ipynb inside the dataset).