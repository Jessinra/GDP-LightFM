import pickle

from logger import Logger
from datetime import datetime

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from lightfm import LightFM
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import precision_at_k

ratings_pivot_csr_filename = "data/intersect-20m/ratings.csr"
ratings_pivot = pickle.load(open(ratings_pivot_csr_filename, 'rb'))
train, test = random_train_test_split(ratings_pivot, test_percentage=0.2)
print("     =====> Dataset loaded")

# ========== Parameter ==========
alpha = 1e-03
epochs = 30
num_components = 32

timestamp = str(datetime.timestamp(datetime.now()))

logger = Logger()
session_log_path = "../log/{}/".format(timestamp)
logger.create_session_folder(session_log_path)
logger.set_default_filename(session_log_path + "log.txt")


# ========== Models ==========
model_k5 = LightFM(no_components=num_components,
                    loss='warp',
                    k=5,
                    learning_schedule='adagrad',
                    user_alpha=alpha,
                    item_alpha=alpha)

model_k10 = LightFM(no_components=num_components,
                    loss='warp',
                    k=10,
                    learning_schedule='adagrad',
                    user_alpha=alpha,
                    item_alpha=alpha)

print("     =====> Model created")

# ========== Train ==========

print("     =====> Running K5 models")
for epoch in tqdm(range(epochs)): 
    
    model_k5.fit_partial(train, epochs=1, num_threads=4)
    
    mean_precision = precision_at_k(model_k5, train, k=5).mean()
    logger.log("Precision k5 : {}".format(mean_precision))
    logger.save_model(model_k5, session_log_path + "models/epoch_{}".format(epoch))

print("     =====> Running K10 models")
for epoch in tqdm(range(epochs)): 
    
    model_k10.fit_partial(train, epochs=1, num_threads=4)

    mean_precision = precision_at_k(model_k10, train, k=10).mean()
    logger.log("Precision k10 : {}".format(mean_precision))
    logger.save_model(model_k10, session_log_path + "models/epoch_{}".format(epoch))