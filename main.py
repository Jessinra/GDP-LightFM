# %%[markdown]
# # <b><i> Testing LightFM </i></b>

#%%[markdown]
# # > Import
# In[ ]:
import pickle
from datetime import datetime

from lightfm import LightFM
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import precision_at_k
from scipy.sparse import identity
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from logger import Logger
#%%[markdown]
# # > Preparation
#%%[markdown]
# ## >> Load data
# In[ ]:
ratings_pivot_csr_filename = "data/intersect-20m/ratings.csr"
ratings_pivot = pickle.load(open(ratings_pivot_csr_filename, 'rb'))
#%%[markdown]
# ## >> Split data
train, test = random_train_test_split(ratings_pivot, test_percentage=0.2)
# %%[markdown]
# ## >> User & Item features
# Identity matrix to represent users and items feature
# In[ ]:
user_identity = identity(train.shape[0])
item_identity = identity(train.shape[1])
#%%[markdown]
# ## >> Set logger
timestamp = str(datetime.timestamp(datetime.now()))

logger = Logger()
session_log_path = "log/{}/".format(timestamp)
logger.create_session_folder(session_log_path)
logger.set_default_filename(session_log_path + "log.txt")
# %%[markdown]
# # > Args
# In[ ]:
alpha = 0.003
epochs = 50
num_components = 32
step = 5
# %%[markdown]
# # > Model
# In[ ]:
model_k10 = LightFM(no_components=num_components,
                    loss='warp',
                    k=10,
                    learning_schedule='adagrad',
                    user_alpha=alpha,
                    item_alpha=alpha)
# %%[markdown]
# # > Train 
logger.log(str(model_k10.get_params()))
for epoch in tqdm(range(epochs)):

    model_k10.fit_partial(train, epochs=step, user_features=user_identity, item_features=item_identity, num_threads=6, verbose=True)

    mean_precision = precision_at_k(model_k10, train, k=10).mean()
    logger.log("Precision k10 : {}".format(mean_precision))
    logger.save_model(model_k10, session_log_path + "models/epoch_{}".format(epoch))
