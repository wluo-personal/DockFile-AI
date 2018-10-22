import lightgbm as lgb
import xgboost as xgb
import catboost as ctb
import tensorflow as tf

import pandas as pd
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
sh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - CLASS:%(name)s- METHOD:%(funcName)s -LINE:%(lineno)d - MSG:%(message)s')
sh.setFormatter(sh_formatter)
logger.addHandler(sh)

    



class lightgbm_gpu:
    def __init__(self):
        logger.info('lightgbm initialized!')
        self.clf = lgb.LGBMClassifier(n_jobs=12,
                            device='gpu')
        logger.info('lightgbm version: {}'.format(lgb.__version__))
        
        
    def fit(self,x,y):
        try:
            logger.info('lightgbm fitting!')
            self.clf.fit(x, y, eval_set=[(x,y)],  verbose=1000,early_stopping_rounds=100,eval_metric='logloss')
            return True
        except Exception as e:
            logger.error(e)
            return False
        

class xgboost_gpu:
    def __init__(self):
        logger.info('xgboost initialized!')
        self.clf = xgb.XGBClassifier(gpu_id=0,tree_method='gpu_hist',max_bin=32)  
        logger.info('xgboost version: {}'.format(xgb.__version__))
        
    def fit(self,x,y):
        try:
            logger.info('xgb fitting!')
            self.clf.fit(x, y, eval_set=[(x,y)],verbose=1000,eval_metric=['logloss','auc'])
            return True
        except Exception as e:
            logger.error(e)
            return False
        

        
class catboost_gpu:
    def __init__(self):
        logger.info('catboost initialized!')
        self.clf = ctb.CatBoostClassifier(iterations=100, 
                              depth=None, 
                              thread_count=10,
                              learning_rate=0.01, 
                              loss_function='Logloss',
                              verbose=5,
                              task_type='GPU')
        logger.info('catboost version: {}'.format(ctb.__version__))
        
    def fit(self,x,y):
        try:
            logger.info('catboost fitting!')
            self.clf.fit(x, y, eval_set=[(x,y)],verbose_eval=1000,metric_period=1000)
            return True
        except Exception as e:
            logger.error(e)
            return False
        
        
class nn_gpu:
    def __init__(self):
        logger.info('nn initialized!')
        self.clf = self._get_model()
        logger.info('tensorflow version: {}'.format(tf.__version__))
        
    def _get_model(self):
        x_in = Input(shape=(8,))
        x = Dense(12,activation='relu')(x_in)
        x = Dense(8,activation='relu')(x)
        x_out = Dense(1,activation='sigmoid')(x)
        model = Model(x_in, x_out)
        model.compile(loss='binary_crossentropy',optimizer=Adam(),metrics=['accuracy'])
        return model
        
    def fit(self,x,y):
        try:
            logger.info('nn fitting!')
            self.clf.fit(x, y, epochs=15, batch_size=10, verbose=2)
            return True
        except Exception as e:
            logger.error(e)
            return False
        
        
        
        
    
            