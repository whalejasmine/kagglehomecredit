
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
import gc
import time
import warnings
import sys
warnings.filterwarnings("ignore")


# In[35]:


from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.model_selection import KFold, StratifiedKFold

'''@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))
'''

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from scipy.stats import ranksums
import xgboost as xgb
sys.path.insert(0, '/home/jingqi/gpbr_module/') #import my module
from bayes_optz import BayesianOptimization
from sklearn.preprocessing import LabelEncoder 


# ## Aggregating datasets

# ### Service functions

# In[8]:


def le(df):
    categorical_mask = (df.dtypes == object)
    categorical_columns = df.columns[categorical_mask].tolist()
    if categorical_columns == []:
        return df
    else:
        le = LabelEncoder()
        df[categorical_columns] = df[categorical_columns].apply(lambda x: le.fit_transform(x))
        return df
def reduce_mem_usage(data, verbose = True):
    start_mem = data.memory_usage().sum() / 1024**2
    if verbose:
        print('Memory usage of dataframe: {:.2f} MB'.format(start_mem))
    
    for col in data.columns:
        col_type = data[col].dtype
        
        if col_type != object:
            c_min = data[col].min()
            c_max = data[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    data[col] = data[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    data[col] = data[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    data[col] = data[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    data[col] = data[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    data[col] = data[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    data[col] = data[col].astype(np.float32)
                else:
                    data[col] = data[col].astype(np.float64)

    end_mem = data.memory_usage().sum() / 1024**2
    if verbose:
        print('Memory usage after optimization: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return data


# In[9]:


def one_hot_encoder(data, nan_as_category = True):
    original_columns = list(data.columns)
    categorical_columns = [col for col in data.columns                            if not pd.api.types.is_numeric_dtype(data[col].dtype)]
    for c in categorical_columns:
        if nan_as_category:
            data[c].fillna('NaN', inplace = True)
        values = list(data[c].unique())
        for v in values:
            data[str(c) + '_' + str(v)] = (data[c] == v).astype(np.uint8)
    data.drop(categorical_columns, axis = 1, inplace = True)
    return data, [c for c in data.columns if c not in original_columns]


# ### Aggregating functions

# In[18]:


file_path = '../../data/hc/'




def cv_scores(df, num_folds, params, stratified = False, verbose = -1, 
              save_train_prediction = False, train_prediction_file_name = 'train_prediction.csv',
              save_test_prediction = True, test_prediction_file_name = 'test_prediction.csv'):
    warnings.simplefilter('ignore')
    

    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    
    del df
    gc.collect()
    print("Starting XGboost. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))



    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits = num_folds, shuffle = True, random_state = 1001)
    else:
        folds = KFold(n_splits = num_folds, shuffle = True, random_state = 1001)
        
    # Create arrays and dataframes to store results
    train_pred = np.zeros(train_df.shape[0])
    train_pred_proba = np.zeros(train_df.shape[0])

    test_pred = np.zeros(train_df.shape[0])
    test_pred_proba = np.zeros(train_df.shape[0])
    
    prediction = np.zeros(test_df.shape[0])
    
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    
    df_feature_importance = pd.DataFrame(index = feats)
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        print('Fold', n_fold, 'started at', time.ctime())
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]
        clf = XGBClassifier(**params)

        clf.fit(train_x, train_y, 
                eval_set = [(train_x, train_y), (valid_x, valid_y)], eval_metric = 'auc', 
                verbose = verbose, early_stopping_rounds = 200)

        #train_pred[train_idx] = clf.predict(train_x, ntree_limit=clf.best_iteration)
        #train_pred_proba[train_idx] = clf.predict_proba(train_x, ntree_limit=clf.best_iteration)[:, 1]
        #test_pred[valid_idx] = clf.predict(valid_x, ntree_limit=clf.best_iteration)

        test_pred_proba[valid_idx] = clf.predict_proba(valid_x, ntree_limit=clf.best_iteration)[:, 1]
        
        prediction += clf.predict_proba(test_df[feats], ntree_limit=clf.best_iteration)[:, 1] / folds.n_splits

        df_feature_importance[n_fold] = pd.Series(clf.feature_importances_, index = feats)
        
        print('Fold %2d AUC : %.6f' % (n_fold, roc_auc_score(valid_y, test_pred_proba[valid_idx])))
        del train_x, train_y, valid_x, valid_y
        gc.collect()

   # roc_auc_train = roc_auc_score(train_df['TARGET'], train_pred_proba)
    #precision_train = precision_score(train_df['TARGET'], train_pred, average = None)
    #recall_train = recall_score(train_df['TARGET'], train_pred, average = None)
    
    roc_auc_test = roc_auc_score(train_df['TARGET'], test_pred_proba)
    #precision_test = precision_score(train_df['TARGET'], test_pred, average = None)
    #recall_test = recall_score(train_df['TARGET'], test_pred, average = None)

    print('Full AUC score %.6f' % roc_auc_test)
    
    df_feature_importance.fillna(0, inplace = True)
    df_feature_importance['mean'] = df_feature_importance.mean(axis = 1)
    
    # Write prediction files
    if save_train_prediction:
        df_prediction = train_df[['SK_ID_CURR', 'TARGET']]
        df_prediction['Prediction'] = test_pred_proba
        df_prediction.to_csv(train_prediction_file_name, index = False)
        del df_prediction
        gc.collect()

    if save_test_prediction:
        df_prediction = test_df[['SK_ID_CURR']]
        df_prediction['TARGET'] = prediction
        df_prediction.to_csv(test_prediction_file_name, index = False)
        del df_prediction
        gc.collect()
    
    return df_feature_importance, test_pred_proba.reshape(-1,1),prediction.reshape(-1,1)


# In[40]:


'''def display_folds_importances(feature_importance_df_, n_folds = 5):
    n_columns = 3
    n_rows = (n_folds + 1) // n_columns
    _, axes = plt.subplots(n_rows, n_columns, figsize=(8 * n_columns, 8 * n_rows))
    for i in range(n_folds):
        sns.barplot(x = i, y = 'index', data = feature_importance_df_.reset_index().sort_values(i, ascending = False).head(20), 
                    ax = axes[i // n_columns, i % n_columns])
    sns.barplot(x = 'mean', y = 'index', data = feature_importance_df_.reset_index().sort_values('mean', ascending = False).head(20), 
                    ax = axes[n_rows - 1, n_columns - 1])
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.show()'''


# ### Table for scores

# In[41]:


'''scores_index = [
    'roc_auc_train', 'roc_auc_test', 
    'precision_train_0', 'precision_test_0', 
    'precision_train_1', 'precision_test_1', 
    'recall_train_0', 'recall_test_0', 
    'recall_train_1', 'recall_test_1', 
    'LB'
]

scores = pd.DataFrame(index = scores_index)'''


# ### First scores with parameters from Tilii kernel

# In[42]:


# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
lgbm_params = {
            'nthread': 16,
            'n_estimators': 10000,
            'learning_rate': .02,
            'num_leaves': 34,
            'colsample_bytree': .9497036,
            'subsample': .8715623,
            'max_depth': 8,
            'reg_alpha': .041545473,
            'reg_lambda': .0735294,
            'min_split_gain': .0222415,
            'min_child_weight': 39.3259775,
            'silent': 1,
            'verbose': -1,
            'booster':'dart' 
}

'''
# In[ ]:


feature_importance, scor = cv_scores(df, 5, lgbm_params, test_prediction_file_name = 'prediction_0.csv')


# In[ ]:


step = 'Tilii`s Bayesian optimization'
scores[step] = scor
scores.loc['LB', step] = .797
scores.T


# In[ ]:


display_folds_importances(feature_importance)


# In[ ]:


feature_importance[feature_importance['mean'] == 0].shape


# In[ ]:


feature_importance.sort_values('mean', ascending = False).head(20)


# ### New Bayesian Optimization

# In[43]:
'''

def lgbm_evaluate(**params):
    warnings.simplefilter('ignore')
    
    params['num_leaves'] = int(params['num_leaves'])
    params['max_depth'] = int(params['max_depth'])
        
    clf = XGBClassifier(**params, n_estimators = 10000, nthread = 16)

    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    
    #train.fillna((-1), inplace=True) 
    #test.fillna((-1), inplace=True)

    #train=np.array(train) 
    #test=np.array(test) 

    
    print("Starting XGboost. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))

    folds = KFold(n_splits = 2, shuffle = True, random_state = 1001)
        
    test_pred_proba = np.zeros(train_df.shape[0])
    
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        clf.fit(train_x, train_y, 
                eval_set = [(train_x, train_y), (valid_x, valid_y)], eval_metric = 'auc', 
                verbose = False, early_stopping_rounds = 100)

        test_pred_proba[valid_idx] = clf.predict_proba(valid_x, ntree_limit=clf.best_iteration)[:, 1]
        
        del train_x, train_y, valid_x, valid_y
        gc.collect()

    return roc_auc_score(train_df['TARGET'], test_pred_proba)


# In[44]:



if __name__ == "__main__":
    df = pd.read_hdf('./data/df_833_.h5','xgboost_data')
    #df=le(df)
    #df=df.astype('float')
    params = {'colsample_bytree': (0.5, 1),
          'learning_rate': (.01, .02), 
          'num_leaves': (20, 50), 
          'subsample': (0.8, 1), 
          'max_depth': (7, 9), 
          'reg_alpha': (.02, .09), 
          'reg_lambda': (.02, .08), 
          'min_split_gain': (.01, .03),
          'min_child_weight': (20, 40)}
    '''bo = BayesianOptimization(lgbm_evaluate, params)
    bo.maximize(init_points = 5, n_iter = 20,acq='rnd')


    best_params = bo.res['max']['max_params']
    best_params['num_leaves'] = int(best_params['num_leaves'])
    best_params['max_depth'] = int(best_params['max_depth'])

    best_params

    bo.res['max']['max_val']'''


    feature_importance, xg_oof_train,xg_oof_test = cv_scores(df, 5, lgbm_params, test_prediction_file_name = 'xgboost_tune_dart.csv')
    
    # for stack
    np.save('xg_oof_train_dart',xg_oof_train)
    np.save('xg_oof_test_dart',xg_oof_test)


    main()
        
        
        
        
        
        
        
        
        

