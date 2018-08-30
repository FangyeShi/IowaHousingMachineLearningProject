import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import preprocessing

# Some utility functions

def log_columns(df_in, col_list):
    # Log transform columns in col_list.
    # Make a copy so the original dataframe will not be altered.
    df_out = df_in.copy()
    for col in col_list:
        df_out[col] = np.log1p(df_out[col])
    return df_out

def normalizeDf(train_df, test_df):
    # Normalize train_df and test_df using MinMaxScaler

    colNames = list(train_df.columns.values)

    # Fit_transform on train_df
    train_x = train_df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    train_x_scaled = min_max_scaler.fit_transform(train_x)
    train_df = pd.DataFrame(train_x_scaled, columns = colNames)

    # Transform on test_df
    test_x = test_df.values
    test_x_scaled = min_max_scaler.transform(test_x)
    test_df = pd.DataFrame(test_x_scaled, columns = colNames)

    return train_df, test_df

def pred_vs_actual(df_encoded, y_pred_train, y_train):
    # Plot residual vs actual

    # ----Calculate Predicted vs. Actual SalePrice----
    pred_depiction = pd.DataFrame({'index_train': df_encoded[df_encoded['Id']<=1460]['Id'], 'pred_train': np.expm1(y_pred_train), 'actual_train': np.expm1(y_train), 'actual_train_log': y_train, 'pred_train_log': y_pred_train})
    pred_depiction['pred_minus_act'] = pred_depiction['pred_train'] - pred_depiction['actual_train']
    pred_depiction['pred_minus_act_absval'] = abs(pred_depiction['pred_train'] - pred_depiction['actual_train'])
    pred_depiction['predlog_minus_actlog'] = pred_depiction['pred_train_log'] - pred_depiction['actual_train_log']
    pred_depiction['predlog_minus_actlog_absval'] = abs(pred_depiction['predlog_minus_actlog'])

    # ----Create Graphs of Predicted vs. Actual SalePrice (and a version logged)----
    sns.regplot(x="actual_train", y="pred_train", data=pred_depiction)
    plt.figure()
    sns.regplot(x="actual_train", y="pred_minus_act", data=pred_depiction)
    plt.figure()
    sns.regplot(x="actual_train_log", y="pred_train_log", data=pred_depiction)
    plt.figure()
    sns.regplot(x="actual_train_log", y="predlog_minus_actlog", data=pred_depiction)

    # Return the dataframe. Useful to find outlier.
    return pred_depiction



def get_avg(folderpath,merge_name):
    '''
    Function 'get_avg' takes a folder path and a file name and save the average predictions from all files in the folder to the new file.

    Parameters
    ----------
    folderpath: string
        Relative path of the folder which contains submissions files.

    merge_name: string
        Name of the .csv file we want to save.
    '''

    combined = pd.DataFrame({'Id':range(1461,2920)})

    for filename in os.listdir(folderpath):
        if filename.endswith(".csv"):
            temp = pd.read_csv(folderpath+'/'+filename)
            combined[filename]=temp['SalePrice']


    combined.drop('Id',axis= 1, inplace=True)
    combined['SalePrice'] = list(combined.mean(axis=1))
    combined['Id'] = range(1461,2920)

    final = pd.DataFrame({'Id':range(1461,2920)})
    final['SalePrice'] = combined['SalePrice']
    final.to_csv('./average_submissions/'+merge_name+'.csv',index=False)
