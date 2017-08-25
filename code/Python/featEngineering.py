# -*- coding: utf-8 -*-
"""
Created on Mon Aug 07 21:31:50 2017

@author: evanm_000
"""

def sqFtFeat(df):
    
   import pandas as pd 
   import numpy as np
   
   df['area_live_finished_log'] = np.log(1 + df['finishedsquarefeet12'])

   df['bathpersqft'] = df['bathroomcnt'] / df['calculatedfinishedsquarefeet']
   df['roompersqft'] = df['roomcnt'] / df['calculatedfinishedsquarefeet']
   df['bedroompersqft'] = df['bedroomcnt'] / df['calculatedfinishedsquarefeet']
   df['Ratio_1'] = df['taxvaluedollarcnt'] / df['taxamount']
   
   df['taxvaluebin'] = pd.cut(df['taxvaluedollarcnt'], 10)
   df['taxvaluedollarcnt'] = np.log(1 + df['taxvaluedollarcnt'])
   
   return df

def ExtractTimeFeats(df):
    
    import pandas as pd
    import datetime as dt
        
    df['transactiondate'] = pd.to_datetime(df['transactiondate'])
    df["trans_month"] = df["transactiondate"].dt.month
    df['trans_day'] = df["transactiondate"].dt.day
    df['trans_year'] = df["transactiondate"].dt.year
    df['trans_qtr'] = df["transactiondate"].dt.quarter
    
    return df

def GetFeatures(df):
    
    import pandas as pd
    
    df["transactiondate"] = pd.to_datetime(df["transactiondate"])
    df["transactiondate_year"] = df["transactiondate"].dt.year
    df["transactiondate_month"] = df["transactiondate"].dt.month
    df['transactiondate'] = df['transactiondate'].dt.quarter
    df = df.fillna(-1.0)
    
    return df
    
    