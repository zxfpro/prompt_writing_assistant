from .utils.leadlag import leadlag
import numpy as np
from esig import tosig
import pandas as pd

def read_data(path):
    data = pd.read_csv(path, index_col=0)["Close"]  #
    data.index = data.index.astype("datetime64[ns]")
    return data


def resample(data,rule=21):
    if isinstance(rule,int):
        ist = [i for i in range(0,len(data),rule)]
        resample_data_list = [data[ist[i]:ist[i+1]] for i in range(len(ist)-1)]
        
    else:
        resample_data_list = [data for _,data in data.resample(rule)]
    return resample_data_list



def deal_data(data,minmax,level,data_deal_type='M'):
    windows = []
    for window in resample(data,data_deal_type):
        values = window.values  # / window.values[0]
        if values.shape[0]==0:
            continue
        path = leadlag(values)
        # path = np.insert(path,0,[0.0,0.0],axis=0)
        windows.append(path)

    orig_logsig = np.array([tosig.stream2logsig(path, level) for path in windows])

    logsig = minmax.fit_transform(orig_logsig)

    logsigs = logsig[1:]
    conditions = logsig[:-1]

    return logsigs,conditions,orig_logsig,windows