from sklearn.preprocessing import MinMaxScaler
from .datadealer import read_data,deal_data
from .tf_model import CVAE

def train(data,level=4,types='M',n_epochs=10000):
    minmax = MinMaxScaler(feature_range=(0.00001, 0.99999))
    logsigs, conditions,orig_logsig, windows = deal_data(data, minmax, level,types)
    generator = CVAE()
    generator.train(logsigs,conditions, n_epochs=n_epochs)

    return generator,minmax,logsigs,conditions

















