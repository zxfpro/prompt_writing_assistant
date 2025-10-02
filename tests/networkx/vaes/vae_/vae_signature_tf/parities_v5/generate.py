from tqdm.auto import tqdm
from multiprocessing import Pool
from tqdm.auto import tqdm
import numpy as np
from .utils import logsig_inversion,tosig


def leadlag(X):
    lag = []
    lead = []
    for val_lag, val_lead in zip(X[:-1], X[1:]):
        lag.append(val_lag)
        lead.append(val_lag)

        lag.append(val_lag)
        lead.append(val_lead)

    lag.append(X[-1])
    lead.append(X[-1])

    return np.c_[lag, lead]


def deviced(window_size,n_points=21):
    if window_size == n_points:
        return 1,0
    else:
        two_cell = n_points -1
        return window_size//two_cell +1 ,two_cell - window_size%two_cell +1


class General:
    def __init__(self,minmax,n_pips=5,n_points=21):
        self.pip = 0.0001  # 最小变化单元 0.0001
        self.n_pips = n_pips * 10000 #最大变化幅度 *self.pip
        self.n_points = n_points  # 21

        self.n_iterations = 100  # 循环次数
        self.n_organisms = 400  # 生物数量
        self.level = 4
        
        self.minmax=minmax
    
    def generate(self,logsig,generator,n_samples=None, normalised=False):
        generated = generator.generate(logsig, n_samples=n_samples)
        if normalised:
            return generated
        if n_samples is None:
            return self.minmax.inverse_transform(generated.reshape(1, -1))[0]
        return self.minmax.inverse_transform(generated)
    
    def multi_works(self,generated):
        np.random.seed()
        paths = self.concatenate(generated)
        return paths

    def concatenate(self,generated):
        path, loss = logsig_inversion.train(generated, self.level, self.n_iterations, self.n_organisms, self.n_points,
                                            self.pip, self.n_pips)
        condition = tosig.stream2logsig(leadlag(path), self.level)
        condition = self.minmax.transform([condition])[0]
        return path,condition
    
    def generated_data(self,price,condition,generator,n_jobs=20, batch_size=20, window_size=50):
        periods, minuend = deviced(window_size=window_size, n_points=self.n_points)
        joined_paths = None
        for i in range(periods):
            generated_list = [self.generate(condition,generator=generator) for i in range(batch_size)]

            with Pool(processes=n_jobs) as pool:
                result = list(tqdm(pool.imap(self.multi_works,generated_list),total=batch_size))
            paths = [i[0] for i in result]
            condition = [i[1] for i in result][0]

            if joined_paths is None:
                joined_paths = paths
            else:
                for i in range(batch_size):
                    joined_paths[i] = np.r_[joined_paths[i], np.add(paths[i][1:], joined_paths[i][-1])]
        if minuend ==0:
            end_paths = np.array(joined_paths)
        else:
            end_paths = np.array(joined_paths)[:, :-minuend]
        return price + np.transpose(end_paths)
