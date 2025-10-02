"""
数据分析工具库
用于常用的数据分析
antuor:LSing
datetime:2021-09-19 20:38:05
"""
import os
import pandas as pd
from ..basis.toolbasis import BasisTools


def normalization(df, feature):
    """
    数据标准化
    :param df:
    :param feature:
    :return:
    """
    series = df[feature]
    return (series - series.mean()) / (series.std())


def isnull(df):
    """
    判断为空
    :param df:
    :return:
    """
    # pd.isnull(train).values.any()
    return pd.isnull(df)


def skew(df, feature):
    #
    # 基本上，偏度度量了实值随机变量的均值分布的不对称性。让我们计算损失的偏度：
    # stats.mstats.skew(train['loss']).data
    # 对数据进行对数变换通常可以改善倾斜，可以使用 np.log
    # stats.mstats.skew(np.log(train['loss'])).data
    # 连续值特征
    # train[cont_features].hist(bins=50, figsize=(16,12))
    return df[feature].skew


def fill_null(df, feature, type='', fillnum=0):
    """
    填充缺失值
    :param df:
    :param feature:
    :param type:
    :return:
    """
    if type == 'fill_mean':
        df[feature] = df[feature].fillna(df[feature].mean())
    elif type == 'fill_number':
        df[feature] = df[feature].fillna(fillnum)
    elif type == 'fill_newtype':
        df[feature] = df[feature].fillna('Null')


def divide_box_operation(self, df, feature, divide_point=[10, 20, 30]):
    """
    分箱操作
    :param df:
    :param feature:
    :param divide_point:
    :return:
    """
    return pd.cut(df[feature], bins=divide_point)


#
# # 对于额外使用流量进行分箱处理
# bin2=[-2500,-2000,-1000,-500,0,500,1000,2500]
# data['flow_label']=pd.cut(data.extra_flow,bins=bin2)
# data.head()
#
# # 对于额外通话时长进行分箱处理
# bin1=[-3000,-2000,-500,0,500,1000,2000,3000,5000]
# data['time_label']=pd.cut(data.extra_time,bins=bin1)
# data.head()
#

def counts_value(self):
    counts_value = []
    for item in self.columns:
        counts_value.append(self.dataFrame[item].value_counts)
    return counts_value


def add_new_feature(df1, df2, feature_on):
    """
    将两个数据融合在一起
    :param df1: 主要dataframe
    :param df2: 被加dataframe
    :param feature_on: 按照此特征为标准进行
    :return: 返回加好的数据
    """
    return df1.merge(df2, left_on=feature_on, right_on=feature_on, how='left')


def concat_data(paths_list):
    """
    将同一文件中的csv读成DF并合并在一起
    :param file_path:
    :return:
    """
    DF = pd.DataFrame()
    for paths in paths_list:
        data = pd.read_csv(paths)
        DF = DF.append(data)
    return DF


def plot(df, kind='bar', fontsize=15):
    df.plot(kind=kind, fontsize=fontsize)


def nunique(df, feature):
    return df[feature].nunique()


class Descruptive_analysis_tools(BasisTools):
    def __init__(self, df):
        super(Descruptive_analysis_tools, self).__init__()
        self.df = df

    def info(self):
        return self.df.info()

    def describe(self):
        return self.df.describe()

    def head(self, num):
        return self.df.head(num)

    def tail(self, num):
        return self.df.tail(num)

    def sample(self, num):
        return self.df.sample(num)

    def count(self, df):
        '''统计数量'''
        df.value_counts()

    def select_dtypes(self):
        return self.df.select_dtypes('object').describe(), \
               self.df.select_dtypes('float').describe()

    def get_dtype(self, feature):
        return self.df.dtypes[feature]


class Exploratory_analysis_tools(BasisTools):
    def __init__(self, df):
        """
        探索性分析工具
        :param df:
        """
        super(Exploratory_analysis_tools, self).__init__()
        self.df = df

    def corrmat_(self, feature, method='pearson'):
        """
        相关性分析
           # pearson：相关系数来衡量两个数据集合是否在一条线上面，即针对线性数据的相关系数计算，针对非线性数据便会有误差。
           # spearman：非线性的，非正太分析的数据的相关系数
           # kendall：用于反映分类变量相关性的指标，即针对无序序列的相关系数，非正太分布的数据
           # 上面的结果验证了，pearson对线性的预测较好，对于幂函数，预测差强人意。

        :param feature: 特征
        :param method: 方式
        :return: 相关矩阵
        """
        return self.df[feature].corr(method=method)


class Data_dealer(BasisTools):
    def __init__(self):
        super(Data_dealer, self).__init__()
        """"""

    def groupby_agg(self, df, group: list, agg: list):
        """
        分组聚合
        :param df:
        :param group: list
        :param agg: list
        :return:
        """
        return df.groupby(group).agg(agg)

    def groupby_apply(self, df, axis):
        """
        计算总和
        :return:
        """
        return df.apply(lambda x: x.sum(), axis=axis)

    def drop_dup(self, df, label):
        """
        去重
        :param df:
        :return:
        """
        return df.drop_duplicates(label)

    def reset_index(self, df):
        df.index = range(len(df))

    def df2list(self, df, feature):
        return df[feature].tolist()

    def read_necessary(self, unnecess=[]):
        """
        读取csv 去除不需要的字段
        :param path: 路径
        :return: DataFrame
        """
        data = pd.read_csv(self.path)
        for name in unnecess:
            try:
                data = data.drop(name, axis=1)
            except Exception as e:
                print(name, ':', e)
        return data

    def save_csv(self,pd,save_path):
        pd.to_csv(save_path,index=None)

    def set_option(self):
        """
        取消叠行列
        :return:
        """
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)

    def df_is_emply(self,df):
        if df.empty:
            return True
        else:
            return False

    def df_safe_read(self, path):
        if not os.path.exists(path):
            return pd.DataFrame()
        else:
            return pd.read_csv(path)
