#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
from pyfinance.ols import OLS


def change_factor_frequency(
        factor_data: pd.DataFrame,
        date_list: None | list | np.ndarray = None,
        change_to: None | str = None
):
    """
    调整因子数据频率

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        因子数据
    date_list : list or np.ndarray, optional
        调仓日期列表，可选，不传入则使用factor_data中出现过的所有日期
    change_to : str, optional, {'M', 'W'}
        转换成什么频率，可选
        * M：调仓日期列表中每月的最后一天
        * W：调仓日期列表中每周的最后一天

    Returns
    -------
        pd.DataFrame - MultiIndex
        频率调整后的因子数据
    """
    factor_data = factor_data.copy()
    if date_list is None:
        date_list = factor_data.index.get_level_values('date').drop_duplicates().sort_values()
    if change_to is not None:
        date_df = pd.DataFrame(date_list)
        if change_to == 'M':
            date_df['mark'] = date_df['date'].dt.strftime('%Y-%m')
            date_df['day'] = date_df['date'].dt.day
        elif change_to == 'W':
            date_df['mark'] = date_df['date'].dt.strftime('%Y-%W')
            date_df['day'] = date_df['date'].dt.strftime('%w').astype('int')
        date_df = date_df[date_df['day'] == date_df.groupby('mark')['day'].transform('max')]
        date_list = date_df['date']
    factor_data = factor_data[factor_data.index.get_level_values('date').isin(date_list)]

    return factor_data


def process_outlier(
        factor_data: pd.DataFrame,
        method: str = 'winsorize',
        factor_list: None | list = None,
        winsorize_fraction: float = 0.01,
        n_sigma: float = 3,
        n_mad: float = 3
):
    """
    处理因子数据中的异常值

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        因子数据
    method : str, {'winsorize', 'sigma', 'mad'}, default 'winsorize'
        处理异常值的方法，默认为'winsorize'
        * winsorize：缩尾法
        * sigma：n-sigma法
        * mad：n-mad法
    factor_list : list, optional
        要进行异常值处理的因子名称，可选，若不传入则处理全部因子
    winsorize_fraction : float, optional
        仅在method=='winsorize'时有效，在缩尾法中确定异常值的分位数。
        默认为0.01（1%分位和99%分位外的数据视为异常值）。
    n_sigma : float, optional
        仅在method=='sigma'时有效，使用多少倍标准差来确定异常值。默认为3。
    n_mad : float, optional
        仅在method=='mad'时有效，使用多少倍MAD来确定异常值。默认为3。

    Returns
    -------
    data : pd.DataFrame - MultiIndex
        处理了异常值以后的因子数据
    """
    data = factor_data.copy()
    factor_list = list(data.columns) if factor_list is None else factor_list
    for factor in factor_list:
        if method == 'winsorize':
            data['upper'] = data.groupby('date')[factor].transform(lambda x: x.quantile(1 - winsorize_fraction))
            data['lower'] = data.groupby('date')[factor].transform(lambda x: x.quantile(winsorize_fraction))
        elif method == 'sigma':
            data['upper'] = data.groupby('date')[factor].transform(lambda x: x.mean() + n_sigma * x.std())
            data['lower'] = data.groupby('date')[factor].transform(lambda x: x.mean() - n_sigma * x.std())
        elif method == 'mad':
            data['upper'] = data.groupby('date')[factor]. \
                transform(lambda x: x.median() + n_mad * (x - x.median()).abs().median())
            data['lower'] = data.groupby('date')[factor]. \
                transform(lambda x: x.median() - n_mad * (x - x.median()).abs().median())
        data.loc[data[factor] > data['upper'], factor] = data['upper']
        data.loc[data[factor] < data['lower'], factor] = data['lower']
        data.drop(columns=['upper', 'lower'], inplace=True)
    return data


def standardize_factor(
        factor_data: pd.DataFrame,
        factor_list: None | list = None,
        suffix: str = '',
):
    """
    对因子进行截面标准化

    Parameters
    ----------
    factor_data : pd.DataFrame
        因子数据
    factor_list : list, optional
        要进行异常值处理的因子名称，可选，若不传入则处理全部因子
    suffix : str, optional
        标准化后因子字段后缀，默认无
        若不传入，则标准化后的因子数据会直接覆盖原始因子数据
        若传入，会在factor_data中生成带指定后缀的标准化后的因子字段

    Returns
    -------
    data : pd.DataFrame - MultiIndex
        标准化后的因子数据
    """
    data = factor_data.copy()
    factor_list = list(data.columns) if factor_list is None else factor_list
    for factor in factor_list:
        data[f'{factor}{suffix}'] = data.groupby('date')[factor].transform(lambda x: (x - x.mean()) / x.std())
    return data


def combine_factors(
    factor_data: pd.DataFrame,
    factor_list: None | list = None,
    method: str = 'equal',
    standardization: bool = True,
) -> pd.Series:
    """
    因子合成

    Parameters
    ----------
    factor_data : pd.DataFrame
        因子数据
    factor_list : list [str], optional
        要进行合成的因子名称列表，可选，若不传入则将全部因子进行合成
    method : str, {'equal', }, default 'equal'
        因子合成的方法。默认为'equal'。
        * equal：等权合成
    standardization : bool, default True
        在因子合成之前是否对单因子进行截面标准化

    Returns
    -------
        pd.Series - MultiIndex
    """
    data = factor_data.copy()
    factor_list = list(data.columns) if factor_list is None else factor_list
    data = data[factor_list]
    if standardization:
        data = standardize_factor(data=data)
    if method == 'equal':
        return data.mean(axis=1)


def neutralize_factors(
        factor_data: pd.DataFrame,
        neutralization_list: list,
):
    """
    中性化时，风险因子为空值的样本会被剔除
    风险因子是定量数据还是分类数据（object、str）取决于dtypes

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        因子数据
    neutralization_list : list [pd.DataFrame]
        用于中性化的风险因子数据

    Returns
    -------
        pd.DataFrame 中性化后的因子
    """
    #
    f_data=factor_data.copy()
    n_list=[i.copy() for i in neutralization_list]

    factor_list = list(f_data.columns)

    for data in [f_data, ] + n_list:
        data=data.reset_index().sort_values(['date','asset'])
    for n_data in n_list:
        f_data = pd.merge_asof(f_data, n_data, on='date', by='asset')
    f_data.set_index(['date', 'asset'], inplace=True)

    risk_list = [i for i in f_data.columns if i not in factor_list]

    df_list = []
    for date, sub_df in f_data.groupby('date'):
        df = sub_df.copy().dropna()
        if df.shape[0] == 0:
            continue
        for risk in risk_list:
            if pd.api.types.is_object_dtype(df[risk]) or pd.api.types.is_string_dtype(df[risk]):
                df = pd.merge(df, pd.get_dummies(df[risk], drop_first=True), left_index=True, right_index=True)
                df.drop(columns=[risk, ], inplace=True)
        y_list = factor_list
        x_list = [i for i in df.columns if i not in factor_list]
        for y in y_list:
            df[y] = OLS(y=df[y], x=df[x_list]).resids
        df = df[y_list]
        df_list.append(df)
    all_df = pd.concat(df_list)
    return all_df
