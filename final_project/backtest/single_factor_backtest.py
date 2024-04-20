#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import quantstats.stats as qs
import matplotlib.dates as mdates
import datetime
import matplotlib.ticker as ticker

# 使用 seaborn 风格
# plt.style.use('seaborn-v0_8-ticks')
plt.style.use('default')

# # 设置中文字体为楷体
# # mpl.rcParams['font.sans-serif'] = ['KaiTi']
# mpl.rcParams['font.sans-serif'] = 'SimHei'  # 选择一个包含中文字符的字体

# # 解决负号显示问题
# mpl.rcParams['axes.unicode_minus'] = False

# 指定包含中文字符的字体，例如SimHei
plt.rcParams['font.sans-serif'] = 'SimHei'
# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False


# 定义颜色列表
colorslist = ['#63be7b', '#fbfbfe', '#f8696b',]
# 创建颜色映射
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
new_cmap = LinearSegmentedColormap.from_list('new_cmap', colorslist, N=800)

# 导入 FactorCapacity 模块
from factor_capacity import FactorCapacity

class SingleFactor_SinglePool_BackTest:
    def __init__(
            self,
            factor_data,
            price_data,
            benchmark_data,
            pool_data,
            factor_name,
            pool_name,
            start_date=None,
            end_date=None,
            is_daily_factor=True,
            group_data=None,
            direction=1,
    ):
        """
        构造函数，用于初始化FactorCapacity类的实例。

        Parameters
        ----------
        factor_data : pd.DataFrame - MultiIndex (date, asset)
            因子数据。
        price_data : pd.DataFrame - MultiIndex (date, asset)
            价格数据。
        benchmark_data : pd.DataFrame - Index (date)
            基准数据。
        pool_data : pd.DataFrame - MultiIndex (date, asset)
            池子数据。
        factor_name : str
            因子名称。
        pool_name : str
            池子名称。
        start_date : str or None, optional
            开始日期，默认为None。
        end_date : str or None, optional
            结束日期，默认为None。
        is_daily_factor : bool, optional
            是否是日度因子，默认为True。
        group_data : pd.DataFrame, optional
            分组数据，默认为None。
        direction : int, optional
            因子方向，默认为1。
        """

        self.factor_data = factor_data
        self.factor_data.columns = ['factor']
        self.factor_name = factor_name
        self.pool_name = pool_name
        self.price_data = price_data
        self.benchmark_data = benchmark_data
        self.pool_data = pool_data
        self.group_data = group_data
        self.is_daily_factor = is_daily_factor
        self.direction = direction
        self.start_date = pd.Timestamp(start_date) if start_date is not None else None
        self.end_date = pd.Timestamp(end_date) if end_date is not None else None

        self.factor_freq_clean_data = None
        self.daily_freq_clean_data = None
        self.grouped_factor_freq_clean_data = None

        self.factor_capacity = FactorCapacity()

    # 数据预处理
    def generate_clean_data(self, quantiles: int = 5):
        """
        赋值Backtest对象的以下属性：
            因子频率清洗后数据factor_freq_clean_data
            日频清洗后数据daily_freq_clean_data
            带有所属行业的因子频率清洗后数据grouped_factor_freq_clean_data

        Parameters
        ----------
        quantiles : int
            将所有股票按照因子值的大小分为n组

        Returns
        -------
        None
        """
        factor_list = list(self.factor_data.keys())
        for factor in factor_list:
            self.get_factor_freq_clean_data(quantiles=quantiles)
            self.get_daily_freq_clean_data()
            if self.group_data is not None:
                self.get_grouped_factor_freq_clean_data()

    def get_factor_freq_clean_data(self, quantiles: int = 5):
        """
        1.  筛选出所有数据中在指定的起止时间范围内的部分，最终数据的起始时间是所有数据的起始时间和指定起始时间的较晚值，
        结束时间是所有数据的结束时间和指定结束时间的较早值
        2.  筛选出在指定股票池中的股票
        3.  将股票池数据、价格数据的频率转变为和因子数据一致
        4.  计算股票的下一期收益率和因子分组情况

        Parameters
        ----------
        quantiles : int
            将所有股票按照因子值的大小分为n组

        Returns
        -------
        factor_freq_clean_data : pd.DataFrame - MultiIndex
            index : date, asset
            columns : forward_return, factor, factor_quantile
        """
        # 按照起止日期对所有数据进行筛选
        start_date_list = []
        end_date_list = []
        for data in [self.factor_data, self.price_data, self.pool_data, self.benchmark_data]:
            start_date_list.append(data.index.get_level_values('date').min())
            end_date_list.append(data.index.get_level_values('date').max())
        if self.start_date:
            start_date_list.append(self.start_date)
        if self.end_date:
            end_date_list.append(self.end_date)
        self.start_date = max(start_date_list)
        self.end_date = min(end_date_list)

        self.factor_data = self.factor_data[
            (self.factor_data.index.get_level_values('date') >= self.start_date) & (
                        self.factor_data.index.get_level_values('date') <= self.end_date)
            ]
        self.price_data = self.price_data[
            (self.price_data.index.get_level_values('date') >= self.start_date) & (
                        self.price_data.index.get_level_values('date') <= self.end_date)
            ]
        self.pool_data = self.pool_data[
            (self.pool_data.index.get_level_values('date') >= self.start_date) & (
                        self.pool_data.index.get_level_values('date') <= self.end_date)
            ]
        self.benchmark_data = self.benchmark_data[
            (self.benchmark_data.index.get_level_values('date') >= self.start_date) & (
                        self.benchmark_data.index.get_level_values('date') <= self.end_date)
            ]
        # 将股票池数据的频率转变为和因子频率一致
        self.pool_data = self.pool_data[
            self.pool_data.index.get_level_values('date').isin(self.factor_data.index.get_level_values('date'))]
        # 筛选出在股票池里的股票
        factor_array = self.factor_data.copy()
        factor_array = factor_array[factor_array.index.isin(self.pool_data.index)]
        # 将价格数据的频率转变为和因子频率一致
        price_array = self.price_data.copy().reset_index().pivot(index='date', columns='asset', values='price')
        price_array = price_array[price_array.index.isin(factor_array.index.get_level_values('date').unique())]
        # 未来1期收益率
        forward_returns = price_array.pct_change(1).shift(-1)
        forward_returns = forward_returns.stack().to_frame().rename({0: 'forward_return'}, axis=1)
        # 拼接因子数据和未来收益率数据
        factor_freq_clean_data = forward_returns.copy()
        factor_freq_clean_data['factor'] = factor_array
        factor_freq_clean_data = factor_freq_clean_data.dropna()

        # 生成因子分组
        def quantile_calc(x, _quantiles):
            return pd.qcut(x, _quantiles, labels=False, duplicates='drop') + 1

        factor_quantile = factor_freq_clean_data.groupby('date')['factor'].transform(quantile_calc, quantiles)
        factor_quantile.name = 'factor_quantile'
        factor_freq_clean_data['factor_quantile'] = factor_quantile.dropna()
        factor_freq_clean_data = factor_freq_clean_data.dropna()

        self.factor_freq_clean_data = factor_freq_clean_data

    def get_daily_freq_clean_data(self):
        """
        针对因子频率不是日频的因子，生成日频的clean_data，便于计算每日净值和收益率序列

        Returns
        -------
        daily_freq_clean_data : pd.DataFrame - MultiIndex
            index : date, asset
            columns : forward_return, factor, factor_quantile
        """
        if self.is_daily_factor:
            self.daily_freq_clean_data = self.factor_freq_clean_data.copy()
        else:
            quantile_data = self.factor_freq_clean_data.copy()
            quantile_data = quantile_data.reset_index().pivot(index='date', columns='asset',
                                                              values='factor_quantile').fillna(0).stack()

            factor_data = self.factor_freq_clean_data.copy()
            factor_data = factor_data.reset_index().pivot(index='date', columns='asset', values='factor').stack()

            # 获取未来1天的收益
            price_array = self.price_data.copy().reset_index().pivot(index='date', columns='asset', values='price')
            # 未来1期收益率
            forward_returns = price_array.pct_change(1).shift(-1)
            forward_returns = forward_returns.stack().to_frame().rename({0: 'forward_return'}, axis=1)

            daily_freq_clean_data = forward_returns.copy()
            daily_freq_clean_data['factor'] = factor_data
            daily_freq_clean_data['factor_quantile'] = quantile_data
            daily_freq_clean_data = daily_freq_clean_data.sort_index(level=['asset', 'date']).groupby(['asset']).ffill()
            daily_freq_clean_data = daily_freq_clean_data.dropna(subset='factor_quantile')
            daily_freq_clean_data['factor_quantile'] = daily_freq_clean_data['factor_quantile'].astype('int')
            daily_freq_clean_data = daily_freq_clean_data[daily_freq_clean_data['factor_quantile'] != 0]

            self.daily_freq_clean_data = daily_freq_clean_data

    def get_grouped_factor_freq_clean_data(self):
        """
        生成带有每只股票每期所属行业字段的clean_data

        Returns
        -------
        grouped_factor_freq_clean_data : pd.DataFrame - MultiIndex
            index : date, asset
            columns : forward_return, factor, factor_quantile, group
        """
        factor_data = self.factor_freq_clean_data.copy().reset_index().sort_values(['date', 'asset'])
        group_data = self.group_data.copy().reset_index().sort_values(['date', 'asset'])

        grouped_factor_freq_clean_data = pd.merge_asof(factor_data, group_data, on='date', by='asset').set_index(
            ['date', 'asset']).sort_index(level=['asset', 'date'])
        grouped_factor_freq_clean_data['group'] = grouped_factor_freq_clean_data.groupby('asset')[
            'group'].ffill().bfill()

        self.grouped_factor_freq_clean_data = grouped_factor_freq_clean_data

    # 因子覆盖率分析
    def get_factor_coverage(self, group=False):
        """
        获取因子频率的因子覆盖率序列（有因子的股票数占当期股票池股票数的比例）

        Parameters
        ----------
        group : boolean, default False
            是否分行业计算因子覆盖率

        Returns
        -------
        result : pd.DataFrame
            index : date
            columns : 因子覆盖率（若分行业，则每一列是每一个行业的因子覆盖率）
        """
        pool_array = self.pool_data.copy()
        factor_array = self.factor_data.copy()
        if group:
            factor_array = factor_array.reset_index().sort_values(['date', 'asset'])
            group_array = self.group_data.copy().reset_index().sort_values(['date', 'asset'])
            factor_array = pd.merge_asof(factor_array, group_array, on='date', by='asset').set_index(
                ['date', 'asset']).sort_index(level=['asset', 'date'])
            factor_array['group'] = factor_array.groupby('asset')['group'].ffill().bfill()
        df = pd.merge(pool_array, factor_array, on=['date', 'asset'], how='left')
        df['计数'] = 1
        if group:
            grouper = ['date', 'group']
        else:
            grouper = ['date']
        df['累计计数'] = df.groupby(grouper)['计数'].cumsum()
        result = pd.DataFrame()
        result['因子覆盖股数'] = df.groupby(grouper)['factor'].count()
        result['当期股票池股数'] = df.groupby(grouper)['累计计数'].max()
        result['因子覆盖率'] = result['因子覆盖股数'] / result['当期股票池股数']
        if group:
            result = result[['因子覆盖率']].reset_index().pivot(index='date', columns='group', values='因子覆盖率')
        else:
            result = result[['因子覆盖率']]

        self.factor_capacity.factor_coverage_array = result.copy()
        return result

    def plot_factor_coverage(self):
        coverage = self.get_factor_coverage()
        fig, ax1 = plt.subplots(figsize=(16, 8))
        ax1.plot(coverage.index, coverage['因子覆盖率'], color='red', label='因子覆盖率')
        ax1.set_xticks(
            coverage.loc[coverage.groupby(coverage.index.year)['因子覆盖率'].cumcount() == 1].index,
            coverage.loc[coverage.groupby(coverage.index.year)['因子覆盖率'].cumcount() == 1].index.strftime("%Y")
        )
        ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax1.set_ylim(0, 1.1)

    # 因子描述性统计
    def analyse_factor_descriptive_statistics(self, by='quantile'):
        """
        按quantile对因子进行描述性统计

        Parameters
        ----------
        factor : str
            因子名称

        Returns
        -------
        result : pd.DataFrame
            index : 分组
            columns : 统计指标
        """
        if by == 'quantile':
            data1 = self.factor_freq_clean_data.copy()[['factor_quantile', 'factor']]
            data2 = data1.copy()
            data2['factor_quantile'] = '总体'
            data = pd.concat([data1, data2])
            grouper = 'factor_quantile'
        elif by == 'year':
            data1 = self.factor_freq_clean_data.copy()[['factor']]
            data2 = data1.copy()
            data1['year'] = data1.index.get_level_values('date').year
            data2['year'] = '总体'
            data = pd.concat([data1, data2])
            grouper = 'year'
        result = pd.DataFrame()
        result['样本量'] = data.groupby(grouper)['factor'].count()
        result['均值'] = data.groupby(grouper)['factor'].mean()
        result['标准差'] = data.groupby(grouper)['factor'].std()
        result['偏度'] = data.groupby(grouper)['factor'].skew()
        result['峰度'] = data.groupby(grouper)['factor'].agg(lambda x: x.kurt())
        result['最小值'] = data.groupby(grouper)['factor'].min()
        result['p10'] = data.groupby(grouper)['factor'].quantile(0.1)
        result['p25'] = data.groupby(grouper)['factor'].quantile(0.25)
        result['p50'] = data.groupby(grouper)['factor'].median()
        result['p75'] = data.groupby(grouper)['factor'].quantile(0.75)
        result['p90'] = data.groupby(grouper)['factor'].quantile(0.9)
        result['最大值'] = data.groupby(grouper)['factor'].max()
        result['中位数绝对偏差'] = data.groupby(grouper)['factor'].agg(lambda x: (x - x.median()).abs().median())
        return result

    def plot_factor_distribution(self):
        """
        绘制因子分布直方图和密度图

        Parameters
        ----------
        factor : str
            因子名称

        Returns
        -------
        """
        data = self.factor_freq_clean_data.copy()['factor']
        fig, ax1 = plt.subplots(figsize=(16, 8))
        ax1 = sns.histplot(data=data, kde=True)
        ax1.set_xlim(left=data.quantile(0.05), right=data.quantile(0.95))

    # 因子IC分析
    def get_ic(self):
        """
        计算因子IC序列

        Returns
        -------
        ic : pd.Series
            index : date
            value : 当期IC值
        """
        ic = self.factor_freq_clean_data.copy()
        ic = ic[['forward_return', 'factor']].groupby(ic.index.get_level_values('date')).corr(method='spearman')
        ic = ic[ic.index.get_level_values(-1) == 'factor'].droplevel(-1)['forward_return'].rename('IC')
        self.factor_capacity.ic_array = ic
        return ic

    def analyse_ic(self):
        """
        分析因子IC

        Returns
        -------
        ic_summary_table : pd.DataFrame
            index : 暂时无意义
            columns : 分析指标
        """

        ic = self.get_ic()

        ic_summary_table = pd.DataFrame([{
            "IC均值": ic.mean(),
            "IC标准差": ic.std(),
            "ICIR": ic.mean() / ic.std(),
            't-statistic': stats.ttest_1samp(ic, 0, nan_policy='omit')[0],
            'p-value': stats.ttest_1samp(ic, 0, nan_policy='omit')[1],
            'IC偏度': ic.skew(),
            'IC峰度': ic.kurt(),
            'IC胜率': len(ic[ic > 0]) / len(ic) if self.direction == 1 else len(ic[ic < 0]) / len(ic)
        }])
        ic_summary_table.index = [self.factor_name, ]

        self.factor_capacity.ic_summary = ic_summary_table

        return ic_summary_table

    def get_quantile_ic(self):
        """
        计算因子的分组序号和该组下期收益的秩相关系数

        Returns
        -------
        ic : pd.Series
            index : date
            value : 当期quantile_IC值
        """
        ic = self.get_quantile_return_data().reset_index(level=0)
        ic = ic.groupby(ic.index.get_level_values('date')).corr(method='spearman')
        ic = ic[ic.index.get_level_values(-1) == 'factor_quantile'].droplevel(-1)['forward_return'].rename(
            'quantile_IC')
        self.factor_capacity.quantile_ic_array = ic
        return ic

    def analyse_quantile_ic(self):
        """
        分析因子quantile_IC（分组序号和该组下期收益的秩相关系数） 【原创指标，用于分析各分组收益的单调性】

        Returns
        -------
        ic_summary_table : pd.DataFrame
            index : 暂时无意义
            columns : 分析指标
        """
        ic = self.get_quantile_ic()

        ic_summary_table = pd.DataFrame([{
            "q_IC均值": ic.mean(),
            "q_IC标准差": ic.std(),
            "ICIR": ic.mean() / ic.std(),
            't-statistic': stats.ttest_1samp(ic, 0, nan_policy='omit')[0],
            'p-value': stats.ttest_1samp(ic, 0, nan_policy='omit')[1],
            'q_IC偏度': ic.skew(),
            'q_IC峰度': ic.kurt(),
            'q_IC胜率': len(ic[ic > 0]) / len(ic) if self.direction == 1 else len(ic[ic < 0]) / len(ic)
        }])
        ic_summary_table.index = [self.factor_name, ]
        self.factor_capacity.quantile_ic_summary = ic_summary_table
        return ic_summary_table

    def get_grouped_ic(self):
        """
        分行业计算因子IC序列

        Returns
        -------
        ic : pd.DataFrame
            index : date
            columns : 各行业
            value : 各行业当期IC值
        """
        ic = self.grouped_factor_freq_clean_data.copy().set_index('group', append=True)
        ic = ic[['forward_return', 'factor']].groupby(
            [ic.index.get_level_values('date'), ic.index.get_level_values('group')]).corr(method='spearman')
        ic = ic[ic.index.get_level_values(-1) == 'factor'].droplevel(-1)['forward_return'].rename('IC').unstack(
            level=-1)
        self.factor_capacity.grouped_ic_array = ic
        return ic

    def analyse_grouped_ic(self):
        """
        分析各行业的因子IC

        Returns
        -------
        ic_summary_table : pd.DataFrame
            index : 各行业
            columns : 分析指标
        """
        ic = self.get_grouped_ic()

        ic_summary_table = pd.DataFrame({
            "IC均值": ic.mean(),
            "IC标准差": ic.std(),
            "ICIR": ic.mean() / ic.std(),
            't-statistic': stats.ttest_1samp(ic, 0, nan_policy='omit')[0],
            'p-value': stats.ttest_1samp(ic, 0, nan_policy='omit')[1],
            'IC偏度': ic.skew(),
            'IC峰度': ic.kurt(),
            'IC胜率': [len(ic[i][ic[i] > 0]) / len(ic[i]) if self.direction == 1 else len(ic[i][ic[i] < 0]) / len(ic[i])
                       for i in ic.columns]
        })
        return ic_summary_table

    def plot_grouped_ic(self):
        """
        绘制分行业IC柱形图

        Returns
        -------

        """
        grouped_ic = self.analyse_grouped_ic().sort_values('IC均值', ascending=False)
        fig, ax1 = plt.subplots(figsize=(16, 8))
        ax1 = sns.barplot(data=grouped_ic, x=grouped_ic['IC均值'], y=grouped_ic.index, color='#9e413e')
        ax1.set_xlim(-round(grouped_ic['IC均值'].abs().max() * 1.05, 1),
                     round(grouped_ic['IC均值'].abs().max() * 1.05, 1))
        ax1.bar_label(ax1.containers[0], fmt='%.3g')

    def analyse_ic_decay(self, max_lag=10):
        """
        分析IC衰减情况

        Parameters
        ----------
        max_lag : int, default 10
            计算IC衰退的最大期限

        Returns
        -------
        result : pd.DataFrame
            index : 衰退期数
            columns : 对应衰退期的IC、ICIR
        """
        ic = self.factor_freq_clean_data.copy().sort_index(level=['asset', 'date'])
        for i in range(1, max_lag + 1):
            ic[f'forward_return_F{i}'] = ic.groupby(ic.index.get_level_values('asset'))['forward_return'].shift(-i)
        ic = ic.groupby(ic.index.get_level_values('date')).corr(method='spearman')
        ic = ic[ic.index.get_level_values(-1) == 'factor'].droplevel(-1)[
            [i for i in list(ic.columns) if 'forward' in i]]
        ic.columns = [str(i) for i in range(0, max_lag + 1)]

        result = pd.DataFrame()
        result['IC'] = ic.mean()
        result['ICIR'] = ic.mean() / ic.std()

        return result

    def plot_ic_dacay(self, max_lag=10):
        """
        绘制IC衰减柱形图

        Parameters
        ----------
        max_lag : int, default 10
            计算IC衰退的最大期限

        Returns
        -------
        """
        decay = self.analyse_ic_decay(max_lag=max_lag)
        fig, ax1 = plt.subplots(figsize=(8, 8))
        ax2 = ax1.twinx()
        x = np.arange(len(decay.index))
        width = 0.35
        ax1.bar(x - width / 2, decay['IC'], width, label='IC')
        ax1.set_ylim(top=max(decay['IC'].abs()) * 1.05, bottom=-max(decay['IC'].abs()) * 1.05)
        ax1.axhline(0, color='black', linewidth=1)
        ax1.set_xticks(np.arange(len(decay.index)))
        ax2.bar(x + width / 2, decay['ICIR'], width, label='ICIR', color='red')
        ax2.set_ylim(top=max(decay['ICIR'].abs()) * 1.05, bottom=-max(decay['ICIR'].abs()) * 1.05)
        ax2.axhline(0, color='black', linewidth=1)
        ax2.set_xticks(np.arange(len(decay.index)))
        fig.legend(loc='lower center', bbox_transform=ax1.transAxes, ncol=2, bbox_to_anchor=(0, 0, 1, 1), frameon=False)

    def plot_ic(self, bar_figure=False):
        '''
        如果是周频及更细的因子就不要绘制柱形图了
        '''
        ic_array = self.get_ic().to_frame()

        if bar_figure:
            fig, ax1 = plt.subplots(figsize=(8, 8))
            ax2 = ax1.twinx()
            ax1.bar(ic_array.index, ic_array['IC'], color='#2F2F2F', width=20, label='IC')
            ax2.plot(ic_array.index, ic_array['IC'].cumsum(), color='red', label='Accumulated_IC')
            ax1.set_xticks(
                ic_array.loc[ic_array.groupby(ic_array.index.year)['IC'].cumcount() == 1].index,
                ic_array.loc[ic_array.groupby(ic_array.index.year)['IC'].cumcount() == 1].index.strftime("%Y")
            )
            ax1.axhline(0, color='black', linewidth=1)
            fig.legend(loc='lower center', bbox_transform=ax1.transAxes, ncol=2, bbox_to_anchor=(0, 0, 1, 1),
                       frameon=False)
        else:
            fig, ax1 = plt.subplots(figsize=(8, 8))
            ax1.plot(ic_array.index, ic_array['IC'].cumsum(), color='red', label='Accumulated_IC')
            ax1.set_xticks(
                ic_array.loc[ic_array.groupby(ic_array.index.year)['IC'].cumcount() == 1].index,
                ic_array.loc[ic_array.groupby(ic_array.index.year)['IC'].cumcount() == 1].index.strftime("%Y")
            )
            fig.legend(loc='lower center', bbox_transform=ax1.transAxes, ncol=2, bbox_to_anchor=(0, 0, 1, 1),
                       frameon=False)

    def plot_quantile_ic(self, bar_figure=False):
        '''
        如果是周频及更细的因子就不要绘制柱形图了
        '''
        ic_array = self.get_quantile_ic().to_frame()

        if bar_figure:
            fig, ax1 = plt.subplots(figsize=(8, 8))
            ax2 = ax1.twinx()
            ax1.bar(ic_array.index, ic_array['quantile_IC'], color='#2F2F2F', width=20, label='quantile_IC')
            ax2.plot(ic_array.index, ic_array['quantile_IC'].cumsum(), color='red', label='Accumulated_IC(Right)')
            ax1.set_xticks(
                ic_array.loc[ic_array.groupby(ic_array.index.year)['quantile_IC'].cumcount() == 1].index,
                ic_array.loc[ic_array.groupby(ic_array.index.year)['quantile_IC'].cumcount() == 1].index.strftime("%Y")
            )
            ax1.axhline(0, color='black', linewidth=1)
            fig.legend(loc='lower center', bbox_transform=ax1.transAxes, ncol=2, bbox_to_anchor=(0, 0, 1, 1),
                       frameon=False)
        else:
            fig, ax1 = plt.subplots(figsize=(8, 8))
            ax1.plot(ic_array.index, ic_array['quantile_IC'].cumsum(), color='red', label='Accumulated_IC(Right)')
            ax1.set_xticks(
                ic_array.loc[ic_array.groupby(ic_array.index.year)['quantile_IC'].cumcount() == 1].index,
                ic_array.loc[ic_array.groupby(ic_array.index.year)['quantile_IC'].cumcount() == 1].index.strftime("%Y")
            )
            fig.legend(loc='lower center', bbox_transform=ax1.transAxes, ncol=2, bbox_to_anchor=(0, 0, 1, 1),
                       frameon=False)

    # 因子自相关性和换手率分析
    def get_factor_autocorrelation(self, max_lag=5):
        """
        生成因子自相关性序列

        Parameters
        ----------
        max_lag : int, default 10
            计算自相关性的最大回看期数

        Returns
        -------
        result : pd.DataFrame
            index : date
            columns : 不同回看期数
        """
        ac_data = self.factor_freq_clean_data.copy()
        ac_data = ac_data[['factor']].reset_index().pivot(index='date', columns='asset', values='factor')
        ac_array = pd.DataFrame()
        for lag in range(1, max_lag + 1):
            for i in range(lag, len(ac_data)):
                ac_array.loc[ac_data.index[i], f'{lag}阶自相关系数'] = ac_data.iloc[i].corr(ac_data.iloc[i - lag],
                                                                                            method='spearman')
        self.factor_capacity.autocorrelation_array = ac_array
        return ac_array

    def analyse_factor_autocorrelation(self, max_lag=5):
        """
        分析因子自相关性

        Parameters
        ----------
        max_lag : int, default 10
            计算自相关性的最大回看期数

        Returns
        -------
        result : pd.DataFrame
            index : 不同阶数
            columns : 因子自相关系数的时序均值、标准差
        """
        ac_array = self.get_factor_autocorrelation(max_lag=max_lag)
        result = pd.DataFrame()
        result['均值'] = ac_array.mean()
        result['标准差'] = ac_array.std()
        return result

    def get_factor_turnover(self, used_factor_freq=True):
        """
        生成因子换手率序列

        Parameters
        ----------
        used_factor_freq : Boolean, default True
            是否以因子的频率为换手的频率计算换手率（直接调用该方法时使用默认True即可）

        Returns
        -------
        turnover : pd.Series - MultiIndex
            index : factor_quantile, date
            value : 当期换手率
        """
        if used_factor_freq:
            turnover = self.factor_freq_clean_data.copy()
        else:
            turnover = self.daily_freq_clean_data.copy()
        turnover = turnover.reset_index(level=-1).set_index('factor_quantile', append=True)
        turnover = turnover.groupby(level=['date', 'factor_quantile']).agg(lambda x: set(x)).sort_index(level=[1, 0])
        turnover['last_asset'] = turnover.groupby(turnover.index.get_level_values('factor_quantile'))['asset'].shift(1)
        turnover['new_names'] = (turnover['asset'] - turnover['last_asset']).dropna()
        turnover['turnover'] = turnover['new_names'].map(lambda x: len(x) if x is not np.nan else 1) / turnover[
            'last_asset'].map(lambda x: len(x) if x is not np.nan else 1)
        turnover = turnover.swaplevel('date', 'factor_quantile')['turnover']
        self.factor_capacity.turnover_array = turnover
        return turnover

    def analyse_factor_turnover(self, used_factor_freq=True):
        """
        分析因子各分组的换手率

        Parameters
        ----------
        used_factor_freq : Boolean, default True
            是否以因子的频率为换手的频率计算换手率（直接调用该方法时使用默认True即可）

        Returns
        -------
        result : pd.DataFrame
            index : factor_quantile
            columns : 换手率均值、标准差
        """
        turnover = self.get_factor_turnover(used_factor_freq=used_factor_freq).to_frame()
        turnover['count'] = turnover.groupby('factor_quantile').cumcount()
        turnover = turnover.loc[turnover['count'] != 0, 'turnover']
        result = pd.DataFrame()
        result['均值'] = turnover.groupby('factor_quantile').mean()
        result['标准差'] = turnover.groupby('factor_quantile').std()
        return result

    # 因子多头和空头组合的行业分布情况分析（必须在创建backtest实例时传入group_data参数才能使用）
    def analyse_factor_group_distribution(self, long=True):
        """
        计算多头和空头组合中各个行业的占比情况

        Parameters
        ----------
        long : Boolean, default True
            默认分析多头组合，改为False分析空头组合

        Returns
        -------
        pd.DataFrame
            index : group
            columns : 该行业在多/空头组合中占比的均值、标准差
        """
        group = self.grouped_factor_freq_clean_data.copy()
        group = group.reset_index('asset', drop=True)[['factor_quantile', 'group']].set_index('factor_quantile',
                                                                                              append=True)
        group = group.groupby(['date', 'factor_quantile']).value_counts(normalize=True)

        max_group = group.loc[
            group.index.get_level_values('factor_quantile') == group.index.get_level_values('factor_quantile').max()]
        max_grouper = max_group.groupby('group')
        max_group_distribution = pd.DataFrame({
            '均值': max_grouper.mean(),
            '标准差': max_grouper.std()
        })
        max_group_distribution.sort_values('均值', ascending=False, inplace=True)

        min_group = group.loc[
            group.index.get_level_values('factor_quantile') == group.index.get_level_values('factor_quantile').min()]
        min_grouper = min_group.groupby('group')
        min_group_distribution = pd.DataFrame({
            '均值': min_grouper.mean(),
            '标准差': min_grouper.std()
        })
        min_group_distribution.sort_values('均值', ascending=False, inplace=True)

        if self.direction == 1:
            if long:
                return max_group_distribution
            else:
                return min_group_distribution
        else:
            if long:
                return min_group_distribution
            else:
                return max_group_distribution

    def analyse_factor_group_distribution_topN_per_year(self, long=True, display_num=5):
        """
        分析每年多头和空头组合中占比最高的n个行业

        Parameters
        ----------
        long : Boolean, default True
            默认分析多头组合，改为False分析空头组合
        display_num : int, default 5
            展示占比最高的前n个行业

        Returns
        -------
        pd.DataFrame
            index : rank
            columns : date（年份）
            value : 行业名
        """
        group = self.grouped_factor_freq_clean_data.copy()
        group = group.reset_index('asset', drop=True)[['factor_quantile', 'group']].set_index('factor_quantile',
                                                                                              append=True)
        group = group.groupby([group.index.get_level_values('date').year, 'factor_quantile']).value_counts(
            normalize=True)

        max_group = group.loc[group.index.get_level_values('factor_quantile') == group.index.get_level_values(
            'factor_quantile').max()].to_frame()
        max_group = max_group.rename(columns={0: 'proportion'})
        max_group['rank'] = max_group.groupby('date')['proportion'].rank(method='first', ascending=False).astype('int')
        max_group = max_group.reset_index().pivot(index='rank', columns='date', values='group')
        max_group = max_group.head(display_num)

        min_group = group.loc[group.index.get_level_values('factor_quantile') == group.index.get_level_values(
            'factor_quantile').min()].to_frame()
        min_group = min_group.rename(columns={0: 'proportion'})
        min_group['rank'] = min_group.groupby('date')['proportion'].rank(method='first', ascending=False).astype('int')
        min_group = min_group.reset_index().pivot(index='rank', columns='date', values='group')
        min_group = min_group.head(display_num)

        if self.direction == 1:
            if long:
                return max_group
            else:
                return min_group
        else:
            if long:
                return min_group
            else:
                return max_group

    def plot_factor_group_distribution(self, long=True):
        group_distribution = self.analyse_factor_group_distribution(long=long)
        fig, ax1 = plt.subplots(figsize=(8, 8))
        ax1.barh(group_distribution.index, group_distribution['均值'])
        ax1.invert_yaxis()

    # 因子分组收益分析
    def get_quantile_return_data(self, commission=0, used_factor_freq=True):
        """
        计算每一期每一组的下期收益

        Parameters
        ----------
        commission : float, default 0
            佣金，双边收取，万分之一则输入1/10000
        used_factor_freq : bool, default True
            若为True，则计算因子频率下的下期收益，若为False，则计算日度频率下的明日收益

        Returns
        -------
        return_array : pd.DataFrame
            index : factor_quantile, date
            columns : forward_return
        """

        if used_factor_freq:
            factor_data = self.factor_freq_clean_data.copy()
        else:
            factor_data = self.daily_freq_clean_data.copy()

        grouper = ['factor_quantile']
        grouper.append(factor_data.index.get_level_values('date'))

        return_array = factor_data.groupby(grouper).agg({'forward_return': 'mean'})
        return_array['turnover'] = self.get_factor_turnover(used_factor_freq=used_factor_freq)
        return_array['forward_return'] = return_array['forward_return'] - return_array[
            'turnover'] * commission * 2  # 双边收取

        return_array = return_array[['forward_return']]

        return return_array

    def get_benchmark_return_array(self, used_factor_freq=True):
        """
        生成基准组合收益率序列

        Parameters
        ----------
        used_factor_freq : bool, default True
            若为True，则计算因子频率下的当期收益，若为False，则计算日度频率下的当日收益

        Returns
        -------
        benchmark_return : pd.DataFrame
            index : factor_quantile, date
            columns : forward_return
        """
        benchmark_return = self.benchmark_data.copy()
        if used_factor_freq:
            benchmark_return = benchmark_return[
                benchmark_return.index.isin(self.factor_freq_clean_data.index.get_level_values('date'))]
        else:
            benchmark_return = benchmark_return[
                benchmark_return.index.isin(self.daily_freq_clean_data.index.get_level_values('date'))]
        benchmark_return = benchmark_return.pct_change(1)
        return benchmark_return

    def get_quantile_return_array(self, commission=0, excess_return=False, used_factor_freq=True):
        """
        生成各因子组合收益率序列

        Parameters
        ----------
        commission : float, default 0
            佣金，双边收取，万分之一则输入1/10000
        excess_return : bool, default False
            若为True，则计算因子组合相对基准超额收益率，若为False，则计算因子组合绝对收益率
        used_factor_freq : bool, default True
            若为True，则计算因子频率下的当期收益，若为False，则计算日度频率下的当日收益

        Returns
        -------
        return_data : pd.DataFrame
            index : date
            columns : 各分组、基准组合、多空组合
            value : 当期收益率
        """
        return_data = self.get_quantile_return_data(commission=commission,
                                                    used_factor_freq=used_factor_freq).reset_index(level=0).pivot(
            columns='factor_quantile', values='forward_return')
        return_data = return_data.shift(1)
        return_data['benchmark'] = self.get_benchmark_return_array(used_factor_freq=used_factor_freq)

        max_quantile = return_data.columns[:-1].max()
        min_quantile = return_data.columns[:-1].min()
        if self.direction == 1:
            return_data['long_short'] = return_data[max_quantile] - return_data[min_quantile]
        else:
            return_data['long_short'] = return_data[min_quantile] - return_data[max_quantile]

        return_data.iloc[0] = 0

        if excess_return:
            for i in return_data.columns:
                if i != 'benchmark' and i != 'long_short':
                    return_data[i] = return_data[i] - return_data['benchmark']

        return return_data

    def get_net_value_array(self, commission=0, excess_return=False, used_factor_freq=True):
        """
        生成各因子组合净值序列

        Parameters
        ----------
        commission : float, default 0
            佣金，双边收取，万分之一则输入1/10000
        excess_return : bool, default False
            若为True，则计算因子组合相对基准净值，若为False，则计算因子组合绝对净值
        used_factor_freq : bool, default True
            若为True，则计算因子频率下的净值，若为False，则计算日度频率下的净值

        Returns
        -------
        return_data : pd.DataFrame
            index : date
            columns : 各分组、基准组合、多空组合
            value : 当期净值
        """
        return_data = self.get_quantile_return_array(commission=commission, excess_return=excess_return,
                                                     used_factor_freq=used_factor_freq)

        nav_array = (return_data + 1).cumprod()
        return nav_array

    def get_single_net_value_array(self, nv_type, commission=0, excess_return=False, used_factor_freq=True):
        nav_array = self.get_net_value_array(commission=commission, excess_return=excess_return,
                                             used_factor_freq=used_factor_freq)

        if nv_type == 'ls':
            return nav_array['long_short']
        elif nv_type == 'l':
            if self.direction == 1:
                return nav_array[nav_array.columns[-3]]
            else:
                return nav_array[nav_array.columns[0]]
        elif nv_type == 's':
            if self.direction == 1:
                return nav_array[nav_array.columns[0]]
            else:
                return nav_array[nav_array.columns[-3]]
        elif isinstance(nv_type, int):
            return nav_array[nv_type]

    def analyse_return_array(self, commission=0):
        """
        因子各分组、基准、多空收益评估

        Parameters
        ----------
        commission : float, default 0
            佣金，双边收取，万分之一则输入1/10000

        Returns
        -------
        result : pd.DataFrame
            index : 因子分组、基准组合、多空组合
            columns : 收益评估指标
        """

        f_return_array = self.get_quantile_return_array(commission=commission)
        d_return_array = self.get_quantile_return_array(commission=commission, used_factor_freq=False)
        f_excess_return_array = self.get_quantile_return_array(commission=commission,
                                                               excess_return=True).drop(['benchmark', ], axis=1)
        d_excess_return_array = self.get_quantile_return_array(commission=commission, excess_return=True,
                                                               used_factor_freq=False).drop(['benchmark', ], axis=1)
        # d_price_array=self.get_net_value_array(commission=commission,used_factor_freq=False)

        result = pd.DataFrame()

        # 绝对收益分析
        result['年化收益'] = qs.cagr(d_return_array, periods=240)
        result['年化波动率'] = qs.volatility(d_return_array, periods=240)
        result['夏普比率'] = (result['年化收益'] - 0.015) / result['年化波动率']
        result['最大回撤'] = qs.max_drawdown(d_return_array)
        result['卡玛比率'] = -result['年化收益'] / result['最大回撤']

        # 超额收益分析
        result['超额年化收益'] = qs.cagr(d_excess_return_array, periods=240)
        result['超额年化波动率'] = qs.volatility(d_excess_return_array, periods=240)
        result['信息比率'] = result['超额年化收益'] / result['超额年化波动率']
        result['超额最大回撤'] = qs.max_drawdown(d_excess_return_array)
        result['超额收益卡玛比率'] = -result['超额年化收益'] / result['超额最大回撤']
        result['相对基准胜率'] = f_excess_return_array.agg(lambda x: len(x[x > 0]) / len(x))
        whole = f_return_array.copy().iloc[1:, :-2]
        result['相对整体胜率'] = whole.agg(lambda x: len(x[x == whole.max(axis=1)]) / len(x))
        result['盈亏比'] = f_excess_return_array.agg(lambda x: x[x > 0].mean() / x[x < 0].abs().mean())

        return result

    def analyse_return_briefly(self, commission=0):
        detailed_table = self.analyse_return_array(commission=commission)
        single_pofo = detailed_table.drop(index=['benchmark', 'long_short'])
        if self.direction == 1:
            sub_table1 = single_pofo.loc[
                single_pofo.index[[-1]], ['年化收益', '夏普比率', '超额年化收益', '信息比率', '超额最大回撤',
                                          '相对基准胜率', '盈亏比']]
        else:
            sub_table1 = single_pofo.loc[
                single_pofo.index[[0]], ['年化收益', '夏普比率', '超额年化收益', '信息比率', '超额最大回撤',
                                         '相对基准胜率', '盈亏比']]
        sub_table1.index = [self.factor_name, ]
        sub_table2 = detailed_table.loc[['long_short'], ['年化收益', '夏普比率', '最大回撤']]
        sub_table2.columns = ['多空年化收益', '多空夏普比率', '多空最大回撤']
        sub_table2.index = [self.factor_name, ]
        result_table = pd.concat([sub_table1, sub_table2], axis=1)
        self.factor_capacity.return_summary = result_table
        return result_table

    def plot_annual_return_heatmap(self, commission=0, excess_return=False):
        """
        绘制逐年收益热力图

        Parameters
        ----------
        commission : float, default 0
            佣金，双边收取，万分之一则输入1/10000

        Returns
        -------
        result : pd.DataFrame
            index : 因子分组、基准组合、多空组合
            columns : 收益评估指标
        """
        return_array = self.get_quantile_return_array(commission=commission, excess_return=excess_return,
                                                      used_factor_freq=False)
        return_array = return_array.groupby(return_array.index.year).agg(lambda x: (1 + x).prod() - 1)
        fig, ax1 = plt.subplots(figsize=(16, 8))
        ax1 = sns.heatmap(
            data=return_array.iloc[:, :-2].rank(axis=1, pct=True),
            cmap=new_cmap,
            annot=return_array.iloc[:, :-2],
            fmt='.2%',
            annot_kws={'color': 'black'},
            cbar=False,
        )
        fig.subplots_adjust(left=0.25, bottom=0.5)
        ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)

    def plot_quantile_annualized_return(self, commission=0, excess_return=True):
        '''
        绘制各分组年化（超额）收益柱形图
        '''
        return_array = self.get_quantile_return_array(commission=commission, excess_return=excess_return,
                                                      used_factor_freq=False)
        result = pd.DataFrame()
        result['年化收益'] = qs.cagr(return_array, periods=240)
        fig, ax1 = plt.subplots(figsize=(16, 8))
        for i in result.index.drop(['benchmark', 'long_short']):
            ax1.bar(str(i), result.at[i, '年化收益'], label=str(result.index))
        ax1.axhline(0, color='black', linewidth=1)

    def plot_quantile_accumulated_net_value(self, commission=0, excess_return=False):
        '''
        绘制分组累计净值折线图
        '''
        nav_array = self.get_net_value_array(used_factor_freq=False, commission=commission,
                                             excess_return=excess_return)
        fig, ax1 = plt.subplots(figsize=(16, 8))
        for i in nav_array.columns.drop(['benchmark', 'long_short']):
            ax1.plot(nav_array.index, nav_array[i], label=str(i))
        fig.legend(loc=2, bbox_transform=ax1.transAxes, bbox_to_anchor=(0, 0, 1, 1))

    def plot_long_short_accumulated_net_value(self, commission=0, excess_return=True):
        '''
        绘制多头、空头、多空、基准净值折线图
        '''
        nav_array = self.get_net_value_array(used_factor_freq=False, commission=commission,
                                             excess_return=excess_return)
        fig, ax1 = plt.subplots(figsize=(16, 8))

        if excess_return:
            if self.direction == 1:
                ax1.plot(nav_array.index, nav_array[nav_array.columns[-3]], label='long_excess')
                ax1.plot(nav_array.index, nav_array[nav_array.columns[0]], label='short_excess')
            else:
                ax1.plot(nav_array.index, nav_array[nav_array.columns[0]], label='long_excess')
                ax1.plot(nav_array.index, nav_array[nav_array.columns[-3]], label='short_excess')
        else:
            if self.direction == 1:
                ax1.plot(nav_array.index, nav_array[nav_array.columns[-3]], label='long_group')
                ax1.plot(nav_array.index, nav_array[nav_array.columns[0]], label='short_group')
            else:
                ax1.plot(nav_array.index, nav_array[nav_array.columns[0]], label='long_group')
                ax1.plot(nav_array.index, nav_array[nav_array.columns[-3]], label='short_group')
        ax1.plot(nav_array.index, nav_array['long_short'], label='long_short_group')
        if not excess_return:
            ax1.plot(nav_array.index, nav_array['benchmark'], label='benchmark_group')
        fig.legend(loc=2, bbox_transform=ax1.transAxes, bbox_to_anchor=(0, 0, 1, 1))