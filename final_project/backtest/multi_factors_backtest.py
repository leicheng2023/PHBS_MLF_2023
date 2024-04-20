#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import pandas as pd

from single_factor_backtest import *
from factor_capacity import *
from typing import Union, Optional, Dict


class MultiFactor_MultiPool_BackTest:
    def __init__(
            self,
            factor_data: pd.DataFrame,  # 多重索引，字段为每个因子，字段名为因子名
            price_data: pd.DataFrame,  # 多重索引，字段为价格
            benchmark_data: dict,  # 同上
            pool_data: dict,  # 字典，key为池子名，value为dataframe，多重索引，同单池子格式
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            is_daily_factor: Union[bool, Dict[str, bool]] = True,  # 布尔值，或字典，key为因子名，value是布尔值
            group_data: Optional[pd.DataFrame] = None,  # 同上
            direction: Union[int, Dict[str, int]] = 1,  # int 或字典，key为因子名，value是int,
            # start_date: None | str = None,
            # end_date: None | str = None,
            # is_daily_factor: bool = True,  # 布尔值，或字典，key为因子名，value是布尔值
            # group_data: None | pd.DataFrame = None,  # 同上
            # direction: int | dict = 1,  # int 或字典，key为因子名，value是int,
            quantiles: int = 5,
    ):
        """
        初始化方法。

        Parameters
        ----------
        factor_data : pd.DataFrame - MultiIndex (date, asset)
            因子数据，每一个因子为一个字段，字段名是因子名。
        price_data : pd.DataFrame - MultiIndex (date, asset)
            价格数据，只有唯一一个price字段。
        benchmark_data : pd.DataFrame - MultiIndex (date, asset)
            基准数据，只有唯一一个benchmark_price字段。
        pool_data : Dict[str, pd.DataFrame - MultiIndex (date, asset)]
            选股池数据，每一个选股池是一个DataFrame且无字段。
        start_date : str, optional
            开始日期，格式如：'20130101'，默认为None。
        end_date : str, optional
            结束日期，格式如：'20221231'。默认为None。
        is_daily_factor : bool or Dict[str, bool], optional
            该因子是否为日频因子，默认为True。若为bool，则所有因子共享该标记；若为字典，则各因子可分别设置。
        group_data : Dict[str, pd.DataFrame - MultiIndex (date, asset)], optional
            分组（行业）数据，默认为None；DataFrame的字段是所属行业。
        direction : int or Dict[str, int], optional
            因子方向，默认为1。若为int，则所有因子共享该方向；若为字典，则各因子可分别设置。
        quantiles : int, optional
            回测分组数，默认为5。

        Returns
        -------
        None

        """
        self.factor_data = factor_data
        self.price_data = price_data
        self.benchmark_data = benchmark_data
        self.pool_data = pool_data

        self.start_date = pd.Timestamp(start_date) if start_date is not None else None
        self.end_date = pd.Timestamp(end_date) if end_date is not None else None

        if isinstance(is_daily_factor, bool):
            self.is_daily_factor = {factor_name: is_daily_factor for factor_name in self.factor_data.columns}
        elif isinstance(is_daily_factor, dict):
            self.is_daily_factor = is_daily_factor

        self.group_data = group_data

        if isinstance(direction, int):
            self.direction = {factor_name: direction for factor_name in self.factor_data.columns}
        elif isinstance(direction, dict):
            self.direction = direction

        self.quantiles = quantiles

        self.factor_nums = self.factor_data.shape[1]
        self.pool_nums = len(self.pool_data)

        self.single_backtest_dataframe = None

    def get_factor_list(self):
        return list(self.factor_data.columns)

    def get_pool_list(self):
        return list(self.pool_data.keys())

    def generate_single_factor_pool_object(self):
        """
        使用传入实例的a个因子和b个股票池，生成a×b个单因子实例
        并完成所有实例的数据清洗
        """
        self.single_backtest_dataframe = pd.DataFrame(
            np.full(shape=(self.factor_nums, self.pool_nums), fill_value=None),
            index=self.get_factor_list(),
            columns=self.get_pool_list(),
        )
        for factor_name in self.get_factor_list():
            for pool_name in self.get_pool_list():
                sub_object = SingleFactor_SinglePool_BackTest(
                    factor_data=self.factor_data[[factor_name]],
                    price_data=self.price_data,
                    benchmark_data=self.benchmark_data[pool_name],
                    pool_data=self.pool_data[pool_name],
                    factor_name=factor_name,
                    pool_name=pool_name,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    is_daily_factor=self.is_daily_factor[factor_name],
                    group_data=self.group_data,
                    direction=self.direction[factor_name]
                )
                sub_object.generate_clean_data(quantiles=self.quantiles)
                self.single_backtest_dataframe.at[factor_name, pool_name] = sub_object

    def get_backtest(self, factor_name: str, pool_name: str) -> SingleFactor_SinglePool_BackTest:
        """
        获取指定因子和股票池的backtest对象

        Parameters
        ----------
        factor_name : str
            因子名
        pool_name : str
            选股池名

        Returns
        -------
        Backtest
            Backtest对象
        """
        return self.single_backtest_dataframe.loc[factor_name, pool_name]

    # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ 集成的单因子分析功能 ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓

    # ==== 因子覆盖率分析 ====
    def plot_factor_coverage(self, factor_name: str, pool_name: str):
        """
        绘制因子覆盖率曲线

        Parameters
        ----
        factor_name: str
            因子名
        pool_name: str
            选股池名

        Returns
        ------
        ax：matplotlib.axes.Axes
            绘制因子覆盖率曲线的坐标轴对象
        """
        return self.get_backtest(factor_name=factor_name, pool_name=pool_name).plot_factor_coverage()

    # ==== 因子描述性统计 ====
    def analyse_factor_descriptive_statistics(self, factor_name: str, pool_name: str, by: str = 'quantile'):
        """
        分析因子的描述性统计信息

        Parameters
        ----
        factor_name: str
            因子名称
        pool_name: str
            池子名称
        by: str, optional
            用于分组的变量（默认为'quantile'）
            quantile：按因子分组分类统计
            year：按年份分类统计

        Returns
        ------
        analysis: pd.DataFrame
            因子描述性统计信息分析结果
        """
        return self.get_backtest(factor_name=factor_name, pool_name=pool_name).analyse_factor_descriptive_statistics(
            by=by)

    def plot_factor_distribution(self, factor_name: str, pool_name: str):
        """
        根据给定的因子名(factor_name)和池名(pool_name)，绘制因子的分布图。

        Parameters
        ----------
        factor_name : str
            The name of the factor.
        pool_name : str
            The name of the data pool.

        Returns
        -------
        None
        """
        return self.get_backtest(factor_name=factor_name, pool_name=pool_name).plot_factor_distribution()

    # ========================================= 因子IC分析 =========================================
    def analyse_ic(self, factor_name: str, pool_name: str, ic_type: str = 'ic'):
        """
        Output factor IC analysis result.

        Parameters
        ----------
        factor_name : str
            因子名称
        pool_name : str
            选股池名称。
        ic_type : {'ic', 'quantile_ic', 'grouped_ic'}, default 'ic'
            要分析的IC类型。
            * ic：普通IC
            * quantile_ic：分位数IC
            * grouped_ic：分行业IC

        Returns
        -------
        dict
            A dictionary containing the IC analysis result.
        """
        bt = self.get_backtest(factor_name=factor_name, pool_name=pool_name)
        if ic_type == 'ic':
            return bt.analyse_ic()
        elif ic_type == 'quantile_ic':
            return bt.analyse_quantile_ic()
        elif ic_type == 'grouped_ic':
            return bt.analyse_grouped_ic()
        else:
            raise ValueError("Invalid IC type. Possible values are 'ic', 'quantile_ic' and 'grouped_ic'.")

    def plot_ic(self, factor_name: str, pool_name: str, ic_type: str = 'ic', bar_figure: bool = False):
        """
        Plot the IC analysis result.

        Parameters
        ----------
        factor_name : str
            因子名称
        pool_name : str
            选股池名称。
        ic_type : {'ic', 'quantile_ic', 'grouped_ic'}, default 'ic'
            要分析的IC类型。
            * ic：普通IC
            * quantile_ic：分位数IC
            * grouped_ic：分行业IC
        bar_figure : bool, default False
            Whether to plot the IC analysis result. Default to False.

        Returns
        -------
        None
        """
        bt = self.get_backtest(factor_name=factor_name, pool_name=pool_name)
        if ic_type == 'ic':
            return bt.plot_ic(bar_figure=bar_figure)
        elif ic_type == 'quantile_ic':
            return bt.plot_quantile_ic(bar_figure=bar_figure)
        elif ic_type == 'grouped_ic':
            return bt.plot_grouped_ic()
        else:
            raise ValueError("Invalid IC type. Possible values are 'ic', 'quantile_ic' and 'grouped_ic'.")

    def analyse_ic_decay(self, factor_name: str, pool_name: str, max_lag: int = 10):
        """

        Parameters
        ----------
        factor_name : str
            因子名称。
        pool_name : str
            选股池名称。
        max_lag : int
            分析IC衰减的最大期数

        Returns
        -------

        """
        return self.get_backtest(factor_name=factor_name, pool_name=pool_name).analyse_ic_decay(max_lag=max_lag)

    def plot_ic_dacay(self, factor_name: str, pool_name: str, max_lag: int = 10):
        """

        Parameters
        ----------
        factor_name : str
            因子名称。
        pool_name : str
            选股池名称。
        max_lag : int
            分析IC衰减的最大期数

        Returns
        -------

        """
        return self.get_backtest(factor_name=factor_name, pool_name=pool_name).plot_ic_dacay(max_lag=max_lag)

    # ========================================= 因子自相关性和换手率分析 =========================================
    def analyse_factor_autocorrelation(self, factor_name: str, pool_name: str, max_lag: int = 5):
        """

        Parameters
        ----------
        factor_name : str
            因子名称。
        pool_name : str
            选股池名称。
        max_lag : int
            最大回看期数

        Returns
        -------

        """
        bt = self.get_backtest(factor_name=factor_name, pool_name=pool_name)
        return bt.analyse_factor_autocorrelation(max_lag=max_lag)

    def analyse_factor_turnover(self, factor_name: str, pool_name: str, used_factor_freq: bool = True):
        """

        Parameters
        ----------

        factor_name : str
            因子名称。
        pool_name : str
            选股池名称。
        used_factor_freq : bool, default
            若为True，则计算单次调仓的平均换手率；若为False，则计算单个交易日的平均换手率

        Returns
        -------

        """
        bt = self.get_backtest(factor_name=factor_name, pool_name=pool_name)
        return bt.analyse_factor_turnover(used_factor_freq=used_factor_freq)

    # ========================================= 因子多头和空头组合的行业分布情况分析 =========================================
    # （必须在创建backtest实例时传入group_data参数才能使用）
    def _check_group_data(self):
        """
        检查是否传入行业数据，若未传入则不可以调用行业分析方法
        """
        if self.group_data is None:
            raise ValueError("Please provide group_data when creating backtest instance.")

    def analyse_factor_group_distribution(self, factor_name: str, pool_name: str, long: bool = True):
        """
        分析因子的多头和空头组合中不同行业的占比情况

        Parameters
        ----------
        factor_name : str
            因子名称。
        pool_name : str
            选股池名称。
        long : bool, optional, default True
            若为True，则分析多头组合；若为False，则分析空头组合。

        Returns
        -------

        """
        self._check_group_data()
        bt = self.get_backtest(factor_name=factor_name, pool_name=pool_name)
        return bt.analyse_factor_group_distribution(long=long)

    def analyse_factor_group_distribution_topN_per_year(
            self,
            factor_name: str,
            pool_name: str,
            long: bool = True,
            display_num: int = 5
    ):
        """
        分析因子的多头和空头组合中占比最高的n个行业

        Parameters
        ----------
        factor_name : str
            因子名称。
        pool_name : str
            选股池名称。
        long : bool, optional, default True
            若为True，则分析多头组合；若为False，则分析空头组合。
        display_num : int, optional, default 5
            显示占比最高的n个行业

        Returns
        -------

        """
        self._check_group_data()
        bt = self.get_backtest(factor_name=factor_name, pool_name=pool_name)
        return bt.analyse_factor_group_distribution_topN_per_year(long=long, display_num=display_num)

    def plot_factor_group_distribution(self, factor_name: str, pool_name: str, long: bool = True):
        """
        绘制因子的多头和空头组合的行业分布图

        Parameters
        ----------
        factor_name : str
            因子名称。
        pool_name : str
            选股池名称。
        long : bool, optional, default True
            若为True，则分析多头组合；若为False，则分析空头组合。

        Returns
        -------

        """
        bt = self.get_backtest(factor_name=factor_name, pool_name=pool_name)
        return bt.plot_factor_group_distribution(long=long)

    # ========================================= 因子分组收益分析 =========================================

    def analyse_return_array(
            self,
            factor_name: str,
            pool_name: str,
            commission: float = 0
    ):
        """
        输出因子分组收益情况的详细分析

        Parameters
        ----------
        factor_name : str
            因子名称。
        pool_name : str
            选股池名称。
        commission : float, optional, default 0
            交易佣金（单边），万分之一为1/10000

        Returns
        -------

        """
        bt = self.get_backtest(factor_name=factor_name, pool_name=pool_name)
        return bt.analyse_return_array(commission=commission)

    def analyse_return_briefly(
            self,
            factor_name: str,
            pool_name: str,
            commission: float = 0
    ):
        """
        输出因子分组收益情况的简要版分析

        Parameters
        ----------
        factor_name : str
            因子名称。
        pool_name : str
            选股池名称。
        commission : float, optional, default 0
            交易佣金（单边），万分之一为1/10000

        Returns
        -------

        """
        bt = self.get_backtest(factor_name=factor_name, pool_name=pool_name)
        return bt.analyse_return_briefly(commission=commission)

    def plot_annual_return_heatmap(
            self,
            factor_name: str,
            pool_name: str,
            commission: float = 0,
            excess_return: bool = False
    ):
        """
        绘制因子不同分组不同年份的收益热力图

        Parameters
        ----------
        factor_name : str
            因子名称。
        pool_name : str
            选股池名称。
        commission : float, optional, default 0
            交易佣金（单边），万分之一为1/10000
        excess_return : bool, optional, default False
            若为True，则使用超额收益；若为False，则使用绝对收益

        Returns
        -------

        """
        bt = self.get_backtest(factor_name=factor_name, pool_name=pool_name)
        return bt.plot_annual_return_heatmap(commission=commission, excess_return=excess_return)

    def plot_quantile_annualized_return(
            self,
            factor_name: str,
            pool_name: str,
            commission: float = 0,
            excess_return: bool = True
    ):
        """
        绘制不同分组的年化收益柱形图

        Parameters
        ----------
        factor_name : str
            因子名称。
        pool_name : str
            选股池名称。
        commission : float, optional, default 0
            交易佣金（单边），万分之一为1/10000
        excess_return : bool, optional, default True
            若为True，则使用超额收益；若为False，则使用绝对收益

        Returns
        -------

        """
        bt = self.get_backtest(factor_name=factor_name, pool_name=pool_name)
        return bt.plot_quantile_annualized_return(commission=commission, excess_return=excess_return)

    def plot_accumulated_net_value(
            self,
            factor_name: str,
            pool_name: str,
            plot_type: str = 'quantile',
            commission: float = 0,
            excess_return: bool = False
    ):
        """
        绘制净值曲线。

        Parameters
        ----------
        factor_name : str
            因子名称。
        pool_name : str
            选股池名称。
        commission : float, optional, default 0
            交易佣金（单边），万分之一为1/10000
        excess_return : bool, optional, default False
            若为True，则使用超额收益；若为False，则使用绝对收益
        plot_type: str, {'quantile', 'long_short'}, default 'quantile'
            设置绘图类型。
            * quantile：绘制分组净值曲线
            * long_short：绘制多头、空头、多头、基准净值曲线

        Returns
        -------
        plot
            Accumulated Net Value over time for different factor quantiles or long/short positions.
        """
        bt = self.get_backtest(factor_name=factor_name, pool_name=pool_name)
        if plot_type == 'quantile':
            return bt.plot_quantile_accumulated_net_value(commission=commission, excess_return=excess_return)
        elif plot_type == 'long_short':
            return bt.plot_long_short_accumulated_net_value(commission=commission, excess_return=excess_return)
        else:
            raise ValueError("Invalid type. Type can only be 'quantile' or 'long_short'.")

    # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ 集成的单因子分析功能 ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

    # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ 多因子对比分析功能 ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
    def get_backtest_list(
            self,
            compare_level: str,
            target: str,
            filter_list: Optional[list] = None
    ):
        """

        Parameters
        ----------
        compare_level : str, {'pool', 'factor'}
            在哪个维度上进行比较。
            * pool：对“特定因子”在“多个选股池”的维度上进行比较
            * factor：对“多个因子”在“特定选股池”的维度上进行比较
        target : str
            特定因子名（compare_level='pool'）或特定选股池名（compare_level='factor'）
        filter_list : None or list, optional, default=None
            若不传入，则对比全部选股池（compare_level='pool'）或对比全部因子（compare_level='factor'）
            若传入选股池名或因子名列表，则仅对比传入的范围。

        Returns
        -------
        filter_list : list
            选股池名称列表（compare_level='pool'）或因子名称列表（compare_level='factor'）
        backtest_list : list
            filter_list对应的backtest对象列表
        """
        if compare_level == 'pool':
            if target not in self.get_factor_list():
                raise ValueError("不存在该因子")
            if filter_list is None:
                filter_list = self.get_pool_list()
            return filter_list, list(self.single_backtest_dataframe.loc[target, filter_list])
        elif compare_level == 'factor':
            if target not in self.get_pool_list():
                raise ValueError("不存在该选股池")
            if filter_list is None:
                filter_list = self.get_factor_list()
            return filter_list, list(self.single_backtest_dataframe.loc[filter_list, target])

    def compare_plot_factor_coverage(
            self,
            compare_level: str,
            target: str,
            filter_list: Optional[list] = None
    ):
        """
        对比特定因子在不同选股池中的因子覆盖率，或不同因子在特定选股池中的因子覆盖率

        Parameters
        ----------
        compare_level : str, {'pool', 'factor'}
            在哪个维度上进行比较。
            * pool：对“特定因子”在“多个选股池”的维度上进行比较
            * factor：对“多个因子”在“特定选股池”的维度上进行比较
        target : str
            特定因子名（compare_level='pool'）或特定选股池名（compare_level='factor'）
        filter_list : None or list, optional, default=None
            若不传入，则对比全部选股池（compare_level='pool'）或对比全部因子（compare_level='factor'）
            若传入选股池名或因子名列表，则仅对比传入的范围。

        Returns
        -------

        """
        fig, ax1 = plt.subplots(figsize=(16, 8))
        backtest_name_list, backtest_list = self.get_backtest_list(compare_level=compare_level, target=target,
                                                                   filter_list=filter_list)
        for bt_name, bt in zip(backtest_name_list, backtest_list):
            sub = bt.get_factor_coverage().rename(columns={'因子覆盖率': bt_name})
            ax1.plot(sub.index, sub[bt_name], label=bt_name)
        ax1.legend()
        ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax1.set_ylim(0, 1.1)
        plt.show()

    def compare_analyse_ic(
            self,
            compare_level: str,
            target: str,
            filter_list: Optional[list] = None,
            ic_type: str = 'ic'
    ):
        """

        Parameters
        ----------
        compare_level : str, {'pool', 'factor'}
            在哪个维度上进行比较。
            * pool：对“特定因子”在“多个选股池”的维度上进行比较
            * factor：对“多个因子”在“特定选股池”的维度上进行比较
        target : str
            特定因子名（compare_level='pool'）或特定选股池名（compare_level='factor'）
        filter_list : None or list, optional, default=None
            若不传入，则对比全部选股池（compare_level='pool'）或对比全部因子（compare_level='factor'）
            若传入选股池名或因子名列表，则仅对比传入的范围。
        ic_type : str, {'ic', 'q_ic}, default 'ic'
            * ic：进行IC分析
            * q_ic：进行quantile_IC分析

        Returns
        -------

        """
        backtest_name_list, backtest_list = self.get_backtest_list(compare_level=compare_level, target=target,
                                                                   filter_list=filter_list)
        s_list = []
        for bt in backtest_list:
            if ic_type == 'ic':
                s_list.append(bt.analyse_ic())
            elif ic_type == 'q_ic':
                s_list.append(bt.analyse_quantile_ic())
        result = pd.concat(s_list)
        result.index = backtest_name_list
        return result

    def compare_analyse_factor_turnover(
            self,
            compare_level: str,
            target: str,
            filter_list: Optional[list] = None,
    ):
        """
        对比特定因子在不同选股池中的换手率，或不同因子在特定选股池中的换手率

        Parameters
        ----------
        compare_level : str, {'pool', 'factor'}
            在哪个维度上进行比较。
            * pool：对“特定因子”在“多个选股池”的维度上进行比较
            * factor：对“多个因子”在“特定选股池”的维度上进行比较
        target : str
            特定因子名（compare_level='pool'）或特定选股池名（compare_level='factor'）
        filter_list : None or list, optional, default=None
            若不传入，则对比全部选股池（compare_level='pool'）或对比全部因子（compare_level='factor'）
            若传入选股池名或因子名列表，则仅对比传入的范围。


        """
        backtest_name_list, backtest_list = self.get_backtest_list(compare_level=compare_level, target=target,
                                                                   filter_list=filter_list)
        s_list = []
        for bt in backtest_list:
            s_list.append(bt.analyse_factor_turnover()[['均值']])
        result = pd.concat(s_list, axis=1)
        result.columns = backtest_name_list
        return result

    def compare_analyse_return_briefly(
            self,
            compare_level: str,
            target: str,
            filter_list: Optional[list] = None,
            commission: float = 0
    ):
        """
        对比特定因子在不同选股池中的换手率，或不同因子在特定选股池中的换手率

        Parameters
        ----------
        compare_level : str, {'pool', 'factor'}
            在哪个维度上进行比较。
            * pool：对“特定因子”在“多个选股池”的维度上进行比较
            * factor：对“多个因子”在“特定选股池”的维度上进行比较
        target : str
            特定因子名（compare_level='pool'）或特定选股池名（compare_level='factor'）
        filter_list : None or list, optional, default=None
            若不传入，则对比全部选股池（compare_level='pool'）或对比全部因子（compare_level='factor'）
            若传入选股池名或因子名列表，则仅对比传入的范围。
        commission : float, optional, default 0
            交易佣金（单边），万分之一为1/10000

        """
        backtest_name_list, backtest_list = self.get_backtest_list(compare_level=compare_level, target=target,
                                                                   filter_list=filter_list)
        s_list = []
        for bt in backtest_list:
            s_list.append(bt.analyse_return_briefly(commission=commission))
        result = pd.concat(s_list)
        result.index = backtest_name_list
        return result

    def compare_plot_accumulated_net_value(
            self,
            compare_level: str,
            target: str,
            filter_list: Optional[list] = None,
            commission: float = 0,
            nv_type: str = 'l',
            excess_return: bool = True
    ):
        """
        绘制特定因子在不同选股池中的净值走势图，或不同因子在特定选股池中的净值走势图

        Parameters
        ----------

        compare_level : str, {'pool', 'factor'}
            在哪个维度上进行比较。
            * pool：对“特定因子”在“多个选股池”的维度上进行比较
            * factor：对“多个因子”在“特定选股池”的维度上进行比较
        target : str
            特定因子名（compare_level='pool'）或特定选股池名（compare_level='factor'）
        filter_list : None or list, optional, default=None
            若不传入，则对比全部选股池（compare_level='pool'）或对比全部因子（compare_level='factor'）
            若传入选股池名或因子名列表，则仅对比传入的范围。
        commission : float, optional, default 0
            交易佣金（单边），万分之一为1/10000
        nv_type : str, {'l', 'ls'}, default 'l'
            * l：绘制多头组合净值
            * ls：绘制多空组合净值
        excess_return : bool, default False
            若为True，则使用超额收益；若为False，则使用绝对收益

        """
        if nv_type == 'l' and excess_return is True:
            suffix = '多头超额'
        elif nv_type == 'l' and excess_return is False:
            suffix = '多头收益'
        elif nv_type == 'ls':
            suffix = '多空收益'
        else:
            raise ValueError("传入值有误")
        fig, ax1 = plt.subplots(figsize=(16, 8))
        backtest_name_list, backtest_list = self.get_backtest_list(compare_level=compare_level, target=target,
                                                                   filter_list=filter_list)
        for bt, bt_name in zip(backtest_list, backtest_name_list):
            nav_array = bt.get_single_net_value_array(nv_type=nv_type, commission=commission,
                                                      excess_return=excess_return, used_factor_freq=False)
            label = f'{bt_name}_{suffix}'
            ax1.plot(nav_array.index, nav_array.values, label=label)
        fig.legend(loc=2, bbox_transform=ax1.transAxes, bbox_to_anchor=(0, 0, 1, 1))
        plt.show()
