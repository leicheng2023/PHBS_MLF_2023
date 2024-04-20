#!/usr/bin/env python
# -*- coding: UTF-8 -*-


class FactorCapacity:
    def __init__(self):
        self.long_excess_annualized_retrun = None
        self.long_excess_drawdown = None
        self.long_information_ratio = None
        self.win_rate_longshort = None
        self.win_rate_long = None

        # 因子覆盖率分析
        self.factor_coverage_array = None

        # 因子IC分析
        self.ic_array = None
        self.quantile_ic_array = None
        self.grouped_ic_array = None

        self.ic_summary = None
        self.quantile_ic_summary = None

        # 因子自相关性和换手率分析
        self.autocorrelation_array = None
        self.turnover_array = None

        # 因子收益分析
        self.return_summary = None