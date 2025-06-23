import backtrader as bt
import pandas as pd
import numpy as np
import datetime
import sys
import os
from termcolor import colored

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")),
)
from z_using_files.backtrader_code.zz500_data import (
    get_csi500_stocks,
    get_trading_days,
    get_daily_data,
    get_monthly_first_last_trading_days,
)


class MonthlyStrategy(bt.Strategy):
    """
    月度动量策略
    1. 在每月最后一个交易日，计算过去一个月的收益率
    2. 筛选出收益率排名前20%的股票
    3. 根据收益率进行加权，作为新持仓的目标权重
    4. 在下一个交易日（即下月第一个交易日）以开盘价执行调仓
    """

    params = (
        ("top_n_percent", 0.20),  # 选择前20%
        ("rebalance_dates", None),  # 传入每月最后一个交易日列表
        ("lookback_period", 20),  # 代表约一个月的交易日
    )

    def __init__(self):
        super().__init__()
        # 将传入的 pandas Timestamps 转换为 python datetime.date 对象，便于比较
        self.rebalance_dates = [
            d.to_pydatetime().date() for d in self.p.rebalance_dates
        ]

        # 核心修改：引入状态标志位和目标持仓字典
        self.rebalance_due = False  # 是否有待执行的调仓任务
        self.target_weights = {}  # 储存计算出的目标权重

        # 使用 self.datetime 访问当前时间点
        self.datetime = self.datas[0].datetime
        # 存储 (日期, 总资产) 元组
        self.value_history = []

    def next(self):
        # 获取当前的回测日期
        current_date = self.datetime.date(0)

        # 记录当前日期和总资产
        self.value_history.append((current_date, self.broker.getvalue()))

        # --- 1. 交易执行模块 ---
        # 检查是否有待处理的调仓任务。
        # 这个模块总是在信号生成模块之前，确保在计算新信号前，完成上一期的交易。
        if self.rebalance_due:
            print(f"--- {current_date}: 执行调仓 ---")

            # 获取当前持仓的股票代码集合
            current_holdings = {
                d._name for d, pos in self.getpositions().items() if pos.size > 0
            }

            # 获取目标持仓的股票代码集合
            target_stocks = set(self.target_weights.keys())

            # 卖出：当前持有但不在目标中的股票，仓位调整为0
            stocks_to_sell = current_holdings - target_stocks
            for stock_code in stocks_to_sell:
                data = self.getdatabyname(stock_code)
                self.order_target_percent(data=data, target=0.0)
                print(f"  [清仓] {stock_code}")

            # 买入/调仓：目标持仓中的股票，调整至目标权重
            # 这也会自动处理从较低权重调整到较高权重的情况
            for stock_code, weight in self.target_weights.items():
                data = self.getdatabyname(stock_code)
                self.order_target_percent(data=data, target=weight)
                print(f"  [买入/调仓] {stock_code}, 目标权重: {weight:.4f}")

            # 任务完成，重置标志位和目标权重
            self.rebalance_due = False
            self.target_weights.clear()
            return  # 当天执行完交易后，不再做其他操作

        # --- 2. 信号生成模块 ---
        # 在每月最后一个交易日，计算过去一个月收益率，选出股票，并按收益率加权分配权重。
        if current_date in self.rebalance_dates:
            print(f"--- {current_date}: 计算下月持仓信号 ---")

            # 1. 计算所有股票的动量（上月回报）
            returns = {}
            for d in self.datas:
                # 确保数据长度足够进行计算
                if len(d) > self.p.lookback_period:
                    try:
                        # (今日收盘 / N日前收盘) - 1
                        momentum = (d.close[0] / d.close[-self.p.lookback_period]) - 1
                        returns[d._name] = momentum
                    except IndexError:
                        # 理论上 len(d) 检查后不会触发，但作为保险
                        continue

            if not returns:
                print("警告：没有足够的数据来计算回报。")
                return

            # 2. 降序排序并选择前 N% 的股票
            sorted_stocks = sorted(returns.items(), key=lambda x: x[1], reverse=True)
            top_n = int(len(sorted_stocks) * self.p.top_n_percent)

            # 筛选出收益率排名前列的股票，并且只保留正收益的股票进行加权
            top_performers = [item for item in sorted_stocks[:top_n] if item[1] > 0]

            if not top_performers:
                print(
                    f"警告：在 {current_date}，排名前 {self.p.top_n_percent:.0%} 的股票均为负收益，将执行空仓。"
                )
                self.target_weights.clear()  # 确保目标为空
                self.rebalance_due = True  # 设置标志，以便下一天卖出所有现有持仓
                return

            # 3. 计算收益率加权
            positive_return_sum = sum(r for d, r in top_performers)

            # 临时存储计算出的权重
            temp_weights = {}
            if positive_return_sum > 0:
                for d_name, r in top_performers:
                    temp_weights[d_name] = r / positive_return_sum

            # 4. 保存计算结果，并设置标志位以便下一天执行
            if temp_weights:
                self.target_weights = temp_weights
                self.rebalance_due = True  # 设置调仓标志
                print(
                    f"信号生成完毕，准备在下一个交易日调仓。选出 {len(self.target_weights)} 只股票。"
                )

    def stop(self):
        print("--- 回测结束 ---")
        print(f"最终资产价值: {self.broker.getvalue():,.2f}")


if __name__ == "__main__":
    """
    回测中证500成分股的月度动量策略

    uv run z_using_files/backtrader_code/strategy_01.py
    """
    print(colored(f"1. 数据获取开始...", "light_yellow"))
    data_start_date, data_end_date = "2023-01-01", "2025-06-01"
    backtest_start_date, backtest_end_date = "2023-02-01", "2025-06-01"

    adjust = "qfq"
    all_stock_daily = pd.DataFrame()

    stocks = get_csi500_stocks(date=data_start_date)
    if stocks is None or len(stocks) == 0:
        print("无法获取中证500成分股")
    else:
        print(f"获取到 {len(stocks)} 只中证500成分股")
        print(stocks[:2])

    for stock in stocks:
        stock_code = stock[0]
        df = get_daily_data(
            stock_code,
            data_start_date.replace("-", ""),
            data_end_date.replace("-", ""),
            adjust,
        )
        if df.empty:
            print(f"{stock_code} 没有获取到数据")
        else:
            all_stock_daily = pd.concat([all_stock_daily, df], ignore_index=True)
    print(f"all_stock_daily.shape:{all_stock_daily.shape}")
    print(f"{all_stock_daily.head(2)}")

    trading_days = get_trading_days(
        pd.to_datetime(data_start_date), pd.to_datetime(data_end_date)
    )
    print(f"获取到 {len(trading_days)} 个交易日")
    print(trading_days[:2])
    first_days_list, last_days_list = get_monthly_first_last_trading_days(trading_days)
    print("每月第一个交易日:")
    print(first_days_list[:2])
    print("每月最后一个交易日:")
    print(last_days_list[:2])

    print(colored(f"2. 设置Backtrader...", "light_yellow"))
    cerebro = bt.Cerebro()

    # 设置回测时间范围
    bt_start = datetime.datetime.strptime(backtest_start_date, "%Y-%m-%d")
    bt_end = datetime.datetime.strptime(backtest_end_date, "%Y-%m-%d")

    # 为每支股票添加数据
    print(colored(f"2.1 正在向 Backtrader 添加数据...", "light_yellow"))
    all_stock_daily["date"] = pd.to_datetime(all_stock_daily["date"])

    # 过滤掉数据不完整的股票
    min_date_required = pd.to_datetime(backtest_start_date)
    # 按股票代码分组，计算每只股票的最小日期
    start_dates = all_stock_daily.groupby("stock_code")["date"].min()
    # 找出那些开始日期晚于我们要求的最小日期的股票
    valid_stocks = start_dates[start_dates < min_date_required].index
    # 只使用这些有效的股票进行回测
    stock_codes_in_universe = valid_stocks

    print(
        f"原始股票数量: {len(all_stock_daily['stock_code'].unique())}, 筛选后剩余: {len(stock_codes_in_universe)}, 去除那些开始日期晚于我们要求的最小日期的股票"
    )

    # 获取回测区间内的股票代码
    # stock_codes_in_universe = all_stock_daily["stock_code"].unique()

    for stock_code in stock_codes_in_universe:
        df = all_stock_daily[all_stock_daily["stock_code"] == stock_code].copy()

        # 确保数据以日期排序并设置索引
        df.sort_values(by="date", inplace=True)
        df.set_index("date", inplace=True)

        if not df.empty:
            # print(
            #     f"正在添加股票: {stock_code}, 数据范围: {df.index.min().date()} to {df.index.max().date()}"
            # )
            pass
        else:
            print(colored(f"警告: 股票 {stock_code} 的数据为空!", "red"))

        # 添加 backtrader 不要求但最好有的 openinterest 列
        # peninterest 的中文意思是 未平仓合约量
        df["openinterest"] = 0

        # 创建数据源
        data = bt.feeds.PandasData(
            dataname=df,
            fromdate=bt_start,  # 从回测开始日期加载
            todate=bt_end,  # 到回测结束日期
        )
        cerebro.adddata(data, name=str(stock_code))

    print(colored(f"3. 设置策略和参数...", "light_yellow"))
    cerebro.addstrategy(MonthlyStrategy, rebalance_dates=last_days_list)

    # 设置 初始资金 佣金 滑点
    cerebro.broker.setcash(100_000_000.0)
    cerebro.broker.setcommission(commission=0.0003)
    cerebro.broker.set_slippage_perc(perc=0.0001)

    print(colored(f"4. 添加分析器...", "light_yellow"))
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio_A,
        _name="sharpe",
        timeframe=bt.TimeFrame.Days,
        compression=1,
        riskfreerate=0.02,
    )
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns", tann=252)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trade_analyzer")  # 交易次数
    cerebro.addanalyzer(bt.analyzers.Transactions, _name="transactions")  # 换手率相关

    print(colored(f"5. 运行回测...", "light_yellow"))
    results = cerebro.run()
    strat = results[0]

    final_value = cerebro.broker.getvalue()
    initial_value = cerebro.broker.startingcash

    print(colored(f"\n=============== 回测结果分析 ===============", "light_yellow"))
    print(f"回测区间: {backtest_start_date} to {backtest_end_date}")
    print(f"初始资产: {initial_value:,.2f}")
    print(f"最终资产: {final_value:,.2f}")

    # 总回报
    total_return = (final_value / initial_value) - 1
    print(f"总回报率: {total_return:.2%}")

    # 年化收益率
    annual_return = strat.analyzers.returns.get_analysis()["rnorm100"]
    print(f"年化收益率: {annual_return:.2f}%")

    # 最大回撤
    max_drawdown = strat.analyzers.drawdown.get_analysis()["max"]["drawdown"]
    print(f"最大回撤: {max_drawdown:.2f}%")

    # 夏普比率
    sharpe_ratio = strat.analyzers.sharpe.get_analysis()["sharperatio"]
    print(f"年化夏普比率 (无风险利率 2%): {sharpe_ratio:.4f}")

    # 交易次数
    trade_info = strat.analyzers.trade_analyzer.get_analysis()
    total_trades = trade_info.total.closed + trade_info.total.open
    print(f"总交易次数: {total_trades}")

    # 换手率 (估算)
    # 计算总交易额
    transactions = strat.analyzers.transactions.get_analysis()
    total_turnover = sum(
        abs(value[0][0] * value[0][1]) for date, value in transactions.items()
    )
    # 平均资产
    avg_portfolio_value = (
        np.mean(
            [v[0] for v in strat.analyzers.returns.get_analysis()["dret"]]
            * initial_value
        )
        if "dret" in strat.analyzers.returns.get_analysis()
        else initial_value
    )
    # 简单估算换手率，更精确的计算需要逐日资产和交易额
    # 这里的换手率 = 总交易额 / (期末资产 * 交易周期数)
    num_months = len(last_days_list)  # 近似交易周期
    turnover_rate = total_turnover / (final_value * num_months) if num_months > 0 else 0
    print(f"总交易额: {total_turnover:,.2f}")
    print(f"估算月均换手率: {turnover_rate:.2%}")

    import matplotlib
    import matplotlib.pyplot as plt

    # 提取资产历史数据
    value_history = strat.value_history
    # print(f"{value_history}")
    if not value_history:
        print("没有记录资产历史数据，请检查策略代码。")
    else:
        # 转换为 DataFrame
        df_values = pd.DataFrame(value_history, columns=["date", "portfolio_value"])
        df_values["date"] = pd.to_datetime(df_values["date"])
        df_values.set_index("date", inplace=True)

        # 绘制净值曲线
        try:
            print("开始绘制净值曲线...")
            matplotlib.use("Agg")  # 非交互式后端
            plt.figure(figsize=(10, 6))
            plt.plot(
                df_values.index,
                df_values["portfolio_value"],
                label="Portfolio Value",
                color="blue",
            )
            plt.title("Portfolio Value Over Time")
            plt.xlabel("Date")
            plt.ylabel("Portfolio Value (CNY)")
            plt.grid(True)
            plt.legend()
            os.makedirs("output", exist_ok=True)
            plot_path = os.path.join("output", "portfolio_value_plot.png")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"净值曲线已保存到 {plot_path}")
        except Exception as e:
            print(f"绘制净值曲线失败: {str(e)}")
