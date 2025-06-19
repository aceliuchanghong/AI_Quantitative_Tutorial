import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import backtrader as bt
import matplotlib.pyplot as plt

# 设置 matplotlib 支持中文
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# 获取交易日历
def get_trading_days(start_date, end_date):
    """获取交易日历"""
    calendar = ak.stock_zh_index_daily(symbol="sh000001")  # 使用上证指数获取交易日历
    calendar["date"] = pd.to_datetime(calendar["date"])
    calendar = calendar[
        (calendar["date"] >= start_date) & (calendar["date"] <= end_date)
    ]
    return calendar["date"].tolist()


# 获取中证 500 成分股列表
def get_csi500_stocks(date):
    """获取指定日期的中证 500 成分股列表"""
    try:
        # 获取中证 500 成分股
        csi500 = ak.index_stock_cons_csindex(symbol="000905")  # 中证 500 代码
        csi500["date"] = pd.to_datetime(csi500["date"])
        csi500 = csi500[csi500["date"] <= pd.to_datetime(date)]
        if not csi500.empty:
            latest_date = csi500["date"].max()
            return csi500[csi500["date"] == latest_date][
                ["stock_code", "stock_name"]
            ].values.tolist()
        return []
    except Exception as e:
        print(f"获取中证 500 成分股失败: {e}")
        return []


# 获取股票历史数据
def get_stock_data(symbol, start_date, end_date):
    """获取个股历史数据"""
    try:
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date.strftime("%Y%m%d"),
            end_date=end_date.strftime("%Y%m%d"),
            adjust="qfq",
        )
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df[["date", "open", "close", "high", "low", "volume"]]
            df.columns = ["date", "open", "close", "high", "low", "volume"]
            df.set_index("date", inplace=True)
            return df
        return None
    except Exception as e:
        print(f"获取股票 {symbol} 数据失败: {e}")
        return None


# 获取流通市值
def get_market_cap(symbol, date):
    """获取指定日期的流通市值"""
    try:
        df = ak.stock_zh_a_spot_em()
        df = df[df["代码"] == symbol]
        if not df.empty:
            return df["流通市值"].iloc[0]
        return None
    except Exception as e:
        print(f"获取股票 {symbol} 流通市值失败: {e}")
        return None


# 选股逻辑：基于前一个月收益率选出前 20% 股票
def select_top_stocks(stocks, start_date, end_date):
    """选择前 20% 表现最好的股票"""
    returns = []
    for symbol, _ in stocks:
        df = get_stock_data(symbol, start_date, end_date)
        if df is not None and len(df) > 1:
            ret = df["close"].iloc[-1] / df["close"].iloc[0] - 1  # 计算月度收益率
            returns.append((symbol, ret))

    # 按收益率降序排序，选择前 20%
    returns.sort(key=lambda x: x[1], reverse=True)
    top_count = max(1, int(len(returns) * 0.2))  # 至少选1只
    return [x[0] for x in returns[:top_count]]


# 自定义 Backtrader 策略
class CSI500Strategy(bt.Strategy):
    params = (
        ("initial_cash", 100_000_000),  # 初始资金 1 亿
        ("commission", 0.0003),  # 双边佣金 0.03%
        ("slippage", 0.0001),  # 双边滑点 0.01%
    )

    def __init__(self):
        self.current_month = None
        self.holdings = {}  # 当前持仓
        self.trading_days = get_trading_days(
            pd.to_datetime("2023-12-01"),  # 提前一个月以确保数据完整
            pd.to_datetime("2025-06-01"),
        )
        self.monthly_rebalance_dates = self.get_monthly_rebalance_dates()

    def get_monthly_rebalance_dates(self):
        """获取每个月第一个交易日"""
        rebalance_dates = []
        current_date = pd.to_datetime("2024-01-01")
        end_date = pd.to_datetime("2025-06-01")
        while current_date <= end_date:
            month_start = current_date.replace(day=1)
            month_trading_days = [
                d
                for d in self.trading_days
                if d.month == month_start.month and d.year == month_start.year
            ]
            if month_trading_days:
                rebalance_dates.append(month_trading_days[0])
            current_date = current_date + pd.offsets.MonthBegin(1)
        return rebalance_dates

    def next(self):
        current_date = self.datas[0].datetime.date(0)
        if current_date not in self.monthly_rebalance_dates:
            return

        # 检查是否需要调仓
        current_month = current_date.strftime("%Y-%m")
        if current_month == self.current_month:
            return
        self.current_month = current_month

        # 获取上个月的开始和结束日期
        last_month_end = current_date - pd.offsets.MonthBegin(1)
        last_month_start = last_month_end.replace(day=1)

        # 获取中证 500 成分股
        stocks = get_csi500_stocks(last_month_end)
        if not stocks:
            print(f"{current_date}: 无法获取成分股")
            return

        # 选股：基于上个月收益率选择前 20%
        selected_stocks = select_top_stocks(stocks, last_month_start, last_month_end)
        if not selected_stocks:
            print(f"{current_date}: 无选股结果")
            return

        # 计算流通市值权重
        total_market_cap = 0
        market_caps = {}
        for symbol in selected_stocks:
            mc = get_market_cap(symbol, last_month_end)
            if mc:
                market_caps[symbol] = mc
                total_market_cap += mc

        # 清仓现有持仓
        for symbol, position in list(self.holdings.items()):
            if position.size > 0:
                self.order_target_percent(data=self.getdatabyname(symbol), target=0)
                print(f"{current_date}: 卖出 {symbol}, 持仓清空")

        # 分配新持仓
        total_weight = 0
        for symbol in selected_stocks:
            if symbol in market_caps:
                weight = market_caps[symbol] / total_market_cap
                total_weight += weight
                self.order_target_percent(
                    data=self.getdatabyname(symbol), target=weight
                )
                print(f"{current_date}: 买入 {symbol}, 权重 {weight:.4f}")
        self.holdings = {
            symbol: self.getposition(self.getdatabyname(symbol))
            for symbol in selected_stocks
        }

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                print(
                    f"{order.data._name} 买入: 价格 {order.executed.price:.2f}, 数量 {order.executed.size}"
                )
            elif order.issell():
                print(
                    f"{order.data._name} 卖出: 价格 {order.executed.price:.2f}, 数量 {order.executed.size}"
                )


# 主回测函数
def run_backtest():
    cerebro = bt.Cerebro()
    cerebro.addstrategy(CSI500Strategy)

    # 获取交易日历以确定数据范围
    trading_days = get_trading_days(
        pd.to_datetime("2023-12-01"), pd.to_datetime("2025-06-01")
    )
    if not trading_days:
        print("无法获取交易日历")
        return

    # 加载中证 500 成分股数据
    stocks = get_csi500_stocks("2024-01-01")
    if not stocks:
        print("无法获取中证 500 成分股")
        return

    # 为每只股票添加数据
    for symbol, name in stocks:
        df = get_stock_data(
            symbol, pd.to_datetime("2023-12-01"), pd.to_datetime("2025-06-01")
        )
        if df is not None:
            data = bt.feeds.PandasData(dataname=df, name=symbol)
            cerebro.adddata(data)

    # 设置初始资金、佣金和滑点
    cerebro.broker.setcash(100_000_000)
    cerebro.broker.setcommission(commission=0.0003)
    cerebro.broker.set_slippage(0.0001)

    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

    print("开始回测...")
    results = cerebro.run()
    strategy = results[0]

    # 输出结果
    print(f"最终资产: {cerebro.broker.getvalue():.2f}")
    print(f"夏普比率: {strategy.analyzers.sharpe.get_analysis()['sharperatio']:.2f}")
    print(
        f"最大回撤: {strategy.analyzers.drawdown.get_analysis()['max']['drawdown']:.2f}%"
    )
    print(f"总回报: {strategy.analyzers.returns.get_analysis()['rtot']*100:.2f}%")

    # 可视化
    cerebro.plot()


if __name__ == "__main__":
    run_backtest()
