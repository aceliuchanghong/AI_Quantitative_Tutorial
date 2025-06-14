import backtrader as bt
import pandas as pd
import akshare as ak
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)


class MAStrategy(bt.Strategy):
    """
    定义移动平均线交叉策略

    快均线 10 天 上穿慢均线 20 天 时买入。
    快均线下穿慢均线时卖出。
    """

    params = (
        ("fast_ma", 10),  # 快均线周期
        ("slow_ma", 20),  # 慢均线周期
    )

    def __init__(self):
        # 定义快慢均线
        self.fast_ma = bt.indicators.SimpleMovingAverage(
            self.datas[0].close, period=self.params.fast_ma
        )
        self.slow_ma = bt.indicators.SimpleMovingAverage(
            self.datas[0].close, period=self.params.slow_ma
        )
        self.order = None  # 跟踪订单

    def next(self):
        # 如果有未完成的订单，跳过
        if self.order:
            return

        # 如果持仓
        if self.position:
            # 快均线下穿慢均线，卖出
            if (
                self.fast_ma[0] < self.slow_ma[0]
                and self.fast_ma[-1] >= self.slow_ma[-1]
            ):
                self.order = self.sell()
        else:
            # 快均线上穿慢均线，买入
            if (
                self.fast_ma[0] > self.slow_ma[0]
                and self.fast_ma[-1] <= self.slow_ma[-1]
            ):
                self.order = self.buy()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                print(
                    f"买入: 价格 {order.executed.price:.2f}, 时间 {self.datas[0].datetime.date(0)}"
                )
            elif order.issell():
                print(
                    f"卖出: 价格 {order.executed.price:.2f}, 时间 {self.datas[0].datetime.date(0)}"
                )
            self.order = None


if __name__ == "__main__":
    """
    uv run z_using_files/test/test_bt.py
    """
    # 获取数据
    etf_daily_data_df = ak.fund_etf_hist_em(
        symbol="510310",
        period="daily",
        start_date="20241231",
        end_date="20250613",
        adjust="qfq",
    )

    # 重命名列以匹配 Backtrader 要求
    column_mapping = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
        "涨跌幅": "change_pct",
        "涨跌额": "change",
        "换手率": "turnover",
    }
    df = etf_daily_data_df.rename(columns=column_mapping)

    # 确保日期列为 datetime 格式并设置为索引
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    # 初始化 Cerebro
    cerebro = bt.Cerebro()

    # 将 Pandas DataFrame 转换为 Backtrader 数据源
    data = bt.feeds.PandasData(dataname=df)

    # 添加数据到 Cerebro
    cerebro.adddata(data)

    # 添加策略
    cerebro.addstrategy(MAStrategy)

    # 设置初始资金
    cerebro.broker.setcash(100000.0)

    # 设置交易佣金（0.1%）
    cerebro.broker.setcommission(commission=0.001)

    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")

    # 打印初始资金
    print(f"初始资金: {cerebro.broker.getvalue():.2f}")

    # 运行回测
    results = cerebro.run()

    # 打印最终资金和分析结果
    print(f"最终资金: {cerebro.broker.getvalue():.2f}")
    print(f"夏普比率: {results[0].analyzers.sharpe.get_analysis()['sharperatio']:.2f}")
    print(
        f"最大回撤: {results[0].analyzers.drawdown.get_analysis()['max']['drawdown']:.2f}%"
    )

    # 可视化结果
    cerebro.plot()
