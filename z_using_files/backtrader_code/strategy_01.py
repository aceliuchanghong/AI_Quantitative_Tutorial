import backtrader as bt
import pandas as pd
from datetime import datetime
import uuid


class MyStrategy(bt.Strategy):
    params = (
        ("stock_pool_size", 500),  # 中证500股票池大小
        ("top_pct", 0.2),  # 选择前20%
        ("commission", 0.0003),  # 双边佣金0.03%
        ("slippage", 0.0001),  # 双边滑点0.01%
        ("rebalance_days", [1]),  # 每月第一个交易日调仓
        ("benchmark", "000905"),  # 中证500指数代码（仅作标识）
    )

    def __init__(self):
        super().__init__()
        self.returns = {}  # 存储股票上月收益率
        self.selected_stocks = []  # 当前选中的股票
        self.trading_day = 0
        self.last_month = None
        self.orders = {}  # 跟踪订单
        # 初始化分析器
        self.analyzers.returns = self.analyzers.getbyname("pyfolio")
        self.analyzers.sharpe = self.analyzers.getbyname("sharpe")
        self.analyzers.drawdown = self.analyzers.getbyname("drawdown")
        self.analyzers.tradeanalyzer = self.analyzers.getbyname("trade")

    def log(self, txt, dt=None, doprint=True):
        if doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print(f"{dt.isoformat()} {txt}")

    def next(self, ctx):
        # 获取当前日期
        current_date = self.datas[0].datetime.date(0)
        current_month = current_date.month

        # 判断是否为新月份
        if self.last_month is None or current_month != self.last_month:
            if self.last_month is not None:  # 不是第一次运行
                # 计算上月收益率（前复权）
                for data in self.datas:
                    if data.close[0] and data.close[-21]:  # 假设21个交易日约一个月
                        ret = data.close[0] / data.close[-21] - 1
                        self.returns[data._name] = ret
            self.last_month = current_month

    def next_open(self, ctx):
        current_date = self.datas[0].datetime.date(0)
        current_month = current_date.month

        # 判断是否为每月第一个交易日
        if self.last_month is None or current_month != self.last_month:
            # 1. 选择前20%股票
            sorted_returns = sorted(
                self.returns.items(), key=lambda x: x[1], reverse=True
            )
            top_n = int(self.params.stock_pool_size * self.params.top_pct)
            self.selected_stocks = [x[0] for x in sorted_returns[:top_n]]

            # 2. 计算流通市值并分配权重
            total_market_cap = 0
            market_caps = {}
            for data in self.datas:
                if data._name in self.selected_stocks:
                    # 假设data.volume存储流通股本，data.close为前复权价格
                    market_cap = data.volume[0] * data.close[0]
                    market_caps[data._name] = market_cap
                    total_market_cap += market_cap

            # 3. 清仓旧持仓
            for data in self.datas:
                if self.getposition(data).size > 0:
                    self.order_target_percent(data, target=0)
                    self.log(f"清仓 {data._name}")

            # 4. 按权重买入新选股
            cash = self.broker.getcash()
            for stock in self.selected_stocks:
                weight = market_caps[stock] / total_market_cap
                target_value = cash * weight
                price = self.datas[self.datas._name2id[stock]].open[0]
                size = int(target_value / (price * (1 + self.params.slippage)))
                order = self.buy(
                    data=self.datas[self.datas._name2id[stock]],
                    size=size,
                    exectype=bt.Order.Market,
                    valid=None,
                )
                self.orders[order.ref] = stock
                self.log(f"买入 {stock}, 数量 {size}, 权重 {weight:.2%}")

    def notify_order(self, order):
        if order.status in [order.Completed]:
            stock = self.orders.get(order.ref, "Unknown")
            if order.isbuy():
                self.log(
                    f"买入成交 {stock}, 价格 {order.executed.price}, 数量 {order.executed.size}"
                )
            elif order.issell():
                self.log(
                    f"卖出成交 {stock}, 价格 {order.executed.price}, 数量 {order.executed.size}"
                )
            if order.ref in self.orders:
                del self.orders[order.ref]

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f"交易关闭 {trade.data._name}, 盈亏 {trade.pnl:.2f}")

    def stop(self):
        # 输出回测结果
        self.log(f"最终账户价值: {self.broker.getvalue():.2f}")
        self.log(f"总回报: {(self.broker.getvalue() / 100000000 - 1) * 100:.2f}%")
        self.log(
            f'年化收益率: {self.analyzers.returns.get_analysis().get("rnorm100", 0):.2f}%'
        )
        self.log(
            f'夏普比率: {self.analyzers.sharpe.get_analysis().get("sharperatio", 0):.2f}'
        )
        self.log(
            f'最大回撤: {self.analyzers.drawdown.get_analysis().get("max", {}).get("drawdown", 0):.2f}%'
        )
        trades = self.analyzers.tradeanalyzer.get_analysis()
        self.log(f'交易次数: {trades.get("total", {}).get("total", 0)}')
        turnover = trades.get("total", {}).get("pnl", {}).get("gross", 0) / 100000000
        self.log(f"换手率: {turnover:.2f}")


# 主程序
if __name__ == "__main__":
    cerebro = bt.Cerebro()
    cerebro.addstrategy(MyStrategy)

    # 添加数据（假设有中证500成分股数据）
    # 这里需要你提供实际的数据源
    # for stock in csi500_stocks:
    #     data = bt.feeds.PandasData(dataname=stock_data)
    #     cerebro.adddata(data)

    # 设置初始资金
    cerebro.broker.setcash(100000000)
    # 设置佣金
    cerebro.broker.setcommission(commission=0.0003)
    # 设置滑点
    cerebro.broker.set_slippage_perc(perc=0.0001)

    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name="pyfolio")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trade")

    # 运行回测
    results = cerebro.run()

    # 可视化净值曲线
    cerebro.plot()
