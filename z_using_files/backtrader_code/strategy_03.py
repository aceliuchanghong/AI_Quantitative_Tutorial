import backtrader as bt
import os
import datetime


class SimpleStrategy(bt.Strategy):
    """
    一个简单的 Backtrader 策略，演示如何查询资金、持仓，
    并在订单和交易通知中打印操作日期。
    """

    def __init__(self):
        """
        策略初始化。
        """
        self.order = None  # 用于跟踪挂单

    def log(self, txt, dt=None):
        """
        自定义日志函数，用于打印带日期的日志。
        """
        dt = dt or self.datas[0].datetime.date(0)
        print(f"{dt.isoformat()} - {txt}")

    def next(self):
        """
        每个bar K线 都会调用的核心方法。
        """
        # 打印当前周期的账户信息
        self.log(
            f"可用资金: {self.broker.getcash():.2f}, 总资产: {self.broker.getvalue():.2f}"
        )

        # 检查是否有持仓
        if self.position:
            self.log(
                f"当前持仓: {self.position.size} 股, 成本价: {self.position.price:.2f}"
            )
        else:
            self.log("当前无持仓")

        # 简化逻辑：在第二个K线（索引为1）时买入
        # len(self) 返回已经处理的K线数量，从1开始计数
        if not self.position and len(self) == 2:
            self.log(">>> 尝试买入...")
            # 下一个买入订单，购买100股
            self.buy(size=100)

        # 简化逻辑：在第五个K线（索引为4）时卖出
        if self.position and len(self) == 5:
            self.log("<<< 尝试卖出平仓...")
            # 平仓所有持仓
            self.close()

    def notify_order(self, order):
        """
        订单状态变化通知。
        """
        # 获取订单发生时的日期
        order_date = self.datas[0].datetime.date(0)

        if order.status in [order.Submitted, order.Accepted]:
            # 订单已提交或已被券商接受，无需处理
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f"买入执行成功: 价格 {order.executed.price:.2f}, "
                    f"成本 {order.executed.value:.2f}, "
                    f"佣金 {order.executed.comm:.2f}",
                    dt=order_date,
                )
            elif order.issell():
                self.log(
                    f"卖出执行成功: 价格 {order.executed.price:.2f}, "
                    f"成本 {order.executed.value:.2f}, "
                    f"佣金 {order.executed.comm:.2f}",
                    dt=order_date,
                )

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(
                f"订单失败/取消/保证金不足/拒绝: {order.getstatusname()}", dt=order_date
            )

        # 重置订单跟踪
        self.order = None

    def notify_trade(self, trade):
        """
        交易完成通知。一个交易由一个买单和一个卖单组成。
        """
        if trade.isclosed:
            # 获取交易平仓时的日期
            trade_date = self.datas[0].datetime.date(0)
            self.log(
                f"交易平仓: 毛利 {trade.pnl:.2f}, 净利 {trade.pnlcomm:.2f}",
                dt=trade_date,
            )


if __name__ == "__main__":
    """
    uv run z_using_files/backtrader_code/strategy_03.py
    """
    cerebro = bt.Cerebro()

    # --- 创建一个虚拟的 CSV 数据文件用于演示 ---
    data_dir = "no_git_oic"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    data_file_name = os.path.join(data_dir, "test_data.csv")
    if not os.path.exists(data_file_name):
        with open(data_file_name, "w") as f:
            # Join all lines and write once, ensuring no trailing empty line
            lines = [
                "Date,Open,High,Low,Close,Volume",
                "2020-01-01,100,105,99,103,100000",
                "2020-01-02,103,107,102,106,120000",
                "2020-01-03,106,108,105,107,90000",
                "2020-01-04,107,110,106,109,110000",
                "2020-01-05,109,112,108,111,130000",
            ]
            f.write("\n".join(lines))
        print(f"已创建虚拟数据文件: {data_file_name}")
    # -----------------------------------------------

    # 2. 添加策略
    cerebro.addstrategy(SimpleStrategy)

    # 3. 添加数据
    # 使用 GenericCSVData 加载 CSV 文件
    data = bt.feeds.GenericCSVData(
        dataname=data_file_name,
        dtformat="%Y-%m-%d",  # CSV 文件中日期的格式
        datetime=0,  # 日期在第0列
        open=1,  # 开盘价在第1列
        high=2,  # 最高价在第2列
        low=3,  # 最低价在第3列
        close=4,  # 收盘价在第4列
        volume=5,  # 成交量在第5列
        openinterest=-1,  # 如果没有持仓量数据，可以设置为 -1
        reverse=False,  # 数据是否需要反转（如果最新的数据在最上面，则需要设置为 True）
    )
    cerebro.adddata(data)

    # 4. 设置初始资金
    print("\n--- 资金操作演示 ---")
    initial_cash = 100000.0
    cerebro.broker.setcash(initial_cash)
    print(f"初始资金设置为: {cerebro.broker.getcash():.2f}")

    # 增加资金
    cerebro.broker.add_cash(10000)
    print(f"增加 10000 后，当前资金: {cerebro.broker.getcash():.2f}")

    # 减少资金 (通过添加负数)
    cerebro.broker.add_cash(-5000)
    print(f"减少 5000 后，最终用于回测的资金: {cerebro.broker.getcash():.2f}")
    print("--------------------")

    # 5. 设置佣金（例如：万分之三）
    cerebro.broker.setcommission(commission=0.0003)

    # 6. 运行回测
    print("\n开始回测...")
    cerebro.run()
    print("回测结束。\n")

    # 7. 打印最终结果
    print(f"最终可用资金: {cerebro.broker.getcash():.2f}")
    print(f"最终总资产: {cerebro.broker.getvalue():.2f}")
