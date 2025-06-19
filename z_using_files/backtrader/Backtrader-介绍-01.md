## Backtrader介绍

`Backtrader` 以大脑 `cerebro` 为统一的调度中心，数据、策略、回测条件等信息都会导入 `cerebro` 中，并由 `cerebro` 启动和完成回测，最后返回回测结果

- 架构如下:
```
Cerebro
├── DataFeeds (数据模块)
│   ├── CSVDataBase 导入CSV
│   ├── PandasData 导入 df
│   └── YahooFinanceData 导入网站数据 ...
├── Strategy (策略模块)
│   ├── next() 主策略函数
│   ├── notify_order、notify_trade 打印订单、交易信息 ...
├── Indicators (指标模块)
│   ├── SMA、EMA 移动均线
│   └── Ta-lib 技术指标库 ...
├── Orders (订单模块)
│   ├── buy() 买入
│   ├── sell() 卖出
│   ├── close() 平仓
│   └── cancel() 取消订单 ...
├── Sizers (仓位模块)
├── Broker (经纪商模块)
│   ├── cash 初始资金
│   ├── commission 手续费
│   └── slippage 滑点 ...
├── Analyzers (策略分析模块)
│   ├── AnnualReturn 年化收益
│   ├── SharpeRatio 夏普比率
│   ├── DrawDown 回撤
│   └── PyFolio 分析工具 ...
└── Observers (观测器模块)
    ├── Broker 资金\市值曲线
    ├── Trades 盈亏曲线
    └── BuySell 买卖点
```

#### 术语解释

- **回测**  
  回测是指利用历史市场数据（如价格、成交量等）模拟交易策略的表现，以评估其在过去市场环境下的潜在收益、风险和稳定性。在Backtrader中，回测通过加载历史数据（如CSV或在线数据源）、定义交易策略逻辑、设置初始资金和交易规则（如佣金、滑点）来执行。回测结果包括净值曲线、收益率、最大回撤等指标，帮助优化策略，但需注意历史数据质量和过拟合风险。

- **滑点**  
  滑点是指实际成交价格与预期成交价格之间的差额，通常由市场流动性不足、价格波动或交易延迟引起。在Backtrader中，滑点可以通过设置`slippage`参数模拟，例如固定滑点（固定价格差）或百分比滑点（按成交金额比例）。滑点的引入使回测更贴近真实交易环境，特别是在高频交易或低流动性市场中，需合理设置以避免过于乐观的回测结果。

- **策略**  
  策略是量化交易的核心，定义了基于市场数据（如K线、指标）进行买入、卖出或持仓的逻辑规则。在Backtrader中，策略通过继承`bt.Strategy`类实现，包含`next()`方法处理每根K线的决策逻辑，以及可选的`notify_order()`和`notify_trade()`方法监控订单和交易状态。策略可结合技术指标（如移动平均线、RSI）、资金管理（如固定比例下单）和风险控制（如止损止盈）设计。

- **中证500成分股**
  中证500成分股是指中证500指数的构成股票，由中证指数有限公司编制，选取A股市场中剔除沪深300指数成分股及总市值排名前300的股票后，总市值排名靠前的500只中小市值股票组成。这些股票代表中国A股市场中等市值公司的整体表现，覆盖多个行业，流通市值较中小盘股为主。在Backtrader回测中，可使用中证500成分股的历史数据（如通过Yahoo Finance或东方财富获取）进行策略测试，以评估策略在中小市值股票市场中的表现。成分股每半年调整一次（6月和12月），需注意调整对策略的影响。


#### 通常的回测流程

- 构建策略
    - 确定策略潜在的可调参数
    - 计算策略中用于生成交易信号的指标
    - 按需打印交易信息
    - 编写买入、卖出的交易逻辑
- 实例化`cerebro`==>驱动回测
    - 由 DataFeeds 加载数据，再将加载的数据添加给 cerebro
    - 将上一步生成的策略添加给 cerebro
    - 按需添加策略分析指标或观测器
    - 通过运行 cerebro.run() 来启动回测
    - 回测完成后，按需运行 cerebro.plot() 进行回测结果可视化展示


```
+-------------------+                       +-------------------+
| 1. 准备回测数据   |                       | 4. 设置回测参数   |
+-------------------+                       +-------------------+
          \                                       /
           \                                     /
            v                                   v
+-------------------+                       +-------------------+
| 2. 编写策略       |                       | 5. 设置绩效分析指标|
+-------------------+                       +-------------------+
          \                                       /
           \                                     /
            v                                   v
                    +-------------------+  
                    | 3. 实例化         |  
                    | cerebro = Cerebro()|   
                    +-------------------+  
                            |                                       
                            |                                     
                            v                                   
                    +-------------------+                       
                    | 6. 运行回测       |                      
                    | cerebro.run()     |                     
                    +-------------------+        
                            |                                       
                            |                                     
                            v                                   
                    +-------------------+   
                    | 7. 获得回测结果   |   
                    +-------------------+  
```

#### 选股回测流程示例

实例化大脑 → 导入数据 → 配置回测条件 → 编写交易逻辑 → 运行回测 → 提取回测结果 

```python
import backtrader as bt
import backtrader.indicators as btind

# 创建策略
class TestStrategy(bt.Strategy):
    params = (
        ('maperiod', 20),  # 移动均线周期，示例设为20
    )

    def log(self, txt, dt=None):
        '''打印日志'''
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}, {txt}')

    def __init__(self):
        '''初始化属性和指标'''
        # 计算简单移动均线
        self.sma = btind.SimpleMovingAverage(self.datas[0].close, period=self.params.maperiod)
        # 跟踪订单状态
        self.order = None

    def notify_order(self, order):
        '''处理订单状态'''
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'买入执行, 价格: {order.executed.price:.2f}, 数量: {order.executed.size}')
            elif order.issell():
                self.log(f'卖出执行, 价格: {order.executed.price:.2f}, 数量: {order.executed.size}')
            self.order = None  # 重置订单状态

    def next(self):
        '''交易逻辑'''
        if self.order:  # 检查是否有未完成订单
            return

        # 示例策略：价格突破均线买入，跌破均线卖出
        if not self.position:  # 没有持仓
            if self.datas[0].close[0] > self.sma[0]:
                self.order = self.buy(size=100)  # 买入100股
                self.log('发出买入信号')
        else:
            if self.datas[0].close[0] < self.sma[0]:
                self.order = self.sell(size=100)  # 卖出100股
                self.log('发出卖出信号')

# 实例化Cerebro引擎
cerebro = bt.Cerebro()

# 加载数据
data = bt.feeds.YahooFinanceCSVData(
    dataname='data.csv',
    fromdate=datetime.datetime(2020, 1, 1),
    todate=datetime.datetime(2023, 12, 31)
)
cerebro.adddata(data)

# 通过经纪商设置初始资金
cerebro.broker.setcash(100000.0)  # 初始资金10万

# 设置交易单位（固定100股）
cerebro.addsizer(bt.sizers.FixedSize, stake=100)

# 设置佣金（示例：0.1%）
cerebro.broker.setcommission(commission=0.001)

# 添加策略
cerebro.addstrategy(TestStrategy)

# 添加分析指标（如夏普比率、年化收益率）
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

# 运行回测
results = cerebro.run()

# 打印分析结果
strat = results[0]
print(f"夏普比率: {strat.analyzers.sharpe.get_analysis()['sharperatio']:.2f}")
print(f"最大回撤: {strat.analyzers.drawdown.get_analysis()['max']['drawdown']:.2f}%")
print(f"总收益率: {strat.analyzers.returns.get_analysis()['rtot']*100:.2f}%")

# 可视化结果
cerebro.plot()
```

#### 实战回测

##### 策略说明

1. 按收益率降序排序，选择前 20% 的股票
2. 在每月最后一个交易日，计算成分股上个月的收益率
3. 在每月第一个交易日，以开盘价清仓旧持仓并买入新选股
4. 持仓权重根据流通市值占比分配
5. 考虑 0.03% 双边佣金和 0.01% 双边滑点
6. 使用 Backtrader 进行回测，设置初始资金 1 亿元
7. 添加夏普比率、最大回撤和总回报分析器
8. 输出回测结果并可视化净值曲线

| 股票池         | 中证 500 成分股。 |
|----------------|--------------------|
| 回测区间       | 2023-12-01 至 2025-06-01。 |
| 持仓周期       | 月度调仓，每月第一个交易日，以开盘价买入或卖出。 |
| 持仓权重       | 流通市值占比。 |
| 总资产         | 100,000,000 元。 |
| 佣金           | 0.0003 双边。 |
| 滑点           | 0.0001 双边。 |
| 策略逻辑       | 选择中证 500 成分股中表现最优的前 20% 的股票作为下一个月的持仓成分股，然后在下个月的第一个交易日，卖出已有持仓，买入新的持仓。 |

- 获取交易日历
```python
import pandas as pd
import akshare as ak

def get_trading_days(start_date, end_date):
    """使用上证指数获取交易日历"""
    calendar = ak.stock_zh_index_daily(symbol="sh000001")
    calendar["date"] = pd.to_datetime(calendar["date"])
    calendar = calendar[
        (calendar["date"] >= start_date) & (calendar["date"] <= end_date)
    ]
    return calendar["date"].tolist()

trading_days = get_trading_days(
    pd.to_datetime("2023-12-01"), pd.to_datetime("2025-06-01")
)
if not trading_days:
    print("无法获取交易日历")
else:
    print(f"获取到 {len(trading_days)} 个交易日")
    # 打印前5个交易日
    print(trading_days[:2])
```

```
获取到 360 个交易日
[Timestamp('2023-12-01 00:00:00'), Timestamp('2023-12-04 00:00:00')]
```

- 获取中证 500 成分股数据
  
```python
import pandas as pd
import akshare as ak

column_mapping = {
    "日期": "date",
    "指数代码": "index_code",
    "指数名称": "index_name",
    "指数英文名称": "index_english_name",
    "成分券代码": "stock_code",
    "成分券名称": "stock_name",
    "成分券英文名称": "stock_english_name",
    "交易所": "exchange",
    "交易所英文名称": "exchange_english_name",
}

def get_csi500_stocks(date):
    """获取指定日期的中证 500 成分股列表"""
    try:
        csi500 = ak.index_stock_cons_csindex(symbol="000905")
        csi500.rename(columns=column_mapping, inplace=True)
        csi500["date"] = pd.to_datetime(csi500["date"])
        latest_date = csi500["date"].max()
        latest_stocks = csi500[csi500["date"] == latest_date]
        print(f"使用最新成分股数据，日期为：{latest_date.strftime('%Y-%m-%d')}")
        return latest_stocks[["stock_code", "stock_name"]].values.tolist()
    except Exception as e:
        print(f"获取中证 500 成分股失败: {e}")
        return []

stocks = get_csi500_stocks("2023-12-01")
if not stocks:
    print("无法获取中证 500 成分股")
else:
    print(f"获取到 {len(stocks)} 只中证 500 成分股")
    print(stocks[:2])
```

```
使用最新成分股数据，日期为：2025-06-17
获取到 500 只中证 500 成分股
[['000009', '中国宝安'], ['000021', '深科技']]
```


#### 展望

很多细节未做深入讲解,下一节继续

### Refernce
- [Backtrader来了](https://mp.weixin.qq.com/s/7S4AnbUfQy2kCZhuFN1dZw)