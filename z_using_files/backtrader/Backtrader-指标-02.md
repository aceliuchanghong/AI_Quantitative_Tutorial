## Backtrader 指标

### 0. 回顾一下

`Backtrader` 以大脑 `cerebro` 为统一的调度中心，数据、策略、回测条件等信息都会导入 `cerebro` 中，并由 `cerebro` 启动和完成回测，最后返回回测结果

##### 选股回测流程示例:

实例化大脑 → 导入数据 → 配置回测条件 → 编写交易逻辑 → 运行回测 → 提取回测结果 

##### Backtrader架构:
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

### 1. 指标在哪些地方使用？

在 Backtrader 的主要功能模块和回测流程，指标的计算和使用主要集中在**策略-Strategy**的开发中，尤其是在以下两个方法中：

- `__init__()`：用于初始化策略，通常在此计算指标。
- `next()`：在每个交易日循环调用，用于调用已计算的指标或执行交易逻辑。

```python
import backtrader as bt
import backtrader.indicators as btind

class MyStrategy(bt.Strategy):
  # 先在 __init__ 中提前算好指标
    def __init__(self):
        sma1 = btind.SimpleMovingAverage(self.data)
        ema1 = btind.ExponentialMovingAverage()
        close_over_sma = self.data.close > sma1
        close_over_ema = self.data.close > ema1
        sma_ema_diff = sma1 - ema1
        # 生成交易信号
        buy_sig = bt.And(close_over_sma, close_over_ema, sma_ema_diff > 0)
    # 在 next 中直接调用计算好的指标
    def next(self):
        if buy_sig:
            self.buy()
```

### 2. 在`__init__()`中提前计算指标

遵循 `__init__()` 指标计算，`next()` 指标调用的原则，以优化回测性能

- 在 `__init__()` 中一次性计算好指标（甚至包括交易信号）。
- 在 `next()` 中直接调用已计算的指标，避免重复计算，提升运行速度。

```python
import backtrader.indicators as btind 

class MyStrategy(bt.Strategy):
    def __init__(self):
        # 在 __init__ 中提前计算指标
        sma1 = btind.SimpleMovingAverage(self.data)  # 简单移动均线
        ema1 = btind.ExponentialMovingAverage()      # 指数移动均线
        close_over_sma = self.data.close > sma1      # 收盘价是否高于简单均线
        close_over_ema = self.data.close > ema1      # 收盘价是否高于指数均线
        sma_ema_diff = sma1 - ema1                   # 均线差值
        # 生成交易信号
        buy_sig = bt.And(close_over_sma, close_over_ema, sma_ema_diff > 0)

    def next(self):
        # 在 next 中直接调用计算好的指标
        if buy_sig:
            self.buy()
```

### 3. 自定义指标

```python
class My_MACD(bt.Indicator):
    lines = ('macd', 'signal', 'histo') # 定义指标线名称
    params = (
        ('period_me1', 12),
        ('period_me2', 26),
        ('period_signal', 9),
    ) # 定义参数

    def __init__(self):
        me1 = EMA(self.data, period=self.p.period_me1)
        me2 = EMA(self.data, period=self.p.period_me2)
        self.l.macd = me1 - me2
        self.l.signal = EMA(self.l.macd, period=self.p.period_signal)
        self.l.histo = self.l.macd - self.l.signal
```

