### 前言

无论是实盘、模拟还是回测，核心都离不开“交易”。看似简单的“交易”二字，背后却关联着众多概念和复杂的运行逻辑。在 Backtrader 中，交易流程大致如下：

1. **设置交易条件**：包括初始资金、交易税费、滑点、成交量限制等；
2. **下达交易指令**：在 Strategy 策略逻辑中调用 `buy`、`sell`、`close` 或 `cancel` 指令；
3. **订单解读**：Order 模块解析交易订单，并将信息传递给经纪商 Broker 模块；
4. **订单检查**：经纪商 Broker 根据订单信息检查并决定是否接受订单；
5. **撮合成交**：经纪商 Broker 接收订单后，按要求撮合成交 trade 并进行结算；
6. **返回执行结果**：Order 模块返回经纪商 Broker 的订单执行结果。

### 回顾一下

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


### 讲解目录
Broker 经纪商模块和 Order 订单模块是交易相关的核心模块。
特别是 **Broker 模块**，从交易条件的设置到交易订单的执行，交易的方方面面都与之相关。

**Backtrader-交易篇** 将从以下五个方面详细讲解交易相关操作：
- 交易条件
- 交易函数
- 交易订单
- 交易执行
- 交易结算

### 交易条件

在回测过程中，交易条件的设置是关键环节，常见的交易条件包括：

- 初始资金
- 交易税费
- 滑点
- 期货保证金比率
- 成交量限制
- 涨跌幅限制
- 订单生成和执行时机的限制

上述大部分交易条件都可以通过 **Broker** 模块进行管理，Backtrader 提供了以下两种主要操作方式：

#### 1. 资金管理

1. **设置初始资金**：
   - 通过 `cash` 参数或 `set_cash()` 方法设置初始资金。
   - 简写形式：`setcash()`。

2. **增加或减少资金**：
   - 使用 `add_cash()` 方法增加（正数）或减少（负数）资金。

3. **查询资金**：
   - 通过 `get_cash()` 或简写形式 `getcash()` 获取当前可用资金。

Broker 会根据提交的订单检查现金需求与当前可用资金是否匹配。每次交易后，`cash` 会自动迭代更新，以反映当前头寸。


```python
import backtrader as bt

if __name__ == "__main__":
    cerebro = bt.Cerebro()
    # 初始化时设置资金
    cerebro.broker.set_cash(10000.0)  # 设置初始资金
    current_cash_long_form = cerebro.broker.get_cash()  # 获取当前可用资金
    print(f"当前金额: {current_cash_long_form}")

    cerebro.broker.add_cash(10000)
    current_cash_after_add = cerebro.broker.getcash()
    print(f"当前金额 + 10000: {current_cash_after_add}")

    cerebro.broker.add_cash(-5000)
    current_cash_after_subtract = cerebro.broker.getcash()
    print(f"当前金额 - 5000: {current_cash_after_subtract}")

    final_cash = cerebro.broker.getcash()
    print(f"当前金额: {final_cash}") # 此处故意留一个坑
```
```
当前金额: 10000.0
当前金额 + 10000: 20000.0
当前金额 - 5000: 15000.0
当前金额: 15000.0
```

#### 2. 持仓查询
资产与持仓管理

Broker 在每次交易后，除了更新可用资金 `cash` 外，还会同步更新以下内容：
- **当前总资产**（`value`）：反映账户的整体资产价值。
- **当前持仓**（`position`）：记录持仓量和持仓成本。


```python
class TestStrategy(bt.Strategy):
    def next(self):
        # 查询可用资金
        print('当前可用资金', self.broker.getcash())
        # 查询总资产
        print('当前总资产', self.broker.getvalue())
        # 查询持仓量
        print('当前持仓量', self.broker.getposition(self.data).size)
        # 查询持仓成本
        print('当前持仓成本', self.broker.getposition(self.data).price)
        # 直接获取持仓（简写方式）
        print('当前持仓量', self.getposition(self.data).size)
        print('当前持仓成本', self.getposition(self.data).price)
```
```
--- 资金操作演示 ---
初始资金设置为: 100000.00
增加 10000 后，当前资金: 100000.00
减少 5000 后，最终用于回测的资金: 100000.00
--------------------

开始回测...
2020-01-01 - 可用资金: 100000.00, 总资产: 100000.00
2020-01-01 - 当前无持仓
2020-01-02 - 可用资金: 100000.00, 总资产: 100000.00
2020-01-02 - 当前无持仓
2020-01-02 - >>> 尝试买入...
2020-01-03 - 买入执行成功: 价格 106.00, 成本 10600.00, 佣金 3.18
2020-01-03 - 可用资金: 89396.82, 总资产: 100096.82
2020-01-03 - 当前持仓: 100 股, 成本价: 106.00
2020-01-04 - 可用资金: 89396.82, 总资产: 100296.82
2020-01-04 - 当前持仓: 100 股, 成本价: 106.00
2020-01-05 - 可用资金: 89396.82, 总资产: 100496.82
2020-01-05 - 当前持仓: 100 股, 成本价: 106.00
2020-01-05 - <<< 尝试卖出平仓...
回测结束。

最终可用资金: 89396.82
最终总资产: 100496.82
```

#### 3. 滑点管理

#### 4. 交易税费管理

#### 5. 成交量限制管理

#### 6. 交易时机管理

### Refernce

- [Backtrader-03](https://mp.weixin.qq.com/s?__biz=MzAxNTc0Mjg0Mg==&mid=2653316528&idx=1&sn=24f2c06b8f7da8dee6fe40f7c65b83a6)
- 
