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

### 成交量限制管理


### 交易时机管理


