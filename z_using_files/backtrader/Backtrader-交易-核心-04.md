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

### 滑点管理

#### 什么是滑点？

在量化交易中，**滑点 (Slippage)** 是指交易指令中预设的执行价格与最终实际成交价格之间的差异。

这种差异主要由以下原因造成：

  * **市场波动性：** 从下单到交易所撮合成交的短暂时间内，市场价格可能已经发生了变化。
  * **网络延迟：** 指令从计算机传输到交易所需要时间。
  * **订单深度不足：** 当订单数量较大时，可能会吃掉多个价位的订单，导致平均成交价偏离预期。

在回测中，我们必须对滑点进行模拟，因为一个忽略了滑点的策略，其实际表现往往会远逊于回测结果。通过在回测中加入滑点，我们可以让策略评估变得更加严谨和贴近真实。

滑点的核心作用是让回测的交易成本更高：**买入时，以更高的价格成交；卖出时，以更低的价格成交**。即所谓的“买得更贵，卖得更便宜”。

主要有两种类型：**百分比滑点**和**固定滑点**。

> **注意**：在 Backtrader 中，如果同时设置了百分比滑点和固定滑点，**百分比滑点 (`slip_perc`) 的优先级更高**，系统将忽略固定滑点的设置。

####  百分比滑点 (Percentage Slippage)

按预设成交价的一定百分比进行滑动。假设设置了 $n%$ 的滑点：

  * **买入**：若指定价格为 $x$，则实际成交价为 $x \times (1 + n\%)$
  * **卖出**：若指定价格为 $x$，则实际成交价为 $x \times (1 - n\%)$

**示例：** 设置 0.01% (0.0001) 的滑点。

```python
# 方式1：在初始化 Broker 时通过 slip_perc 参数设置
cerebro.broker = bt.brokers.BackBroker(slip_perc=0.0001)

# 方式2：通过 broker 的 set_slippage_perc 方法设置
cerebro.broker.set_slippage_perc(perc=0.0001)
```

#### 固定滑点 (Fixed Slippage)

按预设成交价加上或减去一个固定的数值。假设设置了大小为 $n$ 的固定滑点：

  * **买入**：若指定价格为 $x$，则实际成交价为 $x + n$
  * **卖出**：若指定价格为 $x$，则实际成交价为 $x - n$

**示例：** 设置 0.01 的固定滑点。

```python
# 方式1：在初始化 Broker 时通过 slip_fixed 参数设置
cerebro.broker = bt.brokers.BackBroker(slip_fixed=0.01)

# 方式2：通过 broker 的 set_slippage_fixed 方法设置
cerebro.broker.set_slippage_fixed(fixed=0.01)
```

#### 其他高级滑点参数

除了基础的滑点设置，`Backtrader` 的 `broker` 还提供了更精细的控制参数，用于处理应用滑点后可能出现的极端情况。

  * `slip_open` (布尔值):

      * **作用**：决定是否对 **以开盘价成交的订单** 应用滑点。
      * **默认值**：在 `BackBroker()` 初始化时默认为 `False`；在 `set_slippage_perc` 和 `set_slippage_fixed` 方法中默认为 `True`。

  * `slip_match` (布尔值):

      * **作用**：决定是否将滑点处理后的价格与成交当日的 `Low` ~ `High` 价格区间进行匹配。
      * `True`：如果滑点后的价格超出了当日价格范围，系统会尝试将成交价调整到范围的边界（最高价或最低价），以确保订单成交。
      * `False`：如果滑点后的价格超出了当日价格范围，该订单在该K线上将被拒绝，无法成交。
      * **默认值**：`True`。

  * `slip_out` (布尔值):

      * **作用**：决定是否允许成交价**超出**当日的 `Low` ~ `High` 范围。
      * `True`：允许以超出当日价格范围的滑点价成交。
      * `False`：如果滑点价超出了范围，实际成交价将被 **限制（Clamp）** 在当日的最高价或最低价。
      * **默认值**：`False`。

  * `slip_limit` (布尔值):

      * **作用**：决定是否对**限价单 (Limit Order)** 应用滑点。
      * `True`：即使 `slip_match` 被设为 `False`，系统仍然会为限价单撮合价格，确保其在价格范围内成交。
      * `False`：遵循 `slip_match` 的规则。
      * **默认值**：`True`。

#### 参数组合应用实例

```
        date  stock_code   open   high    low  close  volume     amount  amplitude  change_pct  change  turnover
0 2025-07-23      603678  37.29  37.29  36.22  36.31   89233  326423483       2.87       -2.71   -1.01      1.88
```

以下我们以固定滑点为 `0.35`，针对标的 `603678` 在 `2025-07-23 ` 的买入操作进行对比分析。

**当日行情数据:**

  * **日期**: `2025-07-23 `
  * **开盘价 (Open)**: `37.29`
  * **收盘价 (Open)**: `36.31`
  * **最低价 (Low)**: `36.22`
  * **最高价 (High)**: `37.29`

**交易指令假设**：策略在 `2025-07-23` 以**开盘价**执行买入指令。

##### 情况1：不希望开盘成交的订单受滑点影响

**参数设置:**

```python
cerebro.broker.set_slippage_fixed(fixed=0.35,
                                  slip_open=False, # 核心参数
                                  slip_match=True, 
                                  slip_out=False)
```

**成交分析:**

  * 因为核心参数 `slip_open` 被设置为 `False`，所以系统**不会**对以开盘价成交的订单应用滑点。
  * 因此，尽管全局设置了 `fixed=0.35` 的滑点，该笔订单依然会以原始的开盘价 `37.29` 成交。

**最终成交价** = **`37.29`**

##### 情况2：开盘成交订单受滑点影响，且成交价不能超出市场范围

**参数设置:**

```python
cerebro.broker.set_slippage_fixed(fixed=0.35, 
                                  slip_open=True,  # 核心参数
                                  slip_match=True, # 核心参数
                                  slip_out=False)  # 核心参数
```

**成交分析:**

1.  **计算滑点价**：由于 `slip_open=True`，系统需要对开盘价应用滑点。

      * 预设买入价 (开盘价) = `37.29`
      * 固定滑点 = `0.35`
      * 滑点后目标价格 = `37.29 + 0.35 = 37.64`

2.  **匹配市场价格范围**：计算出的滑点后价格 `37.64` **高于** 当日的市场最高价 `37.29`。

3.  **最终成交价决策**：

      * `slip_match=True` 要求系统尽可能撮合交易，而不是直接拒绝订单。
      * `slip_out=False` 禁止最终成交价超出当日的 `Low` \~ `High` 价格范围。
      * 综合以上两点，系统会将最终成交价**向下限制（Clamp）在当日的最高价**。

  * **最终成交价** = `37.29`

**结论**
在这个特殊的案例中，由于开盘价即为当日最高价，滑点后的价格必然会超出范围。因此，无论 `slip_open` 设置为 `True` 还是 `False`，在 `slip_out=False` 的限制下，最终的成交价都会是当日的最高价 `37.29`。这个例子很好地展示了 `slip_out` 参数在防止模拟交易价格脱离实际市场中的关键作用。

---

### 交易税费管理

交易时是否考虑交易费用对回测的结果影响很大，所以在回测是通常会设置交易税费，不同标的的费用收取规则也各不相同。


#### 股票

目前 A 股的交易费用分为 2 部分：佣金和印花税。

  - **佣金**：双边征收，不同证券公司收取的佣金各不相同，一般在 0.02%-0.03% 左右，单笔佣金不少于 5 元。
  - **印花税**：只在卖出时收取，税率为 0.1%。

#### 期货

期货交易费用包括交易所收取手续费和期货公司收取佣金 2 部分。

  - **交易所手续费**：较为固定。
  - **期货公司佣金**：不一致，且不同期货品种的收取方式不相同，有的按照固定费用收取，有的按成交金额的固定百分比收取（计算公式：`合约现价 * 合约乘数 * 手续费费率`）。
  - **保证金**：除了交易费用外，期货交易时还需上交一定比例的保证金。


Backtrader 提供了多种交易费设置方式，既可以简单的通过参数进行设置，也可以结合交易条件自定义费用函数。

#### 交易费用模式

  - **股票 (Stock-like) 模式**: 对应 **PERC 百分比费用模式**。
  - **期货 (Futures-like) 模式**: 对应 **FIXED 固定费用模式**。

#### 核心参数

在设置交易费用时，最常涉及如下 3 个参数：

  - `commission`：手续费 / 佣金。
  - `mult`：乘数。
  - `margin`：保证金 / 保证金比率。

**双边征收**：买入和卖出操作都要收取相同的交易费用。


#### 1. 通过 `BackBroker()` 设置

`BackBroker` 中有一个 `commission` 参数，用来全局设置交易手续费。如果是股票交易，可以简单的通过该方式设置交易佣金。

```python
# 设置 0.0002 = 0.02% 的手续费
cerebro.broker = bt.brokers.BackBroker(commission=0.0002)
```

#### 2. 通过 `setcommission()` 设置

如果想要完整又方便的设置交易费用，可以调用 broker 的 `setcommission()` 方法，该方法基本上可以满足大部分的交易费用设置需求。

```python
cerebro.broker.setcommission(
    # 交易手续费，根据margin取值情况区分是百分比手续费还是固定手续费
    commission=0.0,
    # 期货保证金，决定着交易费用的类型,只有在stocklike=False时起作用
    margin=None,
    # 乘数，盈亏会按该乘数进行放大
    mult=1.0,
    # 交易费用计算方式，取值有：
    # 1. CommInfoBase.COMM_PERC 百分比费用
    # 2. CommInfoBase.COMM_FIXED 固定费用
    # 3. None 根据 margin 取值来确定类型
    commtype=None,
    # 当交易费用处于百分比模式下时，commission 是否为 % 形式
    # True，表示不以 % 为单位，0.XX 形式；False，表示以 % 为单位，XX% 形式
    percabs=True,
    # 是否为股票模式，该模式通常由margin和commtype参数决定
    # margin=None或COMM_PERC模式时，就会stocklike=True，对应股票手续费；
    # margin设置了取值或COMM_FIXED模式时,就会stocklike=False，对应期货手续费
    stocklike=False,
    # 计算持有的空头头寸的年化利息
    # days * price * abs(size) * (interest / 365)
    interest=0.0,
    # 计算持有的多头头寸的年化利息
    interest_long=False,
    # 杠杆比率，交易时按该杠杆调整所需现金
    leverage=1.0,
    # 自动计算保证金
    # 如果False,则通过margin参数确定保证金
    # 如果automargin<0,通过mult*price确定保证金
    # 如果automargin>0,如果automargin*price确定保证金
    automargin=False,
    # 交易费用设置作用的数据集(也就是作用的标的)
    # 如果取值为None，则默认作用于所有数据集(也就是作用于所有assets)
    name=None
)
```

从上述各参数的含义和作用可知，`margin`、`commtype`、`stocklike` 存在 2 种默认的配置规则：

1.  **股票百分比费用**：未设置 `margin`（即 `margin` 为 `0` / `None` / `False`）→ `commtype` 会指向 `COMM_PERC` 百分比费用 → 底层的 `_stocklike` 属性会设置为 `True`。所以如果想为股票设置交易费用，就令 `margin = 0 / None / False`，或者令 `stocklike=True`。
2.  **期货固定费用**：为 `margin` 设置了取值 → `commtype` 会指向 `COMM_FIXED` 固定费用 → 底层的 `_stocklike` 属性会设置为 `False`。所以如果想为期货设置交易费用，就需要设置 `margin`，此外还需令 `stocklike=False`，`margin` 参数才会起作用。
