## 面向 **A** 股的 **AI** 量化实战

### 项目安装

- 克隆本项目到本地后执行

```python
# 依赖安装
uv run install.py
```

- 环境激活
```shell
# 更多依赖安装示例
uv pip install -U akshare
uv add akshare -i https://pypi.tuna.tsinghua.edu.cn/simple
# win 激活
Set-ExecutionPolicy Bypass -Scope Process -Force
.venv\Scripts\activate
# linux 激活
source .venv/bin/activate
```

### 常见mapping
```python
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
    "指数代码": "index_code",
    "指数名称": "index_name",
    "指数英文名称": "index_english_name",
    "成分券代码": "stock_code",
    "成分券名称": "stock_name",
    "成分券英文名称": "stock_english_name",
    "交易所": "exchange",
    "交易所英文名称": "exchange_english_name",
}

table_mapping = {
    "cwzbsj": "财务指标数据表",
    "xjllb": "现金流量表",
    "zcfzb": "资产负债表",
    "lrb": "利润表",
    "stock_daily": "股票每日行情数据表",
}
```

### 名词

>布林带（Bollinger Bands）是一种广泛使用的技术分析工具，由约翰·布林格（John Bollinger）在20世纪80年代开发，用于衡量价格的波动性和识别市场趋势。它通过绘制三条线来直观地展示价格的波动范围和潜在的超买或超卖状态
>- **中轨（Middle Band）**：通常是价格的简单移动平均线（SMA），默认周期为20天，反映价格的中期趋势。
>- **上轨（Upper Band）**：中轨加上价格的标准差（通常为2倍标准差），表示价格的上限波动范围。
>- **下轨（Lower Band）**：中轨减去价格的标准差（通常为2倍标准差），表示价格的下限波动范围。
>布林带通过价格与三条线的位置关系，帮助投资者分析市场趋势和潜在交易机会：
>- **价格波动性**：
>   - 当布林带宽度（上轨与下轨之间的距离）变窄，表示市场波动性低，可能预示着即将出现大幅波动（称为“布林带收缩”）。
>   - 当布林带宽度变宽，表示市场波动性高，通常伴随着趋势性行情。
>- **超买与超卖**：
>   - 价格触及或突破上轨，可能表示市场处于**超买**状态，价格可能回调。
>   - 价格触及或跌破下轨，可能表示市场处于**超卖**状态，价格可能反弹。
>- **趋势判断**：
>   - 价格持续在上轨附近运行，通常表示强势上升趋势。
>   - 价格持续在下轨附近运行，通常表示强势下降趋势。
>   - 价格在中轨附近波动，表示市场处于震荡状态。

| 指标名称 | 简要描述 | 交易中典型用途 | 
| :---- | :---- | :---- | 
| SMA (简单移动平均) | 一段时间内收盘价的算术平均值 | 识别短期、中期、长期趋势 |
| EMA (指数移动平均) | 加权移动平均，近期价格权重更大 | 趋势跟踪，比SMA更敏感 | 
| RSI (相对强弱指数) | 衡量近期价格变化的幅度，判断超买超卖 | 识别潜在反转点，衡量市场动能 | 
| Bollinger Bands | 由移动平均线及上下两条标准差带构成 | 判断价格相对高低，衡量市场波动性 |
| MACD (平滑异同移动平均) | 两条指数移动平均线之差及其信号线 | 趋势识别，金叉死叉信号 | 
| OBV (能量潮) | 累计成交量，将成交量变化与价格方向联系起来 | 判断资金流入流出，验证趋势强度 | 

### Refernce

- [akshare入门](https://akshare.akfamily.xyz/introduction.html)
- [AI量化交易操盘手](https://github.com/aceliuchanghong/ai_quant_trade)
- [backtrader](https://github.com/aceliuchanghong/backtrader)
- [backtrader的微信教学](https://mp.weixin.qq.com/mp/appmsgalbum?action=getalbum&album_id=2380299870701420545)
- [中文backtrader开源笔记](https://github.com/aceliuchanghong/learn_backtrader)
- [中文backtrader开源笔记2](https://github.com/aceliuchanghong/backtrader_other)
- [Pybroker量化教学](https://github.com/aceliuchanghong/python_data_course)
-
