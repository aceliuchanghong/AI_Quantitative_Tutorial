## 基于 510310 ETF（沪深300ETF）的 AI 量化实战

### 项目安装

- 克隆本项目到本地后执行

```python
# 依赖安装
uv run install.py
```

- 环境激活
```shell
# 更多依赖安装示例
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
```


### Refernce

- [akshare入门](https://akshare.akfamily.xyz/introduction.html)
- [AI量化交易操盘手](https://github.com/aceliuchanghong/ai_quant_trade)
- [backtrader](https://github.com/aceliuchanghong/backtrader)
- [中文backtrader开源笔记](https://github.com/aceliuchanghong/learn_backtrader)
- [中文backtrader开源笔记2](https://github.com/aceliuchanghong/backtrader_other)
- [backtrader官方文档](https://www.backtrader.com/home/helloalgotrading/)
- 