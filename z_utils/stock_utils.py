import akshare as ak
import sys
import os
import pandas as pd

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")),
)
from z_utils.stock_sqlite import cache_to_sqlite
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

@cache_to_sqlite()
def get_daily_data(stock_code, start_date, end_date):
    """获取单只股票的日频行情数据"""
    try:
        # A股日线行情接口
        # 输出:日期 股票代码 开盘 收盘 最高 最低 成交量 成交额 振幅 涨跌幅 涨跌额 换手率
        df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="hfq",
        )
        # print(f"{df.head()}")
        df.rename(columns=column_mapping, inplace=True)
        if len(df) > 0:
            df["stock_code"] = stock_code
            return df[
                [
                    "date",
                    "stock_code",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "amount",
                    "amplitude",
                    "change_pct",
                    "change",
                    "turnover",
                ]
            ]
        else:
            print(f"{stock_code} 没有获取到数据")
            return pd.DataFrame()
    except Exception as e:
        print(f"获取 {stock_code} 数据失败: {e}")
        return []