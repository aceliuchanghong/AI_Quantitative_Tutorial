import pandas as pd
import akshare as ak
import sys
import os

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")),
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


@cache_to_sqlite()
def get_trading_days(start_date, end_date):
    """获取交易日历"""
    calendar = ak.stock_zh_index_daily(symbol="sh000001")  # 使用上证指数获取交易日历
    calendar["date"] = pd.to_datetime(calendar["date"])
    calendar = calendar[
        (calendar["date"] >= start_date) & (calendar["date"] <= end_date)
    ]
    return calendar["date"].tolist()


@cache_to_sqlite(debug=False)
def get_daily_data(stock_code, start_date, end_date, adjust="qfq"):
    """
    获取单只股票的日频行情数据
    日期 股票代码 开盘 收盘 最高 最低 成交量 成交额 振幅 涨跌幅 涨跌额 换手率
    """
    try:
        df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )
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


def get_monthly_first_last_trading_days(trading_days):
    """
    输入: 交易日列表 str 或 datetime 类型
    输出：每月第一个和最后一个交易日的 list
    """
    # 确保是 datetime 类型
    trading_days = pd.to_datetime(trading_days)

    # 创建一个 Series，这样更容易处理
    s = pd.Series(trading_days, index=trading_days)

    # 按月份分组并使用 agg 获取第一个和最后一个日期
    monthly_days = s.groupby(s.dt.to_period("M")).agg(["first", "last"])

    first_days = monthly_days["first"].tolist()
    last_days = monthly_days["last"].tolist()

    return first_days, last_days


if __name__ == "__main__":
    # uv run z_using_files/backtrader_code/zz500_data.py
    start_data, end_data = "2023-01-01", "2025-06-01"
    adjust = "qfq"
    all_stock_daily = pd.DataFrame()

    stocks = get_csi500_stocks(date=start_data)
    if stocks is None or len(stocks) == 0:
        print("无法获取中证500成分股")
    else:
        print(f"获取到 {len(stocks)} 只中证500成分股")
        print(stocks[:2])

    for stock in stocks:
        stock_code = stock[0]
        df = get_daily_data(
            stock_code, start_data.replace("-", ""), end_data.replace("-", ""), adjust
        )
        if df.empty:
            print(f"{stock_code} 没有获取到数据")
        else:
            all_stock_daily = pd.concat([all_stock_daily, df], ignore_index=True)
    print(f"all_stock_daily.shape:{all_stock_daily.shape}")

    trading_days = get_trading_days(
        pd.to_datetime(start_data), pd.to_datetime(end_data)
    )
    print(f"获取到 {len(trading_days)} 个交易日")
    print(trading_days[:2])
    first_days_list, last_days_list = get_monthly_first_last_trading_days(trading_days)
    print("每月第一个交易日:")
    print(first_days_list[:2])
    print("每月最后一个交易日:")
    print(last_days_list[:2])
