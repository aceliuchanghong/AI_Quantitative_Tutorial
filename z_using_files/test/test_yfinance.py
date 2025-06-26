import yfinance as yf
import pandas as pd
import sys
import os

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")),
)
from z_utils.proxy import set_proxy
from z_utils.stock_sqlite import cache_to_sqlite


@set_proxy()
@cache_to_sqlite()
def get_stock_data(ticker_symbol):
    # 获取股票数据
    ticker = yf.Ticker(ticker_symbol)

    # 获取股息数据
    dividends = ticker.dividends
    dividends_df = pd.DataFrame(dividends).reset_index()
    dividends_df.columns = ["Date", "Dividend"]
    dividends_df["Type"] = "Dividend"

    # 获取拆股数据
    splits = ticker.splits
    splits_df = pd.DataFrame(splits).reset_index()
    splits_df.columns = ["Date", "Split_Ratio"]
    splits_df["Type"] = "Split"

    # 合并数据
    combined_df = pd.concat([dividends_df, splits_df], ignore_index=True)
    combined_df = combined_df.sort_values("Date")

    return combined_df


if __name__ == "__main__":
    """
    uv run z_using_files/test/test_yfinance.py
    """
    xx = get_stock_data("KO")
    print(f"{len(xx)}")
    print(f"{xx}")
    df = get_stock_data("603678.SS")
    print("火炬电子 (603678.SS) 股息和拆股数据:")
    print(df)
