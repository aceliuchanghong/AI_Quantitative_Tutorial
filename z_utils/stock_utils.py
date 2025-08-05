import yfinance as yf
import akshare as ak
import sys
import os
import pandas as pd
from termcolor import colored
from datetime import datetime, timedelta

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")),
)

from z_utils.stock_sqlite import cache_to_sqlite
from z_utils.proxy import set_proxy


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


@cache_to_sqlite()
def get_a_share_stock_list():
    """
    获取当前A股所有股票的代码、名称和最新价

    Returns:
        pandas.DataFrame: 包含 '代码', '名称', '最新价' 列的 DataFrame
    """
    try:
        # 获取A股实时行情数据
        stock_df = ak.stock_zh_a_spot_em()
        # 选择需要的列
        result_df = stock_df[["代码", "名称", "最新价"]].copy()
        return result_df
    except Exception as e:
        print(f"获取数据时发生错误: {e}")
        return None


@cache_to_sqlite()
def get_trading_days(start_date, end_date):
    """获取交易日历"""
    calendar = ak.stock_zh_index_daily(symbol="sh000001")
    calendar["date"] = pd.to_datetime(calendar["date"])

    # 筛选指定日期范围
    mask = (calendar["date"] >= start_date) & (calendar["date"] <= end_date)
    trading_days = calendar.loc[mask, "date"]

    # 转换为 'YYYY-MM-DD' 字符串格式的列表
    return trading_days.dt.strftime("%Y-%m-%d").tolist()


def get_akshare_symbol(stock_code: str) -> str:
    """
    根据A股股票代码判断并返回akshare所需的symbol格式。
    上海证券交易所的股票代码以'6'开头，前面需要加上'sh'。
    深圳证券交易所的股票代码以'0'或'3'开头，前面需要加上'sz'。

    Args:
        stock_code (str): 6位数的A股股票代码。

    Returns:
        str: 适用于akshare的股票代码 (e.g., 'sh600519')。

    Raises:
        ValueError: 如果股票代码不是一个有效的6位数字字符串 或者不是以上海'6'或深圳'0'、'3'开头。
    """
    if (
        not isinstance(stock_code, str)
        or not stock_code.isdigit()
        or len(stock_code) != 6
    ):
        raise ValueError(f"无效的股票代码格式: '{stock_code}'。应为6位数字字符串。")

    if stock_code.startswith("6"):
        return f"sh{stock_code}"
    elif stock_code.startswith(("0", "3")):
        return f"sz{stock_code}"
    else:
        raise ValueError(
            f"无法识别的A股股票代码: '{stock_code}'。应以上海'6'或深圳'0'、'3'开头。"
        )


def get_yfinance_ticker(stock_code: str) -> str:
    """
    根据A股股票代码判断并返回yfinance所需的ticker格式。
    上海证券交易所的股票代码以'6'开头。
    深圳证券交易所的股票代码以'0'或'3'开头。

    Args:
        stock_code (str): 6位数的A股股票代码.

    Returns:
        str: 适用于yfinance的股票代码 (e.g., '600519.SS').

    Raises:
        ValueError: 如果股票代码不是以'6', '0', '3'开头，则抛出此异常。
    """
    if (
        not isinstance(stock_code, str)
        or not stock_code.isdigit()
        or len(stock_code) != 6
    ):
        raise ValueError(f"无效的股票代码格式: '{stock_code}'。应为6位数字字符串。")

    if stock_code.startswith("6"):
        return f"{stock_code}.SS"
    elif stock_code.startswith("0") or stock_code.startswith("3"):
        return f"{stock_code}.SZ"
    else:
        raise ValueError(
            f"无法识别的A股股票代码: '{stock_code}'。应以上海'6'或深圳'0'、'3'开头。"
        )


@cache_to_sqlite()
@set_proxy()
def get_daily_data_yfinance(
    stock_code: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    使用yfinance获取单只A股的日频行情数据 并格式化为指定样式。

    Args:
        stock_code (str): 6位数的A股股票代码.
        start_date (str): 开始日期, 格式 'YYYY-MM-DD' 或 'YYYYMMDD'.
        end_date (str): 结束日期, 格式 'YYYY-MM-DD' 或 'YYYYMMDD'.

    Returns:
        pd.DataFrame: 包含格式化后日频数据的DataFrame 如果获取失败则返回空的DataFrame.
    """
    try:
        # 1. 格式化yfinance的ticker
        ticker = get_yfinance_ticker(stock_code)

        # yfinance的end_date是“不包含”的，所以我们需要将其向后推一天
        start_date_fmt = pd.to_datetime(start_date).strftime("%Y-%m-%d")
        end_date_fmt = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime(
            "%Y-%m-%d"
        )

        # 从yfinance下载数据
        df = yf.download(ticker, start=start_date_fmt, end=end_date_fmt, progress=False)

        if df.empty:
            print(
                f"未能使用yfinance获取到股票 {stock_code} ({ticker}) 在指定日期范围内的数据。"
            )
            return pd.DataFrame()

        # 将'Date'从索引变成普通列
        df.reset_index(inplace=True)

        # 2. 标准化列名
        df.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            },
            inplace=True,
        )

        # 3. 【关键修改】对OHLC价格数据进行四舍五入，保留两位小数
        price_cols = ["open", "high", "low", "close"]
        df[price_cols] = df[price_cols].round(2)

        # 4. 添加原始的、不带后缀的股票代码列
        df["stock_code"] = stock_code

        # 5. 计算缺失的指标列
        # 计算前一天的收盘价，用于计算涨跌幅等指标
        prev_close = df["close"].shift(1)

        df["change"] = (df["close"] - prev_close).round(2)
        df["change_pct"] = ((df["close"] - prev_close) / prev_close * 100).round(2)
        df["amplitude"] = ((df["high"] - df["low"]) / prev_close * 100).round(2)
        df["amount"] = (df["close"] * df["volume"]).round(2)

        # 将第一行计算产生的NaN值替换为0
        df.fillna(0, inplace=True)

        # 6. 【关键修改】将日期列格式化为 'YYYY-MM-DD' 字符串
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")

        # 7. 调整列的顺序以匹配期望的输出
        output_columns = [
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
        ]

        # 确保所有期望的列都存在
        for col in output_columns:
            if col not in df.columns:
                df[col] = 0  # 或者其他默认值

        return df[output_columns]

    except Exception as e:
        print(f"处理股票 {stock_code} 数据时发生错误: {e}")
        return pd.DataFrame()


@cache_to_sqlite()
@set_proxy()
def get_intraday_data_for_date(
    stock_code: str, input_date: str, interval: str = "1m"
) -> pd.DataFrame:
    """
    使用yfinance获取单只A股在指定某一天的分钟级别行情数据 主要用于绘图。

    open	开盘价	在这一分钟的第一笔成交价格。
    close	收盘价	在这一分钟的最后一笔成交价格。这也是绘制分时走势线时通常使用的价格点
    volume	成交量	在这一分钟内，总共买卖的股票数量。单位是“股”
    amount	成交额	在这一分钟内，总共买卖的股票金额。单位是“元”。它约等于 价格 * 成交量。

    - yfinance对历史分钟数据的获取有严格的日期范围限制。
    - '1m'间隔的数据通常只能获取最近7天。
    - 其他分钟间隔(如'5m', '15m')通常只能获取最近60天。
    - 如果请求的日期超出限制或当天为非交易日 将返回空的DataFrame。

    Args:
        stock_code (str): 6位数字的A股股票代码。
        input_date (str): 希望获取数据的日期, 格式 'YYYY-MM-DD' 或 'YYYYMMDD'。
        interval (str): 数据间隔, e.g., '1m', '5m', '15m', '30m', '60m'。

    Returns:
        pd.DataFrame: 包含格式化后核心分钟数据的DataFrame 列包括：
                      ['datetime', 'stock_code', 'open', 'high', 'low', 'close', 'volume', 'amount']。
                      如果获取失败则返回空的DataFrame。
    """
    try:
        # 1. 验证和格式化日期
        request_dt = pd.to_datetime(input_date)
        # yfinance对分钟数据的限制是基于当前时间的，所以使用datetime.now()
        days_diff = (datetime.now(request_dt.tz) - request_dt).days

        # 2. 检查日期是否符合yfinance的限制
        if interval == "1m" and days_diff > 30:
            print(
                f"提醒: 分钟级别数据('1m')通常只能获取最近30天。请求的日期 {request_dt.strftime('%Y-%m-%d')} 可能超出范围。"
            )

        # 3. 准备API调用的开始和结束日期
        start_date_fmt = request_dt.strftime("%Y-%m-%d")
        end_date_fmt = (request_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        # 4. 获取适用于yfinance的ticker
        ticker = get_yfinance_ticker(stock_code)

        # 5. 从yfinance下载分钟级别数据
        df = yf.download(
            tickers=ticker,
            start=start_date_fmt,
            end=end_date_fmt,
            interval=interval,
            progress=False,
            auto_adjust=False,  # 保留 'Adj Close'
        )
        import time

        time.sleep(2)

        if df.empty:
            # 这个提示很正常，可能是非交易日
            # print(f"未获取到股票 {stock_code} ({ticker}) 在 {start_date_fmt} 的'{interval}'分钟数据。可能当天为非交易日或无数据。")
            return pd.DataFrame()

        # 6. 数据清洗和格式化 (核心逻辑修改点)
        df.reset_index(inplace=True)
        # yfinance返回的列名可能是 'Datetime' 或 'index'

        df.rename(
            columns={
                "Datetime": "datetime",
                "index": "datetime",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            },
            inplace=True,
        )

        # 仅保留绘图所需的核心列
        core_cols = ["datetime", "open", "high", "low", "close", "volume"]
        df = df[core_cols]

        # 7. 数据类型处理和计算成交金额
        price_cols = ["open", "high", "low", "close"]
        df[price_cols] = df[price_cols].round(2)  # A股价格通常保留两位小数
        df["volume"] = df["volume"].astype(int)

        # 计算估算成交金额 (Amount) = 收盘价 * 成交量
        # 这是分钟级别常用的估算方法，对于绘图和趋势分析足够精确
        df["amount"] = (df["close"] * df["volume"]).round(0).astype(int)

        # 8. 添加股票代码并格式化时间
        df["stock_code"] = stock_code
        # 将datetime对象转换为标准字符串格式，便于查看或存储
        df["datetime"] = (df["datetime"] + pd.Timedelta(hours=8)).dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # 9. 调整列的顺序并返回
        output_columns = [
            "datetime",
            "stock_code",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
        ]
        return df[output_columns]

    except ValueError as ve:
        # 捕获 get_yfinance_ticker 抛出的异常
        print(f"处理股票 {stock_code} 时发生错误: {ve}")
        return pd.DataFrame()
    except Exception as e:
        print(f"处理股票 {stock_code} 在 {input_date} 的数据时发生未知错误: {e}")
        return pd.DataFrame()


@cache_to_sqlite()
def get_today_stock_data(stock_code: str) -> pd.DataFrame | None:
    """
    获取指定A股股票【仅当天】从开盘到现在的1分钟级别交易数据 并按要求格式化。

    :param stock_code: 股票代码, 例如 '603678'
    :return: 包含当天分钟级别交易数据的 pandas.DataFrame 如果出错或无数据则返回 None
    """
    symbol = get_akshare_symbol(stock_code)

    try:
        trade_date_df = ak.tool_trade_date_hist_sina()
        is_trade_day = datetime.now().strftime("%Y-%m-%d") in [
            d.strftime("%Y-%m-%d") for d in trade_date_df["trade_date"]
        ]

        if not is_trade_day:
            print(f"获取数据失败：今天是【非交易日】。")
            return None

        # 获取1分钟级别的数据
        stock_minute_df = ak.stock_zh_a_minute(symbol=symbol, period="1", adjust="")

        if stock_minute_df.empty:
            print(
                f"未能获取到股票 {stock_code} 的数据。请检查代码是否正确或当前是否为开盘后的交易时间。"
            )
            return None

        columns_map = {
            "day": "datetime",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }
        stock_minute_df = stock_minute_df.rename(columns=columns_map)
        stock_minute_df["datetime"] = pd.to_datetime(stock_minute_df["datetime"])

        today_date = datetime.now().date()
        today_mask = stock_minute_df["datetime"].dt.date == today_date
        stock_minute_df = stock_minute_df[today_mask]

        if stock_minute_df.empty:
            print(f"在获取到的数据中，未能找到属于今天({today_date})的记录。")
            return None

        desired_columns = ["datetime", "open", "high", "low", "close", "volume"]
        stock_minute_df = stock_minute_df[desired_columns].copy()

        return stock_minute_df

    except Exception as e:
        print(f"获取股票 {stock_code} 数据时发生错误: {e}")
        return None


if __name__ == "__main__":
    """
    uv run z_utils/stock_utils.py
    """
    stock_code = "603678"
    input_date = "2025-07-02"
    start_data, end_data = "2025-07-22", "2025-07-23"
    df = get_daily_data_yfinance(stock_code, start_data, end_data)
    print(colored(f"{df}", "light_yellow"))
