from datetime import datetime, timedelta
import sys
import os
import pandas as pd
from termcolor import colored


sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")),
)

from z_utils.stock_utils import (
    get_intraday_data_for_date,
    get_trading_days,
    get_today_stock_data,
)


if __name__ == "__main__":
    """
    uv run z_using_files/test/test_torch.py
    """
    start_date = "2025-07-02"
    end_date = datetime.today().strftime("%Y-%m-%d")
    stock_code = "603678"
    re_run = False

    date_list = get_trading_days(start_date, end_date)

    for today in date_list:
        print(colored(f"today:{today}", "light_yellow"))
        today_df = get_intraday_data_for_date(stock_code, today)
        print(colored(f"{type(today_df)}\n{today_df}", "light_yellow"))

    print(colored(f"date_list:{date_list}", "light_yellow"))

    now_df = get_today_stock_data(stock_code, _re_run=re_run)

    print(colored(f"\ntoday:{today}", "light_yellow"))
    print(colored(f"{type(now_df)}\n{now_df}", "light_yellow"))
