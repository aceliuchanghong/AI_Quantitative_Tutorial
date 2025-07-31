from datetime import datetime, timedelta
import sys
import os
import pandas as pd
from termcolor import colored


sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")),
)

from z_utils.stock_utils import get_intraday_data_for_date, get_trading_days


if __name__ == "__main__":
    """
    uv run z_using_files/test/test_torch.py
    """
    start_date = "2025-07-02"
    end_date = datetime.today().strftime("%Y-%m-%d")
    stock_code = "603678"

    date_list = get_trading_days(start_date, end_date)

    for today in date_list:
        print(colored(f"today:{today}", "light_yellow"))
        today_df = get_intraday_data_for_date(stock_code, today)
        print(colored(f"{today_df}", "light_yellow"))
