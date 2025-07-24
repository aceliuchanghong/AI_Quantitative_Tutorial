import sys
import os
from termcolor import colored

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")),
)
from z_using_files.backtrader_code.zz500_data import get_daily_data


if __name__ == "__main__":
    """
    uv run z_using_files/backtrader_code/strategy_04.py
    """
    stock_code = "603678"  # 火炬电子
    start_data, end_data = "2025-07-23", "2025-07-23"
    adjust = "qfq"
    df = get_daily_data(
        stock_code, start_data.replace("-", ""), end_data.replace("-", ""), adjust
    )
    print(colored(f"{df}", "light_yellow"))
