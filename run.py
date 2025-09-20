import click
import os
import asyncio
from termcolor import colored

from core.pipeline import Pipeline
from z_utils.stock_utils import (
    get_intraday_data_for_date,
    get_today_stock_data,
    get_trading_days,
)


@click.group()
def cli():
    pass


@cli.command()
@click.option("--stock-code", default="603678", help="A股股票代码")
@click.option("--start-date", default="2025-07-01", help="绘制起始日期")
@click.option("--end-date", default="2025-08-01", help="绘制截至日期")
@click.option("--today", is_flag=True, default=False, help="绘制今日")
@click.option("--rerun", is_flag=True, default=False, help="今日数据重新获取")
def draw_pics(
    stock_code: str, start_date: str, end_date: str, today: bool, rerun: bool
):
    """绘制从起始日期到截至日期每一日的分钟级别交易图片"""
    """
    uv run run.py draw-pics --stock-code 603678 --start-date 2025-07-30 --end-date 2025-08-11
    uv run run.py draw-pics --stock-code 603678 --today --rerun
    
    uv run run.py draw-pics --stock-code 600258 --start-date 2025-07-07 --end-date 2025-08-05
    uv run run.py draw-pics --stock-code 600258 --today
    """
    pipeline = Pipeline()
    trade_df = []

    if not today:
        date_list = get_trading_days(start_date, end_date)
        for today in date_list:
            today_df = get_intraday_data_for_date(stock_code, today, _re_run=rerun)
            if len(today_df) > 0:
                trade_df.append(today_df)
    else:
        today_df = get_today_stock_data(stock_code, _re_run=rerun)
        trade_df.append(today_df)

    print(colored(f"{trade_df}", "light_yellow"))

    output_paths = asyncio.run(pipeline.draw_all_pics_async(trade_df, stock_code))
    print(colored("\n所有图片绘制完成:", "green"))
    for path in output_paths:
        if path:
            print(f" - {path}")

    return output_paths


if __name__ == "__main__":
    """
    uv run run.py draw-pics --stock-code 603678 --start-date 2025-07-30 --end-date 2025-07-31
    uv run run.py draw-pics --stock-code 603678 --today --rerun
    """
    cli()
