import sys
import os
from termcolor import colored
import asyncio

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")),
)

from core.deal_pics import plot_daily_stock_trend


class Pipeline:
    def __init__(self, concurrency_limit=4):
        self.semaphore = asyncio.Semaphore(concurrency_limit)

    async def _draw_one_pic_wrapper(self, data_df, stock_code):
        """
        控制单个绘图任务的并发
        """
        async with self.semaphore:
            out_path = await asyncio.to_thread(
                plot_daily_stock_trend, data_df, stock_code
            )
            return out_path

    async def draw_all_pics_async(self, data_list, stock_code):
        """
        异步地、并发地为所有数据生成图片。
        """
        tasks = [
            self._draw_one_pic_wrapper(data_df, stock_code) for data_df in data_list
        ]

        print(colored(f"\n即将开始处理 {len(tasks)} 个绘图任务...", "cyan"))
        out_path_list = await asyncio.gather(*tasks)

        return out_path_list
