import pandas as pd
import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import pandas as pd
import os
import gc
from typing import Optional
import warnings

# 忽略常见无关警告
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


def plot_daily_stock_trend(df: pd.DataFrame, stock_code: str):
    """
    根据单日每分钟的股票交易数据绘制收盘价走势图，并保存为图片。

    文件名将根据当日涨跌情况自动命名为 'YYYY-MM-DD-涨.png' 或 'YYYY-MM-DD-跌.png'。
    图片将保存到 'no_git_oic/pics_out/{stock_code}/' 目录下。

    Args:
        df (pd.DataFrame): 包含 'datetime', 'open', 'close' 列的DataFrame。
        stock_code (str): 股票代码，用于创建保存目录。
    """
    # 1. 数据校验和预处理
    df = df.copy()

    if df.columns.astype(str)[0].strip().startswith("("):
        new_columns = [col.split(",")[0].strip("() '\"") for col in df.columns]
        df.columns = new_columns
    elif isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = df.columns.str.strip()

    if df.empty:
        print("错误 输入的DataFrame为空 无法绘图。")
        return

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # 2. 判断涨跌情况并获取日期
    first_open = df["open"].iloc[0]
    last_close = df["close"].iloc[-1]

    # 获取日期字符串，格式为 'YYYY-MM-DD'
    date_str = df["datetime"].iloc[0].strftime("%Y-%m-%d")

    # 根据A股习惯，红涨绿跌
    if last_close >= first_open:
        status_str = "涨"
        line_color = "red"
    else:
        status_str = "跌"
        line_color = "green"

    # 3. 开始绘图
    try:
        fig = Figure(figsize=(16, 9))
        canvas = FigureCanvasAgg(fig)  # 显式绑定渲染器
        ax = fig.add_subplot(111)

        # 绘制收盘价曲线
        ax.plot(df["datetime"], df["close"], color=line_color, label="Close")

        # 4. 美化图表
        # 设置标题，包含股票代码、日期和涨跌状态
        title = f"{stock_code} {date_str}"
        ax.set_title(title, fontsize=20, weight="bold")

        # 设置X轴和Y轴标签
        ax.set_xlabel("Time", fontsize=14)
        ax.set_ylabel("Price", fontsize=14)

        # 设置X轴刻度格式为 '小时:分钟'
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))

        fig.autofmt_xdate()  # 自动旋转日期标签

        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=12)

        # 紧凑布局
        fig.tight_layout(pad=2.0)

        # --- 5. 保存 ---
        output_dir = os.path.join("no_git_oic", "pics_out", stock_code)
        os.makedirs(output_dir, exist_ok=True)

        filename = f"{date_str}-{status_str}.png"
        file_path = os.path.join(output_dir, filename)

        fig.savefig(file_path, dpi=150, bbox_inches="tight")
        print(f"✅ 图片已保存: {file_path}")

        ax.clear()
        fig.clear()
        del fig, ax, canvas
        gc.collect()

        return file_path

    except Exception as e:
        print(
            f"❌ 绘图失败--股票 {stock_code}, 日期 {date_str if 'date_str' in locals() else 'unknown'}: {e}"
        )
        return None
