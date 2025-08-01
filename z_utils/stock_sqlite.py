import os
import sqlite3
from datetime import datetime, date
import hashlib
import json
import pandas as pd
import functools
from io import StringIO
from termcolor import colored
import traceback
import inspect
from pathlib import Path


def json_serializer(obj):
    if isinstance(obj, (datetime, date, pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def init_db(db_name="stock_data.db"):
    try:
        conn = sqlite3.connect(db_name)
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS stock_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                function_name TEXT NOT NULL,
                params TEXT NOT NULL,
                result_data TEXT NOT NULL,
                data_count INTEGER NOT NULL,
                data_type TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(function_name, params)
            )
            """
        )
        c.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_func_params 
            ON stock_cache (function_name, params)
            """
        )
        conn.commit()
        return conn
    except Exception as e:
        print(colored(f"数据库 {db_name} 初始化失败: {str(e)}", "red"))
        raise


def get_params_hash(params):
    params_str = json.dumps(params, sort_keys=True, default=json_serializer)
    return hashlib.md5(params_str.encode()).hexdigest()


def query_cache(conn, function_name, params, debug=True):
    try:
        params_hash = get_params_hash(params)
        c = conn.cursor()
        c.execute(
            """
            SELECT result_data, data_count, data_type FROM stock_cache
            WHERE function_name = ? AND params = ?
            """,
            (function_name, params_hash),
        )
        result = c.fetchone()
        if result:
            result_data, data_count, data_type = result
            if debug:
                print(colored(f"{function_name} 函数缓存命中，参数: {params}", "green"))
            if data_type == "dataframe":
                return pd.read_json(StringIO(result_data)), data_count
            else:
                return json.loads(result_data), data_count
        if debug:
            print(colored(f"{function_name} 函数缓存未命中，参数: {params}", "yellow"))
        return None, 0
    except Exception as e:
        print(colored(f"查询缓存失败: {str(e)}", "red"))
        raise


def save_to_cache(conn, function_name, params, result_data, data_count):
    try:
        params_hash = get_params_hash(params)
        c = conn.cursor()
        if isinstance(result_data, pd.DataFrame):
            data_type = "dataframe"
            serialized_data = result_data.to_json()
        else:
            data_type = "json"
            serialized_data = json.dumps(result_data, default=json_serializer)
        c.execute(
            """
            INSERT OR REPLACE INTO stock_cache (function_name, params, result_data, data_count, data_type, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                function_name,
                params_hash,
                serialized_data,
                data_count,
                data_type,
                datetime.now(),
            ),
        )
        conn.commit()
        print(colored(f"{function_name} 函数结果已保存到缓存，参数: {params}", "green"))
    except Exception as e:
        print(colored(f"保存缓存失败: {str(e)}", "red"))
        raise


def cache_to_sqlite(
    db_name="stock_data.db", default_return=None, debug=False, re_run=False
):
    """
    增强版缓存装饰器，支持三种方式控制是否重跑：
    1. 调用时传入 _re_run=True
    2. 设置环境变量 CACHE_RE_RUN=1 或 NO_CACHE=1
    3. 装饰器参数 re_run=True（默认行为）
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            function_name = func.__name__

            # 提取控制参数（不参与缓存哈希）
            force_rerun = kwargs.pop("_re_run", False)  # 用户调用时可传 _re_run=True

            # --- 解析参数（保持原逻辑）---
            params = kwargs.copy()
            sig = inspect.signature(func)
            func_param_names = list(sig.parameters.keys())

            args_to_process = list(args)
            if args_to_process and func_param_names:
                first_param_name = func_param_names[0]
                if first_param_name in ("self", "cls"):
                    args_to_process.pop(0)  # 移除 self/cls
                    data_arg_names = func_param_names[1 : len(args_to_process) + 1]
                else:
                    data_arg_names = func_param_names[: len(args_to_process)]
                params.update(dict(zip(data_arg_names, args_to_process)))
            # --- 参数解析结束 ---

            # ✅ 三层判断：是否跳过缓存？
            skip_cache = (
                force_rerun  # 1. 调用时指定 _re_run=True
                or os.getenv("CACHE_RE_RUN", "").lower() in ("1", "true", "on")
                or os.getenv("NO_CACHE", "").lower() in ("1", "true", "on")
                or re_run  # 3. 装饰器默认 re_run=True
            )

            conn = None
            try:
                conn = init_db(db_name)
            except Exception as e:
                print(colored(f"数据库初始化失败: {str(e)}", "red"))
                return default_return

            try:
                # 如果不跳过缓存，则尝试读取缓存
                if not skip_cache:
                    cached_data, data_count = query_cache(
                        conn, function_name, params, debug=debug
                    )
                    if cached_data is not None:
                        conn.close()
                        return cached_data
                elif debug:
                    print(
                        colored(
                            f"跳过缓存，强制重跑 {function_name} (原因: {_rerun_reason(force_rerun)})",
                            "blue",
                        )
                    )

                # 执行函数
                result = func(*args, **kwargs)

                # 空值不缓存
                is_result_empty = (
                    result is None
                    or (isinstance(result, pd.DataFrame) and result.empty)
                    or (isinstance(result, (list, dict)) and not result)
                )
                if is_result_empty:
                    if debug:
                        print(
                            colored(f"{function_name} 返回空结果，不缓存。", "yellow")
                        )
                    return result

                data_count = (
                    len(result) if isinstance(result, (pd.DataFrame, list, dict)) else 1
                )
                save_to_cache(conn, function_name, params, result, data_count)
                return result

            except Exception as e:
                error_msg = f"函数 {function_name} 执行失败，参数: {params}\n{traceback.format_exc()}"
                print(colored(error_msg, "red"))
                return default_return

            finally:
                if conn:
                    conn.close()

        return wrapper

    return decorator


def _rerun_reason(forced_by_arg):
    if forced_by_arg:
        return "传入 _re_run=True"
    elif os.getenv("CACHE_RE_RUN", "").lower() in ("1", "true"):
        return "环境变量 CACHE_RE_RUN=1"
    elif os.getenv("NO_CACHE", "").lower() in ("1", "true"):
        return "环境变量 NO_CACHE=1"
    else:
        return "装饰器 re_run=True"


def cache_to_sqlite_async(
    db_name="stock_data.db", default_return=None, debug=False, re_run=False
):
    """
    装饰器：为【异步】函数添加数据库缓存功能，支持三种方式控制是否重跑：
    1. 调用时传入 _re_run=True
    2. 设置环境变量 CACHE_RE_RUN=1 或 NO_CACHE=1
    3. 装饰器参数 re_run=True（默认行为）
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            function_name = func.__name__

            # 提取控制参数：是否强制重跑？（不参与缓存哈希）
            force_rerun = kwargs.pop("_re_run", False)  # 用户可传 _re_run=True

            # --- 解析参数（用于缓存键，不含 self/cls）---
            params = kwargs.copy()
            try:
                sig = inspect.signature(func)
                func_param_names = list(sig.parameters.keys())

                args_to_process = list(args)
                if args_to_process and func_param_names:
                    first_param_name = func_param_names[0]
                    if first_param_name in ("self", "cls"):
                        args_to_process.pop(0)  # 移除 self/cls
                        data_arg_names = func_param_names[1 : len(args_to_process) + 1]
                    else:
                        data_arg_names = func_param_names[: len(args_to_process)]
                    params.update(dict(zip(data_arg_names, args_to_process)))
            except Exception as e:
                if debug:
                    print(colored(f"参数解析失败: {str(e)}", "yellow"))
            # --- 参数解析结束 ---

            # ✅ 三层判断：是否跳过缓存？
            skip_cache = (
                force_rerun
                or os.getenv("CACHE_RE_RUN", "").lower() in ("1", "true", "on")
                or os.getenv("NO_CACHE", "").lower() in ("1", "true", "on")
                or re_run
            )

            conn = None
            try:
                conn = init_db(db_name)
            except Exception as e:
                print(colored(f"数据库初始化失败: {str(e)}", "red"))
                return default_return

            try:
                # 尝试从缓存读取（除非跳过缓存）
                if not skip_cache:
                    cached_data, data_count = query_cache(
                        conn, function_name, params, debug=debug
                    )
                    if cached_data is not None:
                        conn.close()
                        return cached_data
                elif debug:
                    reason = _rerun_reason(force_rerun)
                    print(
                        colored(
                            f"跳过缓存，强制重跑异步函数 {function_name} (原因: {reason})",
                            "blue",
                        )
                    )

                # 执行异步函数
                result = await func(*args, **kwargs)

                # 判断是否为空结果（不缓存）
                is_result_empty = (
                    result is None
                    or (isinstance(result, pd.DataFrame) and result.empty)
                    or (isinstance(result, (list, dict)) and not result)
                )
                if is_result_empty:
                    if debug:
                        print(
                            colored(f"{function_name} 返回空结果，不缓存。", "yellow")
                        )
                    return result

                # 计算数据量
                data_count = (
                    len(result) if isinstance(result, (pd.DataFrame, list, dict)) else 1
                )

                # 保存到缓存（使用剥离 self 后的 params）
                save_to_cache(conn, function_name, params, result, data_count)

                return result

            except Exception as e:
                error_msg = f"异步函数 {function_name} 执行失败，参数: {params}\n{traceback.format_exc()}"
                print(colored(error_msg, "red"))
                return default_return

            finally:
                if conn:
                    conn.close()

        return wrapper

    return decorator


if __name__ == "__main__":

    @cache_to_sqlite()
    def get_test_data(stock_code, start_date, end_date, adjust="qfq"):

        print(f"no cache")
        return {
            "stock_code": stock_code,
            "start_date": start_date,
            "end_date": end_date,
            "adjust": adjust,
        }

    """
    uv run z_utils/stock_sqlite.py
    """

    stock_code = "603678"
    start_data, end_data = "2025-07-23", "2025-07-23"
    adjust = "qfq"
    re_run = False

    import time

    start_time = time.time()

    df = get_test_data(
        stock_code,
        start_data.replace("-", ""),
        end_data.replace("-", ""),
        adjust,
        _re_run=re_run,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(colored(f"耗时: {elapsed_time:.2f}秒", "light_yellow"))
    print(colored(f"{df}", "light_yellow"))
