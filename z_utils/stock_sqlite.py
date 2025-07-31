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
    """
    自定义JSON序列化器
    """
    # 优先处理日期和时间戳
    if isinstance(obj, (datetime, date, pd.Timestamp)):
        return obj.isoformat()
    # 如果对象是Path类型，则将其转换为字符串
    if isinstance(obj, Path):
        return str(obj)
    # 如果遇到其他无法处理的类型，则抛出原始的TypeError
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# 推荐使用 vscode-sqlite 插件来查看数据库文件
def init_db(db_name="stock_data.db"):
    """初始化数据库并创建缓存表"""
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
    """为参数字典生成唯一的MD5哈希值，作为缓存的键"""
    # 使用 json.dumps 将参数字典转换为字符串
    # 关键修复：传入 default=json_serializer 参数，让其能够正确处理datetime/Timestamp等对象
    params_str = json.dumps(params, sort_keys=True, default=json_serializer)
    return hashlib.md5(params_str.encode()).hexdigest()


def query_cache(conn, function_name, params, debug=True):
    """根据函数名和参数哈希查询缓存"""
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
            # 根据存储时的数据类型，反序列化数据
            if data_type == "dataframe":
                return pd.read_json(StringIO(result_data)), data_count
            else:
                return json.loads(result_data), data_count
        print(colored(f"{function_name} 函数缓存未命中，参数: {params}", "yellow"))
        return None, 0
    except Exception as e:
        print(colored(f"查询缓存失败: {str(e)}", "red"))
        raise


def save_to_cache(conn, function_name, params, result_data, data_count):
    """将函数的执行结果保存到数据库缓存中"""
    try:
        params_hash = get_params_hash(params)
        c = conn.cursor()
        # 判断结果是DataFrame还是其他JSON兼容类型
        if isinstance(result_data, pd.DataFrame):
            data_type = "dataframe"
            serialized_data = result_data.to_json()
        else:
            data_type = "json"
            # 关键修复：同样使用自定义序列化器，处理返回值中可能存在的datetime/Timestamp对象
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
    装饰器：自动为【同步】函数添加数据库缓存功能。
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            function_name = func.__name__

            # --- V V V --- START OF THE ROBUST FIX --- V V V ---
            # 关键修复：健壮地分离 self/cls 参数和真正的缓存参数

            params = kwargs.copy()

            # 使用 inspect 获取函数的参数签名
            sig = inspect.signature(func)
            func_param_names = list(sig.parameters.keys())

            # 定位哪些是位置参数
            args_to_process = list(args)

            # 如果 args 存在，且第一个参数是 self 或 cls，则将其移除
            if args_to_process:
                first_param_name = func_param_names[0] if func_param_names else ""
                # 检查是否是实例方法或类方法
                if first_param_name in ("self", "cls"):
                    # 这是实例/类方法，第一个 arg 是 self/cls 对象，必须忽略
                    instance_or_class = args_to_process.pop(0)  # 移除 self/cls 对象
                    data_arg_names = func_param_names[
                        1 : len(args_to_process) + 1
                    ]  # 获取对应的数据参数名
                else:
                    # 这是普通函数或静态方法
                    data_arg_names = func_param_names[: len(args_to_process)]

                # 将剩余的位置参数与它们的名称配对
                params.update(dict(zip(data_arg_names, args_to_process)))
            # --- ^ ^ ^ --- END OF THE ROBUST FIX --- ^ ^ ^ ---

            conn = None  # 在 try 外部初始化 conn
            try:
                conn = init_db(db_name)
            except Exception as e:
                print(colored(f"数据库初始化失败: {str(e)}", "red"))
                return default_return

            try:
                if not re_run:
                    # 现在传递给 query_cache 的 params 绝对不包含 self/cls 对象
                    cached_data, data_count = query_cache(
                        conn, function_name, params, debug=debug
                    )
                    if cached_data is not None:
                        conn.close()
                        return cached_data
                elif debug:
                    print(
                        colored(
                            f"装饰器设置了 re_run=True, 强制重新运行 {function_name}",
                            "blue",
                        )
                    )

                # 如果缓存未命中或 re_run=True，则执行原函数
                # 执行时必须使用完整的 *args，因为它包含 self 对象
                result = func(*args, **kwargs)

                # 检查空结果的逻辑可以稍微优化，同时处理 dict
                is_result_empty = False
                if result is None:
                    is_result_empty = True
                elif isinstance(result, pd.DataFrame):
                    is_result_empty = result.empty  # 对DataFrame使用.empty属性
                elif isinstance(result, (list, dict)):
                    is_result_empty = not result  # 对list和dict，'not'可以正常工作

                if is_result_empty:
                    if debug:
                        print(
                            colored(
                                f"{function_name} 返回空结果，不进行缓存。",
                                "yellow",
                            )
                        )
                    return result

                # 计算数据量并保存/更新到缓存
                data_count = (
                    len(result) if isinstance(result, (pd.DataFrame, list, dict)) else 1
                )

                # 保存缓存时，使用不含 self 的 params
                save_to_cache(conn, function_name, params, result, data_count)

                if debug:
                    print(
                        colored(
                            f"{function_name} 执行成功，数据量: {data_count}", "green"
                        )
                    )
                return result
            except Exception as e:
                # 打印错误时，使用清理过的 params，避免日志过长或再次序列化失败
                error_msg = (
                    f"函数 {function_name} 执行失败，参数: {params}\n"
                    f"错误详情:\n{traceback.format_exc()}"
                )
                print(colored(error_msg, "red"))
                return default_return
            finally:
                if conn:
                    conn.close()

        return wrapper

    return decorator


def cache_to_sqlite_async(
    db_name="stock_data.db", default_return=None, debug=False, re_run=False
):
    """
    装饰器：专门为【异步】函数添加数据库缓存功能。
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            function_name = func.__name__

            # --- V V V --- START OF THE ROBUST FIX --- V V V ---
            # 关键修复：健壮地分离 self/cls 参数和真正的缓存参数

            params = kwargs.copy()

            # 使用 inspect 获取函数的参数签名
            sig = inspect.signature(func)
            func_param_names = list(sig.parameters.keys())

            # 定位哪些是位置参数
            args_to_process = list(args)

            # 如果 args 存在，且第一个参数是 self 或 cls，则将其移除
            if args_to_process:
                first_param_name = func_param_names[0] if func_param_names else ""
                # 检查是否是实例方法或类方法
                if first_param_name in ("self", "cls"):
                    # 这是实例/类方法，第一个 arg 是 self/cls 对象，必须忽略
                    instance_or_class = args_to_process.pop(0)  # 移除 self/cls 对象
                    data_arg_names = func_param_names[
                        1 : len(args_to_process) + 1
                    ]  # 获取对应的数据参数名
                else:
                    # 这是普通函数或静态方法
                    data_arg_names = func_param_names[: len(args_to_process)]

                # 将剩余的位置参数与它们的名称配对
                params.update(dict(zip(data_arg_names, args_to_process)))

            # --- ^ ^ ^ --- END OF THE ROBUST FIX --- ^ ^ ^ ---

            conn = None
            try:
                conn = init_db(db_name)
            except Exception as e:
                print(colored(f"数据库初始化失败: {str(e)}", "red"))
                return default_return

            try:
                if not re_run:
                    # 现在传递给 query_cache 的 params 绝对不包含 self/cls 对象
                    cached_data, data_count = query_cache(
                        conn, function_name, params, debug=debug
                    )
                    if cached_data is not None:
                        conn.close()
                        return cached_data
                elif debug:
                    print(
                        colored(
                            f"装饰器设置了 re_run=True, 强制重新运行 {function_name}",
                            "blue",
                        )
                    )

                # 执行原始函数时，必须使用完整的原始 *args，因为它包含 self
                result = await func(*args, **kwargs)

                is_result_empty = False
                if result is None:
                    is_result_empty = True
                elif isinstance(result, pd.DataFrame):
                    is_result_empty = result.empty  # 对DataFrame使用.empty属性
                elif isinstance(result, (list, dict)):
                    is_result_empty = not result  # 对list和dict，'not'可以正常工作

                if is_result_empty:
                    if debug:
                        print(
                            colored(
                                f"{function_name} 返回空结果，不进行缓存。",
                                "yellow",
                            )
                        )
                    return result

                data_count = (
                    len(result) if isinstance(result, (pd.DataFrame, list)) else 1
                )

                # 保存缓存时，使用不含 self 的 params
                save_to_cache(conn, function_name, params, result, data_count)

                return result
            except Exception as e:
                error_msg = (
                    f"异步函数 {function_name} 执行失败，参数: {params}\n"
                    f"错误详情:\n{traceback.format_exc()}"
                )
                print(colored(error_msg, "red"))
                return default_return
            finally:
                if conn:
                    conn.close()

        return wrapper

    return decorator


if __name__ == "__main__":

    @cache_to_sqlite(debug=False)
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

    import time

    start_time = time.time()

    df = get_test_data(
        stock_code, start_data.replace("-", ""), end_data.replace("-", ""), adjust
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(colored(f"耗时: {elapsed_time:.2f}秒", "light_yellow"))
    print(colored(f"{df}", "light_yellow"))
