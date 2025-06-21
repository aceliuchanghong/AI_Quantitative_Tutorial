import sqlite3
from datetime import datetime, date
import hashlib
import json
import pandas as pd
import functools
from io import StringIO
from termcolor import colored
import traceback


def json_serializer(obj):
    """
    自定义JSON序列化器。
    当json.dumps遇到它不认识的类型时，会调用这个函数。
    这里我们主要处理日期、时间和pandas时间戳对象。
    """
    if isinstance(obj, (datetime, date, pd.Timestamp)):
        # 将日期/时间对象转换为ISO 8601格式的字符串，这是一种标准格式
        return obj.isoformat()
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


def cache_to_sqlite(db_name="stock_data.db", default_return=None, debug=True):
    """
    装饰器：自动为函数添加数据库缓存功能。
    支持自定义数据库名称和函数执行失败时的默认返回值。
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 自动获取被装饰的函数名
            function_name = func.__name__

            # 构造参数字典，将位置参数和关键字参数统一处理
            params = kwargs.copy()
            if args:
                # 为了更准确地将位置参数和其名称对应，我们获取函数的参数名
                arg_names = func.__code__.co_varnames[: len(args)]
                params.update(dict(zip(arg_names, args)))

            # 1. 初始化数据库连接
            try:
                conn = init_db(db_name)
            except Exception as e:
                print(colored(f"数据库初始化失败: {str(e)}", "red"))
                return default_return

            # 2. 查询缓存
            cached_data, data_count = query_cache(
                conn, function_name, params, debug=debug
            )
            if cached_data is not None:
                conn.close()
                return cached_data

            # 3. 如果缓存未命中，则执行原函数
            try:
                result = func(*args, **kwargs)
                # 如果函数返回空结果，则不缓存，直接返回
                if result is None or (
                    isinstance(result, (pd.DataFrame, list)) and len(result) == 0
                ):
                    conn.close()
                    print(
                        colored(
                            f"{function_name} 返回空结果，不进行缓存，参数: {params}",
                            "yellow",
                        )
                    )
                    return result

                # 4. 计算数据量并保存到缓存
                data_count = (
                    len(result) if isinstance(result, (pd.DataFrame, list)) else 1
                )
                save_to_cache(conn, function_name, params, result, data_count)
                print(
                    colored(f"{function_name} 执行成功，数据量: {data_count}", "green")
                )
                return result
            except Exception as e:
                # 如果函数执行出错，打印详细错误信息
                error_msg = (
                    f"函数 {function_name} 执行失败，参数: {params}\n"
                    f"错误详情:\n{traceback.format_exc()}"
                )
                print(colored(error_msg, "red"))
                return default_return
            finally:
                # 确保数据库连接总是被关闭
                if conn:
                    conn.close()

        return wrapper

    return decorator
