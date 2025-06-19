import sqlite3
from datetime import datetime
import hashlib
import json
import pandas as pd
import functools
from io import StringIO
from termcolor import colored
import traceback


# 推荐 vscode-sqlite 插件
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
        # print(colored(f"数据库 {db_name} 初始化成功", "green"))
        return conn
    except Exception as e:
        print(colored(f"数据库 {db_name} 初始化失败: {str(e)}", "red"))
        raise


def get_params_hash(params):
    """生成参数的唯一标识"""
    params_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(params_str.encode()).hexdigest()


def query_cache(conn, function_name, params):
    """查询缓存"""
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
            print(colored(f"{function_name} 函数缓存命中，参数: {params}", "green"))
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
    """保存到缓存"""
    try:
        params_hash = get_params_hash(params)
        c = conn.cursor()
        if isinstance(result_data, pd.DataFrame):
            data_type = "dataframe"
            serialized_data = result_data.to_json()
        else:
            data_type = "json"
            serialized_data = json.dumps(result_data)
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


def cache_to_sqlite(db_name="stock_data.db", default_return=None):
    """装饰器：自动为函数添加数据库缓存功能，支持自定义默认返回值"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 自动获取函数名
            function_name = func.__name__
            # 构造参数字典
            params = kwargs.copy()
            if args:
                params.update({f"arg{i}": arg for i, arg in enumerate(args)})

            # 初始化数据库连接
            try:
                conn = init_db(db_name)
            except Exception as e:
                print(colored(f"数据库初始化失败: {str(e)}", "red"))
                return default_return

            # 查询缓存
            cached_data, data_count = query_cache(conn, function_name, params)
            if cached_data is not None:
                conn.close()
                return cached_data

            # 执行原函数
            try:
                result = func(*args, **kwargs)
                if result is None or (
                    isinstance(result, (pd.DataFrame, list)) and len(result) == 0
                ):
                    conn.close()
                    print(
                        colored(f"{function_name} 返回空结果，参数: {params}", "yellow")
                    )
                    return result

                # 保存到缓存
                data_count = (
                    len(result) if isinstance(result, (pd.DataFrame, list)) else 0
                )
                save_to_cache(conn, function_name, params, result, data_count)
                conn.close()
                print(
                    colored(f"{function_name} 执行成功，数据量: {data_count}", "green")
                )
                return result
            except Exception as e:
                error_msg = (
                    f"函数 {function_name} 执行失败，参数: {params}\n"
                    f"错误详情:\n{traceback.format_exc()}"
                )
                print(colored(error_msg, "red"))
                conn.close()
                return default_return

        return wrapper

    return decorator
