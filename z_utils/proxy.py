import os
from functools import wraps


def set_proxy(ip: str = "127.0.0.1", port: int = 10808):
    """
    一个可以设置和清理代理的装饰器工厂函数。

    直接使用 @set_proxy() 来应用默认代理，
    或者使用 @set_proxy(ip="your_ip", port=your_port) 来指定自定义代理。

    Args:
        ip (str, optional): 代理服务器的 IP 地址。默认为 "127.0.0.1"
        port (int, optional): 代理服务器的端口号。默认为 10808
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            proxy_url = f"http://{ip}:{port}"
            original_http_proxy = os.environ.get("HTTP_PROXY")
            original_https_proxy = os.environ.get("HTTPS_PROXY")

            os.environ["HTTP_PROXY"] = proxy_url
            os.environ["HTTPS_PROXY"] = proxy_url

            try:
                result = func(*args, **kwargs)
                return result
            finally:

                if original_http_proxy:
                    os.environ["HTTP_PROXY"] = original_http_proxy
                else:
                    del os.environ["HTTP_PROXY"]
                if original_https_proxy:
                    os.environ["HTTPS_PROXY"] = original_https_proxy
                else:
                    del os.environ["HTTPS_PROXY"]

        return wrapper

    return decorator
