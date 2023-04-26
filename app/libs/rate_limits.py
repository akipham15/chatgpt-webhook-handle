from functools import wraps

from logzero import logger
from redis import ConnectionPool, Redis

from app.config import Config

REDIS_POOL = ConnectionPool(
    host=Config.CACHE_REDIS_HOST,
    port=Config.CACHE_REDIS_PORT,
    db=Config.CACHE_REDIS_INDEX,
)


class ChatLimit(object):
    def __init__(self, name, expire=3, limit=0, client='WORKPLACE', extra_log: dict = None):
        self.name = name
        self.extra_log = extra_log
        self._rate_limit_key = "rate_limit:{0}_{1}".format(name, client)
        self.expire = expire
        self.limit = limit
        self._redis = Redis(connection_pool=REDIS_POOL)

    def get_usage(self) -> int:
        value = self._redis.get(self._rate_limit_key) or 0
        return int(value)

    def increment_usage(self, increment_by=1):
        value = self.get_usage() + increment_by
        self._redis.set(self._rate_limit_key, value, self.expire)

    def has_been_reached(self, limit: int = None) -> bool:
        if limit is None:
            limit = self.limit
        if not limit:
            return False
        else:
            is_reach = self.get_usage() >= limit
            if is_reach and self.extra_log:
                log_content = {
                    'action': 'function_reach_limit',
                    'status': False,
                    'detail': '{} has been reach limit: {}'.format(self.name, limit),
                    'wait_time': self.get_wait_time(),
                    **self.extra_log
                }
                logger.warning(log_content)
            return is_reach

    def get_wait_time(self) -> float:
        expire = self._redis.pttl(self._rate_limit_key)
        # Fallback if key has not yet been set or TTL can't be retrieved
        expire = expire / 1000.0 if expire > 0 else float(self.expire)
        if self.has_been_reached():
            return expire
        else:
            return expire / (self.limit - self.get_usage())


def rate_limit(name: str, expire: int = 3, limit: int = None, extra_log: dict = None):
    def decorator_func(func):
        @wraps(func)
        def function(self, *args, **kwargs):
            chat_limit = ChatLimit(name=name, expire=expire, limit=limit, extra_log=extra_log)
            if chat_limit.has_been_reached():
                logger.warning(f'{func.__name__} reach limit. Release after {chat_limit.get_wait_time()}')
                pass
            else:
                chat_limit.increment_usage()
                return func(self, *args, **kwargs)

        return function

    return decorator_func
