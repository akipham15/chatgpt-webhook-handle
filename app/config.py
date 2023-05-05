import os

from dotenv import load_dotenv

load_dotenv()


def get_config(key: str, default=None):
    value = os.environ.get(key, default)
    return value


class BaseConfig(object):
    CACHE_TYPE = get_config('CACHE_TYPE')
    CACHE_REDIS_HOST = get_config('CACHE_REDIS_HOST')
    CACHE_REDIS_PORT = get_config('CACHE_REDIS_PORT')
    CACHE_REDIS_INDEX = get_config('CACHE_REDIS_INDEX')
    CACHE_REDIS_DB = get_config('CACHE_REDIS_DB')
    CACHE_REDIS_URL = get_config('CACHE_REDIS_URL')
    CACHE_DEFAULT_TIMEOUT = get_config('CACHE_DEFAULT_TIMEOUT')

    SQLALCHEMY_DATABASE_URI = get_config('SQLALCHEMY_DATABASE_URI', 'sqlite:////tmp/itsupport.db')


class Config(BaseConfig):
    LIMIT_MESSAGE_TOKEN = 256
    LIMIT_MESSAGE_EXPIRE = 60 * 60 * 24
    LIMIT_MESSAGE_NUMBER = 10000

    DEFAULT_QUERY_DISTANCE = 0.29

    WELCOME_TRAIN_DATA_PATH = get_config('WELCOME_TRAIN_DATA_PATH', './data/train/welcome.csv')
    QA_TRAIN_DATA_PATH = get_config('QA_TRAIN_DATA_PATH', './data/train/FPTCOM_QA_0505_02.csv')
