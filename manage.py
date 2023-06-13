import os
import shutil

from flask.cli import FlaskGroup
from logzero import logger

from app import constants
from app.config import Config
from app.langchain_chatgpt import create_persist_directory
from app.main import app

manager = FlaskGroup(app)


@manager.command('retrain')
def retrain():
    train_path = Config.QA_TRAIN_DATA_PATH
    persist_name = constants.PERSIST_DIRECTORY_FPT_EXCHANGE
    fieldnames = ['question', 'answer']

    persist_directory_existed = os.path.exists(persist_name)
    logger.info(persist_directory_existed)
    if persist_directory_existed:
        logger.info(f'remove persist_directory {persist_name}')
        shutil.rmtree(persist_name)
    else:
        logger.info(f'create persist_directory {persist_name}')
        create_persist_directory(train_path, persist_name, fieldnames)


if __name__ == "__main__":
    manager()
