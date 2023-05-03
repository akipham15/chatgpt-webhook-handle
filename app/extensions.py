from celery import Celery
from celery import Task
from flask import Flask

from app.config import Config
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def celery_init_app(celery_app: Celery, flask_app: Flask) -> Celery:
    class FlaskTask(Task):
        def __call__(self, *args: object, **kwargs: object) -> object:
            with flask_app.app_context():
                return self.run(*args, **kwargs)

    # celery_app = Celery(flask_app.name, task_cls=FlaskTask)
    celery_app.name = flask_app.name
    celery_app.task_cls = FlaskTask
    flask_app.config.from_mapping(
        CELERY=dict(
            broker_url=Config.CACHE_REDIS_URL,
            result_backend=Config.CACHE_REDIS_URL,
            task_ignore_result=True,
        ),
    )
    flask_app.config.from_prefixed_env()
    celery_app.config_from_object(flask_app.config["CELERY"])

    celery_app.set_default()
    flask_app.extensions["celery"] = celery_app
    return celery_app


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)

    # connect db
    db.init_app(app)

    # return app
    return app


celery_app = Celery(__name__)
flask_app = create_app()
celery_app = celery_init_app(celery_app, flask_app)
