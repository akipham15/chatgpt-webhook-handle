from mongoengine import connect

from app.extensions import celery_app

connect(db='wpbotchatgpt', host='localhost', port=27017)
celery_app.autodiscover_tasks(['app.tasks'], force=True)
