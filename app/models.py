from mongoengine import Document, IntField, StringField

from app.extensions import db


class Chat(Document):
    telegram_id = StringField(required=False)
    email = StringField(required=False)
    username = StringField(required=False)
    input = StringField(required=False)
    model = StringField(required=False)
    conversation_id = StringField(required=False)
    message_id = StringField(required=False)
    response = StringField(required=False)
    created = IntField(required=False)


class ChatGPTUser(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.Text, unique=True, nullable=False)
    token = db.Column(db.Text, unique=True, nullable=False)

    def __repr__(self):
        return '<User %r>' % self.email
