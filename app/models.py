from mongoengine import Document, IntField, StringField

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