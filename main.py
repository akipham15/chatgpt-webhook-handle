import os
from datetime import datetime
from uuid import uuid4

import requests
from dotenv import load_dotenv
from flask import Flask
from flask import request, jsonify
from logzero import logger
from mongoengine import (Document, IntField, StringField, connect)

from chatgpt import get_answer_old, get_answer
from langchain_chatgpt import get_answer

load_dotenv()
EXCEPT_TEXT = ['/new']
WEBHOOK_TOKEN = os.getenv('WEBHOOK_TOKEN')

# connect mongodb
connect(db='wpbotchatgpt', host='localhost', port=27017)

app = Flask(__name__)

logger.info(WEBHOOK_TOKEN)


@app.route("/webhook/handle/{}".format(WEBHOOK_TOKEN), methods=['GET', 'POST'])
def webhook_handle():
    if request.method == 'POST':
        logger.info('POST request')
        data = request.get_json()
        logger.info(data)
        if data:
            email = data.get('user_id')
            message = data.get('text')
            # answer = handle_text_and_get_answer(email, message)
            answer = get_answer_from_chain(message)
            logger.info(answer)

            response = send_facebook_message(email, answer)
            if response.status_code == 200:
                return jsonify({'text': ''})
            else:
                return jsonify({'text': answer})

    return jsonify({'text': 'hmm...?!'})


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


def get_chat_from_user(email=None):
    num_of_history = 2
    user_chats = Chat.objects(email=str(email)).order_by('-created')[:num_of_history]
    chats = []
    context = None

    if user_chats:
        for doc in user_chats:
            if doc.input in EXCEPT_TEXT:
                break
            else:
                if not context and doc.conversation_id:
                    context = doc.conversation_id
                if doc.response:
                    chats.append({"role": "assistant", "content": doc.response})
                if doc.input:
                    chats.append({"role": "user", "content": doc.input})

    return chats[::-1], context


def send_facebook_message(email, message_text):
    url = "https://alerts.soc.fpt.net/webhooks/{}/facebook".format(os.environ.get('IRIS_TOKEN'))

    payload = {
        "text": message_text,
        "email": email
    }
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, json=payload)
    logger.info(response.status_code)
    return response


def get_answer_from_chain(message: str):
    answer = get_answer(message)
    return answer


def handle_text_and_get_answer(email, message_text):
    # query
    conversations, context = get_chat_from_user(email)
    if context:
        conversations.insert(0, {"role": "system", "content": context})
    conversations.append({"role": "user", "content": message_text})

    if not message_text in EXCEPT_TEXT:
        # completion = None
        completion = get_answer(conversations, 'gpt-3.5-turbo')
        if not completion:
            completion = get_answer_old(conversations, 'text-davinci-003')
        logger.info(completion)

        # get answer from chatgpt
        if completion:
            if completion.get('choices'):
                first_choice = completion.get('choices')[0]
                if first_choice.get('message'):
                    answer = first_choice.get('message', {}).get('content')
                else:
                    answer = first_choice.get('text', '')

                    pep_keywords = ['system:', 'assistant:']

                    for keyword in pep_keywords:
                        if keyword in answer:
                            answer = answer.replace(keyword, '')

                chat_result = Chat(
                    email=str(email),
                    input=str(message_text),
                    model=completion.get('model'),
                    conversation_id=context if context else str(uuid4()),
                    message_id=completion.get('id'),
                    response=answer,
                    created=datetime.utcnow().timestamp()
                )
                chat_result.save()
            else:
                answer = 'Lỗi gì ấy nhỉ ?.. À thì ra là không có câu trả lời!'
        else:
            answer = 'Thử lại nhé, bot lại sập rồi :('
    else:
        chat_result = Chat(
            email=str(email),
            input=str(message_text),
            model='',
            message_id='',
            response='',
            created=datetime.utcnow().timestamp()
        )
        chat_result.save()
        answer = 'Hội thoại mới đã sẵn sàng :)'

    return answer


if __name__ == "__main__":
    app.run(host='127.0.0.1')
