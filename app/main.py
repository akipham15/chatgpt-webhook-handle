import os
from datetime import datetime
from uuid import uuid4

import requests
from dotenv import load_dotenv
from flask import Flask
from flask import request, jsonify
from logzero import logger
from mongoengine import connect

from app import constants
from app.chatgpt import get_answer_old, get_answer
from app.config import Config
from app.langchain_chatgpt import get_answer_with_documents, get_default_answer
from app.libs.rate_limits import ChatLimit
from app.models import Chat

load_dotenv()
EXCEPT_TEXT = ['/new']
WEBHOOK_TOKEN = os.getenv('WEBHOOK_TOKEN')

# connect mongodb
connect(db='wpbotchatgpt', host='localhost', port=27017)

app = Flask(__name__)
app.config.from_object(Config)

@app.route("/healthz", methods=['GET'])
def healthz():
    return 'ok'

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
            chat_limit = ChatLimit(name=email, expire=Config.LIMIT_MESSAGE_EXPIRE, limit=Config.LIMIT_MESSAGE_NUMBER)
            if chat_limit.has_been_reached():
                answer = get_default_answer(constants.LIMITED_ANSWERS)
                logger.warning(f'user limited, usage: {chat_limit.get_usage()}')
            else:
                answer = get_answer_from_chain(email, message, use_histories=False)

            log_content = {
                'action': 'webhook_handle',
                'email': email,
                'question': message,
                'answer': answer
            }
            logger.info(log_content)

            response = send_facebook_message(email, answer)
            if response.status_code == 200:
                return jsonify({'text': ''})
            else:
                return jsonify({'text': answer})

    return jsonify({'text': 'hmm...?!'})


def langchain_get_chat_from_user(email=None, num_of_history: int = 2):
    user_chats = Chat.objects(email=str(email)).order_by(
        '-created')[:num_of_history]
    chats = []

    if user_chats:
        for doc in user_chats:
            if doc.input in EXCEPT_TEXT:
                break
            else:
                history = (doc.input, doc.response)
                chats.append(history)

    return chats[::-1]


def get_chat_from_user(email=None, num_of_history: int = 2):
    user_chats = Chat.objects(email=str(email)).order_by(
        '-created')[:num_of_history]
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
                    chats.append(
                        {"role": "assistant", "content": doc.response})
                if doc.input:
                    chats.append({"role": "user", "content": doc.input})

    return chats[::-1], context


def send_facebook_message(email, message_text):
    url = "https://alerts.soc.fpt.net/webhooks/{}/facebook".format(
        os.environ.get('IRIS_TOKEN'))

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


def get_answer_from_chain(email: str, message: str, use_histories=False):
    if not message in EXCEPT_TEXT:
        histories = []
        if use_histories:
            histories = langchain_get_chat_from_user(
                email=email, num_of_history=1)
            logger.info(histories)

        answer, token_use = get_answer_with_documents(message, histories)
        chat_result = Chat(
            email=str(email),
            input=str(message),
            model='text-davinci-003',
            conversation_id=str(uuid4()),
            response=answer,
            created=datetime.utcnow().timestamp()
        )
        chat_result.save()
        # update user limit
        if answer not in (constants.DEFAULT_ANSWERS + constants.MESSAGE_TOO_LONG):
            chat_limit = ChatLimit(name=email, expire=Config.LIMIT_MESSAGE_EXPIRE, limit=Config.LIMIT_MESSAGE_NUMBER)
            chat_limit.increment_usage(increment_by=token_use)
    else:
        chat_result = Chat(
            email=str(email),
            input=str(message),
            model='',
            message_id='',
            response='',
            created=datetime.utcnow().timestamp()
        )
        chat_result.save()
        answer = 'Hội thoại mới đã sẵn sàng :)'

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
