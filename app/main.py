import os
from datetime import datetime
from uuid import uuid4

from dotenv import load_dotenv
from flask import request, jsonify
from logzero import logger
from mongoengine import connect

from app import constants
from app.chatgpt import get_answer_old, get_answer
from app.config import Config
from app.extensions import create_app
from app.langchain_chatgpt import get_default_answer, get_answer_with_documents
from app.libs.rate_limits import ChatLimit
from app.models import Chat
from app.tasks import get_answer_from_chain

load_dotenv()
WEBHOOK_TOKEN = os.getenv('WEBHOOK_TOKEN')

# connect mongodb
connect(db='wpbotchatgpt', host='localhost', port=27017)

app = create_app()
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
            email = data.get('workchat_email', data.get('user_id'))
            message = data.get('text', data.get('message'))
            if not message:
                return jsonify({'text': ''})

            log_content = {
                'action': 'webhook_handle',
                'email': email,

            }
            logger.info(log_content)
            # answer = handle_text_and_get_answer(email, message)
            chat_limit = ChatLimit(name=email, expire=Config.LIMIT_MESSAGE_EXPIRE, limit=Config.LIMIT_MESSAGE_NUMBER)
            if chat_limit.has_been_reached():
                answer = get_default_answer(constants.LIMITED_ANSWERS)
                logger.warning(f'user limited, usage: {chat_limit.get_usage()}')
                return jsonify({'text': answer})
            else:
                logger.info('get answer from queue')
                get_answer_from_chain.delay(email, message, use_histories=True, history_length=3)
                # answer = get_answer_from_chain(email, message, use_histories=False)
                # if answer:
                #     return jsonify({'text': answer})

    return jsonify({'text': ''})


def get_chat_from_user(email=None, num_of_history: int = 2):
    user_chats = Chat.objects(email=str(email)).order_by(
        '-created')[:num_of_history]
    chats = []
    context = None

    if user_chats:
        for doc in user_chats:
            if doc.input in constants.EXCEPT_TEXT:
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


def handle_text_and_get_answer(email, message_text):
    # query
    conversations, context = get_chat_from_user(email)
    if context:
        conversations.insert(0, {"role": "system", "content": context})
    conversations.append({"role": "user", "content": message_text})

    if not message_text in constants.EXCEPT_TEXT:
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
    get_answer_with_documents('hello', [])
    # app.run(host='127.0.0.1')
