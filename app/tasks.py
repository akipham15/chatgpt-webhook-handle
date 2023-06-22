from datetime import datetime
from uuid import uuid4

from logzero import logger

from app import constants
from app.config import Config
from app.extensions import celery_app
from app.langchain_chatgpt import get_answer_with_documents
from app.libs.rate_limits import ChatLimit, rate_limit
from app.libs.utils import send_facebook_message, langchain_get_chat_from_user
from app.models import Chat


@celery_app.task(name="add_together")
def add_together(a: int, b: int) -> int:
    return a + b


@celery_app.task(bind=True, name="get_answer_from_chain", time_limit=360, max_retries=6, ignore_result=True)
@rate_limit(name='get_answer_from_chain', expire=60, limit=400)
def get_answer_from_chain(self, email: str, message: str, use_histories=False, history_length=3):
    log_content = {
        'action': 'webhook_answer',
        'email': email,
        'question': message,
        'answer': None
    }

    if not message in constants.EXCEPT_TEXT:
        histories = []
        if use_histories:
            histories = langchain_get_chat_from_user(email=email, num_of_history=history_length)
            logger.info(histories)

        answer, token_use = get_answer_with_documents(message, histories, email=email)
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

    response = send_facebook_message(email, answer)
    log_content['status_code'] = response.status_code
    log_content['answer'] = answer
    logger.info(log_content)

    return answer
