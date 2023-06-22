import os

import requests
from logzero import logger

from app import constants


def send_facebook_message(email, message_text):
    url = "https://alerts.soc.fpt.net/webhooks/{}/facebook_no_register".format(
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


def get_chat_history(inputs) -> str:
    res = []
    for human, ai in inputs:
        res.append(f"Human:{human}")
        res.append(f"Chatbot:{ai}")
    return "\n".join(res)


def langchain_get_chat_from_user(email=None, num_of_history: int = 2, use_object_format=False):
    from app.models import Chat
    user_chats = Chat.objects(email=str(email)).order_by('-created')[:num_of_history]
    chats = []

    if user_chats:
        for doc in user_chats:
            if doc.input in constants.EXCEPT_TEXT:
                break
            else:
                human = doc.input
                ai = doc.response
                if use_object_format:
                    # chats.append({"role": "assistant", "content": ai})
                    chats.append({"role": "user", "content": human})
                else:
                    chats.append(f"Chatbot: {ai}")
                    chats.append(f"Human: {human}")

    return chats[::-1]
