import os

import requests
from logzero import logger


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
