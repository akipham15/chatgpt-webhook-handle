import os
import time

import openai
import requests
from dotenv import load_dotenv
from logzero import logger

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY


# list engines
# engines = openai.Engine.list()
# for engine in engines.data:
#     print(engine.id)

def get_answer_old(conversations: list, engine_id='text-davinci-003', retry=0):
    logger.info(conversations)

    prompts = []
    for chat in conversations:
        prompts.append('{role}:{content}'.format(
            role=chat.get('role'), content=chat.get('content')))

    prompt = '\n'.join(prompts)

    completion = openai.Completion.create(
        engine=engine_id,
        prompt=prompt,
        max_tokens=1024,
        temperature=0.8,
        n=1,
        stop=['\n\n\n']
    )
    # print the completion
    # print(completion)
    if completion and completion.choices:
        # return completion.choices[0].text
        # print(completion)
        return completion
    else:
        retry += 1
        if retry < 3:
            logger.info('retry get answer from openai: {}'.format(retry))
            time.sleep(0.1)
            return get_answer_old(conversations=conversations, retry=retry)
        else:
            return None


class ChatCompletion(object):
    def __init__(self, api_key=None):
        self.api_key = api_key

    def create(self, conversations: list, model):
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            'Authorization': 'Bearer {}'.format(self.api_key),
            'Content-Type': 'application/json'
        }
        payload = {
            "model": model,
            "max_tokens": 1024,
            "temperature": 0.8,
            "n": 1,
            "stop": ['\n\n\n'],
            # "messages": [{"role": "user", "content": "hôm nay ăn gì nhỉ ?"}],
            "messages": conversations,
        }

        response = requests.request("POST", url, headers=headers, json=payload)
        return response


# create a completion
def get_answer(conversations: list, model='gpt-3.5-turbo', retry=0):
    logger.info(conversations)
    completion = None

    try:
        # completion = ChatCompletion.create(
        #     model=engine_id,
        #     max_tokens=1024,
        #     temperature=0.8,
        #     n=1,
        #     stop=['\n\n\n'],
        #     messages=conversations)
        # print the completion
        # print(completion)

        response = ChatCompletion(API_KEY).create(conversations, model=model)
        if response.status_code == 200:
            completion = response.json()
        else:
            logger.error(response.text)
    except Exception as e:
        logger.error(str(e))
    else:
        if completion and completion.get('choices'):
            return completion
        else:
            retry += 1
            if retry < 1:
                logger.info('retry get answer from openai: {}'.format(retry))
                time.sleep(0.3)
                return get_answer(conversations=conversations, retry=retry)

    return None


def main():
    get_answer_old('tại sao con lợn biết bay ?')


if __name__ == '__main__':
    models = openai.Model.list()
    print(models)
