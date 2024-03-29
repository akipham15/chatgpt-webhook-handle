import csv
import os
import random

from dotenv import load_dotenv
from langchain import PromptTemplate, LLMChain
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAIChat
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from logzero import logger

from app import constants
from app.config import Config
from app.libs.chatgptapi import get_answer_from_chatgpt
from app.libs.utils import langchain_get_chat_from_user

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# LLM = OpenAI(batch_size=constants.MIN_DOCS + 2, temperature=0.1, max_tokens=1024)
LLM = ChatOpenAI(model_name=constants.OPENAI_MODEL_NAME, temperature=0)


def recreate_training_file(train_path, output_path, is_new=False):
    logger.info(f'recreate_training_file from: {train_path}')
    write_mode = 'a'
    if is_new:
        write_mode = 'w+'

    logger.info(f'write_mode: {write_mode}')
    with open(train_path, 'r', encoding='utf-8-sig') as file:
        csv_reader = csv.reader(file, delimiter=',', quotechar='"')
        with open(output_path, write_mode) as output_file:
            csv_writer = csv.writer(output_file, delimiter=',', quotechar='"')
            # write csv file
            for row in csv_reader:
                question = row[0]
                answer = row[1]

                if question and answer:
                    question = question.lower().strip()
                    answer = answer.strip()
                    csv_writer.writerow([question, answer])

            output_file.close()
        file.close()

    logger.info('recreate_training_file done.')


def lower_train_file_question() -> str:
    train_path = Config.QA_TRAIN_DATA_PATH
    custom_train_path = Config.CUSTOM_TRAIN_DATA_PATH

    location = os.path.dirname(train_path)
    filename = os.path.basename(Config.QA_TRAIN_DATA_PATH)
    lower_filename = f'lower_{filename}'
    lower_train_path = os.path.join(location, lower_filename)

    recreate_training_file(train_path, lower_train_path, is_new=True)
    recreate_training_file(custom_train_path, lower_train_path)

    logger.info('write lower csv file success.')
    return lower_train_path


def create_persist_directory():
    # lowercase content
    lower_train_path = lower_train_file_question()

    # loader = TextLoader(lower_train_path)
    loader = CSVLoader(file_path=lower_train_path, csv_args={
        'delimiter': ',',
        'quotechar': '"',
        'fieldnames': ['question', 'answer']
    })

    # text_splitter = CharacterTextSplitter(chunk_overlap=70, chunk_size=1000, separator="\n\n\n")
    # loader = DirectoryLoader("./data/fpt/", glob="**/*.txt", loader_cls=TextLoader)
    text_splitter = CharacterTextSplitter(
        chunk_overlap=100, chunk_size=1000, separator="\n\n\n")

    documents = loader.load()
    # logger.info(documents)
    logger.info(len(documents))
    docs = text_splitter.split_documents(documents)
    # docs = text_splitter.split_text(pages)

    embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    docsearch = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory=constants.PERSIST_DIRECTORY_FPT_EXCHANGE)
    # texts = text_splitter.split_text(documents)
    # docsearch = Chroma.from_texts(texts, embedding=embedding, persist_directory=PERSIST_DIRECTORY)
    docsearch.persist()


def get_qa_chain():
    create_persist_directory()

    embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    docsearch = Chroma(
        persist_directory=constants.PERSIST_DIRECTORY_FPT_EXCHANGE,
        embedding_function=embedding)

    # method 1
    qa_chain = load_qa_chain(LLM, chain_type="stuff")

    # method 2
    # template = """
    # QUESTION: {question}
    # =========
    # {summaries}
    # =========
    # FINAL ANSWER:
    # """
    # PROMPT = PromptTemplate(template=template, input_variables=["summaries", "question"])
    # qa_chain = load_qa_with_sources_chain(LLM, chain_type="stuff", prompt=PROMPT)

    logger.info('get_qa_chain done.')
    return qa_chain, docsearch


def get_token_cost(callback):
    print(f"Total Tokens: {callback.total_tokens}")
    print(f"Prompt Tokens: {callback.prompt_tokens}")
    print(f"Completion Tokens: {callback.completion_tokens}")
    print(f"Total Cost (USD): ${callback.total_cost}")

    return callback.total_tokens


def get_default_answer(answers: list = constants.DEFAULT_ANSWERS):
    return answers[random.randint(0, len(answers) - 1)]


def filter_docs(source_docs: list, distance, is_shuffle=False, k=20):
    valid_docs = []
    for doc in source_docs:
        _doc, query_distance = doc
        if query_distance <= distance and len(valid_docs) < k:
            valid_docs.append((_doc, query_distance))
        else:
            break

    if is_shuffle:
        random.shuffle(valid_docs)

    return valid_docs


def generate_answer(query: str, histories: list):
    logger.info(histories)
    template = """You are a chatbot having a conversation with a human.

    {chat_history}
    Human: {human_input}
    Chatbot:"""

    prompt_with_history = PromptTemplate(
        input_variables=["chat_history", "human_input"],
        template=template
    )
    # memory = ConversationBufferMemory(memory_key="chat_history")
    llm_chain = LLMChain(
        llm=OpenAIChat(temperature=0.6, model="gpt-3.5-turbo"),
        prompt=prompt_with_history,
        verbose=True,
    )

    histories_string = '\n'.join(histories)
    result = llm_chain.predict(
        chat_history=histories_string, human_input=query)
    logger.info(result)
    return result


def update_csv(histories):
    from csv import writer

    with open(Config.CUSTOM_TRAIN_DATA_PATH, 'a+') as file:
        csv_writer = writer(file)
        row = histories[-2:]
        row = [item.replace('Chatbot:', '').replace('Human:', '').replace('question:', '').replace(
            'answer:', '').strip() for item in row]
        csv_writer.writerow(row)
        logger.info('Update csv done.')


def update_docsearch(docsearch, histories):
    train_data = []
    for item in histories:
        content = item.get('content')
        if item.get('role') == 'user':
            question = f'question: {content}'
            if not content.endswith('?'):
                question = f'{question} ?'
            train_data.append(question)
        if item.get('role') == 'assistant':
            answer = f'answer: {content}'
            train_data.append(answer)

    train_query = '\n'.join(train_data)
    logger.info(f'train query: {train_query}')
    update_csv(train_data)
    docsearch.add_texts(train_data)
    logger.info('update done.')


def get_answer_with_documents(query: str, histories: list, email=None):
    if histories is None:
        histories = []

    query_token = LLM.get_num_tokens(query)
    log_content = {
        'action': 'get_answer_with_docs',
        'query': query,
        'valid_docs': [],
        'result': None,
        'token_use': 0,
    }

    result = None
    token_use = query_token
    if query_token <= Config.LIMIT_MESSAGE_TOKEN:
        # query and get answer from db if posible
        qa_chain, docsearch = get_qa_chain()

        with get_openai_callback() as cb:
            query = str(query).lower()
            score_query_docs = docsearch.similarity_search_with_score(
                query, k=3)
            token_use += get_token_cost(cb)

        # logger.info(score_query_docs)
        valid_docs = filter_docs(score_query_docs, 0.22)
        if valid_docs:
            match_doc, doc_distance = valid_docs[0]
            match_content = match_doc.page_content
            if 'answer:' in match_content:
                result = match_content.split('answer:')[-1].strip()

        if query.startswith('/train') and email in Config.PERMISSION_USERS:
            logger.info('train data')
            histories = langchain_get_chat_from_user(email=email, num_of_history=1, use_object_format=True)
            query_object = {"role": "assistant", "content": query.replace('/train', '')}
            histories.append(query_object)
            update_docsearch(docsearch, histories)

            result = 'Train done.'
            return result, token_use

        if not result:
            valid_docs = filter_docs(score_query_docs, 0.28, is_shuffle=True)
            if valid_docs:
                match_docs = [doc[0] for doc in valid_docs]
                # match_doc, doc_distance = valid_docs[0]
                with get_openai_callback() as cb:
                    # method 1
                    result = qa_chain.run(
                        input_documents=match_docs, question=query)
                    logger.info(result)

                    # method 2
                    # response = qa_chain({"input_documents": match_docs, "question": query}, return_only_outputs=True)
                    # logger.info(response)
                    # result = response.get('output_text')

                    if 'answer:' in result:
                        result = result.split('answer:')[-1].strip()
                    if 'SOURCES:' in result:
                        result = result.split('SOURCES:')[0].strip()

                    token_use += get_token_cost(cb)
                result = f'{result}\n{constants.RESPONSE_POSFIX_DOC_REFERENCE}'
            else:
                # result = get_default_answer()
                # with get_openai_callback() as cb:
                # result = generate_answer(query, histories)
                # token_use += get_token_cost(cb)

                # get answer from chatgpt
                histories = langchain_get_chat_from_user(email=email, num_of_history=3, use_object_format=True)
                query_object = {"role": "user", "content": query}
                histories.append(query_object)
                result, cost = get_answer_from_chatgpt(api_token=OPENAI_API_KEY, histories=histories)

                token_use += cost
                result = f'{result}\n{constants.RESPONSE_POSFIX}'

        log_content['valid_docs'] = valid_docs
    else:
        result = get_default_answer(constants.MESSAGE_TOO_LONG)

    log_content['result'] = result
    log_content['token_use'] = token_use
    logger.info(log_content)

    return result, token_use


def main():
    # query = "What did the president say about Ketanji Brown Jackson"
    while True:
        print("\nInput question: ")
        query = input()
        result = get_answer_with_documents(query, [])
        print(result)


if __name__ == '__main__':
    main()
