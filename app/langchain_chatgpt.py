import os
import random

from dotenv import load_dotenv
from langchain import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts import PromptTemplate

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from logzero import logger

from app import constants
from app.config import Config
import csv

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
LLM = OpenAI(batch_size=constants.MIN_DOCS + 2, temperature=0, max_tokens=1024)


def lower_train_file_question(train_path: str) -> str:
    location = os.path.dirname(train_path)
    filename = os.path.basename(train_path)
    lower_filename = f'lower_{filename}'
    lower_train_path = os.path.join(location, lower_filename)

    with open(train_path, 'r', encoding='utf-8-sig') as file:
        csv_reader = csv.reader(file, delimiter=',', quotechar='"')
        with open(lower_train_path, 'w+') as lower_file:
            csv_writer = csv.writer(lower_file, delimiter=',', quotechar='"')
            # write csv file
            for row in csv_reader:
                question = row[0]
                answer = row[1]

                if question and answer:
                    question = question.lower().strip()
                    answer = answer.strip()
                    csv_writer.writerow([question, answer])

    logger.info('write lower csv file success.')
    return lower_train_path


def create_persist_directory(train_path: str, persist_name: str, fieldnames=None):

    # lowercase content
    lower_train_path = lower_train_file_question(train_path)

    if not fieldnames:
        fieldnames = []

    # loader = TextLoader(lower_train_path)
    loader = CSVLoader(file_path=lower_train_path, csv_args={
        'delimiter': ',',
        'quotechar': '"',
        'fieldnames': fieldnames
    },
        # source_column='answer',
    )

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
        documents=docs, embedding=embedding, persist_directory=persist_name)
    # texts = text_splitter.split_text(documents)
    # docsearch = Chroma.from_texts(texts, embedding=embedding, persist_directory=PERSIST_DIRECTORY)
    docsearch.persist()


def get_qa_chain(train_path: str, persist_name: str, fieldnames=None):
    persist_directory_existed = os.path.exists(persist_name)
    logger.info(persist_directory_existed)
    if not persist_directory_existed:
        logger.info(f'create persist_directory {persist_name}')
        create_persist_directory(train_path, persist_name, fieldnames)

    embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    docsearch = Chroma(persist_directory=persist_name,
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


def get_chat_history(inputs) -> str:
    res = []
    for human, ai in inputs:
        res.append(f"Human:{human}")
        res.append(f"AI:{ai}")
    return "\n".join(res)


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


def get_answer_with_documents(query: str, histories: list):
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
        qa_chain, docsearch = get_qa_chain(train_path=Config.QA_TRAIN_DATA_PATH,
                                           persist_name=constants.PERSIST_DIRECTORY_FPT_EXCHANGE,
                                           fieldnames=['question', 'answer'])

        with get_openai_callback() as cb:
            query = str(query).lower()
            score_query_docs = docsearch.similarity_search_with_score(
                query, k=3)
            token_use += get_token_cost(cb)

        valid_docs = filter_docs(score_query_docs, 0.235)
        if valid_docs:
            match_doc, doc_distance = valid_docs[0]
            match_content = match_doc.page_content
            if 'answer:' in match_content:
                result = match_content.split('answer:')[-1].strip()

        if not result:
            valid_docs = filter_docs(score_query_docs, 0.33, is_shuffle=True)
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
            else:
                result = get_default_answer()

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
        result = get_answer_with_documents(query)
        print(result)


if __name__ == '__main__':
    main()
