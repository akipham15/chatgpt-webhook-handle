import os
import random

from dotenv import load_dotenv
from langchain import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from logzero import logger
from langchain.document_loaders.csv_loader import CSVLoader


from app import constants
from app.config import Config

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
LLM = OpenAI(batch_size=constants.MIN_DOCS + 1, temperature=0, max_tokens=1024)


def create_persist_directory(train_path: str, persist_name: str, fieldnames=None):
    # train_path = "./data/fpt/email.txt"
    # loader = TextLoader(train_path)

    if not fieldnames:
        fieldnames = []
    loader = CSVLoader(file_path=train_path, csv_args={
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
    qa_chain = load_qa_chain(LLM, chain_type="stuff", )

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


def get_answer_result(docsearch, query, histories):
    vectordbkwargs = {"search_distance": 0.395}

    retriever = docsearch.as_retriever(
        search_kwargs={"k": constants.MIN_DOCS, "distance_metric": 'cos'})
    # docsearch.add_texts(["Ankush went to Princeton"])
    # qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=query_docs)

    qa = ConversationalRetrievalChain.from_llm(
        llm=LLM,
        retriever=retriever,
        return_source_documents=True,
        verbose=True,
        get_chat_history=get_chat_history,
    )

    with get_openai_callback() as cb:
        ans = qa({"question": query, "chat_history": histories,
                  "vectordbkwargs": vectordbkwargs})
        logger.info(ans)
        result = ans.get('answer')
        token_use = get_token_cost(cb)
        return result, token_use


def get_answer_via_docs(qa_chain, docs, query: str):
    if not isinstance(docs, list):
        docs = [docs]

    with get_openai_callback() as cb:
        result = qa_chain.run(input_documents=docs, question=query)
        token_use = get_token_cost(cb)

    return result, token_use


def get_query_distance(docsearch, query, k=1, is_shuffle=False, distance=Config.DEFAULT_QUERY_DISTANCE):
    # query_docs = docsearch.similarity_search(query, k=MIN_DOCS)
    score_query_docs = docsearch.similarity_search_with_score(
        query, k=k)

    # logger.info(score_query_docs)
    valid_docs = []
    for doc in score_query_docs:
        _doc, query_distance = doc
        if query_distance <= distance:
            valid_docs.append((_doc, query_distance))

    if is_shuffle:
        random.shuffle(valid_docs)

    doc_valid, doc_distance = valid_docs[0]
    logger.info(doc_valid)
    logger.info(doc_distance)
    return doc_valid, doc_distance


def get_answer_with_documents(query: str, histories: list):
    qa_chain, docsearch = get_qa_chain(
        Config.QA_TRAIN_DATA_PATH, constants.PERSIST_DIRECTORY_FPT_EXCHANGE, ['question', 'answer', 'note'])

    if histories is None:
        histories = []

    query_token = LLM.get_num_tokens(query)
    logger.info(f'query_token: {query_token}')

    token_use = 0
    if query_token <= Config.LIMIT_MESSAGE_TOKEN:
        valid_doc, query_distance = get_query_distance(docsearch, query)
        if valid_doc:
            # result, token_use = get_answer_result(docsearch, query, histories)
            result, token_use = get_answer_via_docs(qa_chain, valid_doc, query)
        else:
            _qa_chain, docsearch = get_qa_chain(
                Config.WELCOME_TRAIN_DATA_PATH, constants.PERSIST_DIRECTORY_FPT_WELCOME, ['question', 'answer'])

            valid_doc, query_distance = get_query_distance(
                docsearch, query, k=3, is_shuffle=True, distance=0.395)
            if query_distance < Config.DEFAULT_QUERY_DISTANCE:
                result = valid_doc.page_content
                if 'answer:' in result:
                    result = result.split('answer:')[-1].strip()
            else:
                result = get_default_answer()
    else:
        result = get_default_answer(constants.MESSAGE_TOO_LONG)

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
