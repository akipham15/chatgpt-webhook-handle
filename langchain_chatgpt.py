import os
import random

from dotenv import load_dotenv
from langchain import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from logzero import logger

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MIN_DOCS = 1
PERSIST_DIRECTORY = "fpt-mail-exchange"


def create_persist_directory():
    # loader = DirectoryLoader("./data/fpt", glob="**/*.*")
    loader = TextLoader("./data/fpt/email.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_overlap=50, chunk_size=500, separator="\n")

    docs = text_splitter.split_documents(documents)

    embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    docsearch = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory=PERSIST_DIRECTORY)
    # texts = text_splitter.split_text(documents)
    # docsearch = Chroma.from_texts(texts, embedding=embedding, persist_directory=PERSIST_DIRECTORY)
    docsearch.persist()


def get_qa_chain():
    persist_directory_existed = os.path.exists(PERSIST_DIRECTORY)
    logger.info(persist_directory_existed)
    if not persist_directory_existed:
        logger.info(f'create persist_directory {PERSIST_DIRECTORY}')
        create_persist_directory()

    llm = OpenAI(batch_size=1, temperature=0, max_tokens=750)
    embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    docsearch = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding)
    qa_chain = load_qa_chain(llm, chain_type="map_reduce")

    logger.info('get_qa_chain done.')
    return qa_chain, docsearch


def get_chat_history(inputs) -> str:
    res = []
    for human, ai in inputs:
        res.append(f"Human:{human}\nAI:{ai}")
    return "\n".join(res)


def get_answer(query):
    qa_chain, docsearch = get_qa_chain()

    vectordbkwargs = {"search_distance": 0.33}
    default_answers = ['Bạn có thể liên hệ TuLT để có câu trả lời nhé!!', 'Cái này em cũng không biết ạ :|']

    # query_docs = docsearch.similarity_search(query, k=MIN_DOCS)
    score_query_docs = docsearch.similarity_search_with_score(query, k=MIN_DOCS)
    print(score_query_docs)
    # print(len(score_query_docs))
    _doc, query_distance = score_query_docs[0]
    logger.info(query_distance)
    if query_distance >= vectordbkwargs['search_distance']:
        result = default_answers[random.randint(0, len(default_answers) - 1)]
        return result

    retriever = docsearch.as_retriever(search_kwargs={"k": MIN_DOCS, "distance_metric": 'cos'})
    # docsearch.add_texts(["Ankush went to Princeton"])
    # qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=query_docs)

    llm = OpenAI(batch_size=1, temperature=0, max_tokens=750)

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        verbose=True,
        get_chat_history=get_chat_history,
    )

    # result = qa_chain.run(input_documents=query_docs, question=query)
    # result = ''

    with get_openai_callback() as cb:
        chat_history = []
        ans = qa({"question": query, "chat_history": chat_history, "vectordbkwargs": vectordbkwargs})
        result = ans.get('answer')
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")

    return result


def main():
    # query = "What did the president say about Ketanji Brown Jackson"
    while True:
        print("\nInput question: ")
        query = input()
        result = get_answer(query)
        print(result)


if __name__ == '__main__':
    main()
