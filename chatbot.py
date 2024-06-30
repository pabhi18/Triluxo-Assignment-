import os
from dotenv import load_dotenv
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def response(query):
    template = """Use the following pieces of context to answer the question at the end.

    {context}

    Question: {question}

    Helpful Answer:"""

    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    loader = WebBaseLoader("https://brainlox.com/courses/category/technical")
    webpage_data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
    )

    splits = text_splitter.split_documents(webpage_data)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_api_key)
    vectorstore = Chroma.from_documents(splits, embedding=embeddings)

    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=gemini_api_key)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    custom_rag_prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(query)