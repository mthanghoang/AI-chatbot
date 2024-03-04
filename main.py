import os
import streamlit as st
import pickle
import time
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain, ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

st.title("ИИ-чатбот")
main_placeholder = st.empty()
# llm = OpenAI(temperature=0.9, max_tokens=500)


file_path = "faiss_store_openai.pkl"

query = main_placeholder.text_input("Ваш вопрос: ")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.9)
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
            result = chain({"question": query}, return_only_outputs=True)
            st.header("Answer")
            st.write(result["answer"])
            sources = result["sources"]
            st.subheader("Источники для вашей справки:")
            docs = retriever.get_relevant_documents(query)
            sources_set = {doc.metadata['source'] for doc in docs}
            for source in sources_set:
                st.write(source)