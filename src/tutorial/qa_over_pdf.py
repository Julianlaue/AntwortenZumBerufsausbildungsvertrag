from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI
from openai.error import OpenAIError

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import streamlit as st

import os

def clear_submit():
    st.session_state["submit"] = False

# set openAI API key
os.environ["OPENAI_API_KEY"] = "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"

# load the document
loader = PyPDFLoader('C:/Users/lange/Documents/PyCharmProjects/chatbot-fuer-die-handwerkskammer-hannover/data/HandwerkskammerBotInfo.pdf')
faq = loader.load()

# split the documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=50)
texts = text_splitter.split_documents(faq)

# select which embeddings we want to use
embeddings = OpenAIEmbeddings()

# create the vectorestore to use as the index
db = Chroma.from_documents(texts, embeddings)

# expose this index in a retriever interface
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})

# create a chain to answer questions
qa = ConversationalRetrievalChain.from_llm(
    llm=OpenAI(model_name='gpt-4'), chain_type="stuff", retriever=retriever, return_source_documents=False)

chat_history = []
cont = True

st.set_page_config(page_title="HandwerksGPT", page_icon="🔧", layout="wide")
st.header("🔧Fragen an die Handwerkskammer")

query = st.text_area("Stellen Sie Fragen an die Handwerkskammer zum Berufsausbildungsvertrag", on_change=clear_submit)

button = st.button("Submit")

if button or st.session_state.get("submit"):
    st.session_state["submit"] = True
    # Output Columns
    answer_col = st.columns(1)
    try:
        answer = qa({'question': query, 'chat_history': chat_history})

        st.markdown("#### Answer")
        st.markdown(answer["answer"])

    except OpenAIError as e:
        st.error(e._message)


