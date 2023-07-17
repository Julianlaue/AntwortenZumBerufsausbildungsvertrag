from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI
from openai.error import OpenAIError

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import streamlit as st
import json

import os


def clear_submit():
    st.session_state["submit"] = False


# set up a function for the introduction page
def intro_page():
    st.title("Welcome to HandwerksGPT!")
    st.markdown("Please input your OpenAI API key to continue:")

    # Create an input box for the OpenAI API key
    openai_key = st.text_input('OpenAI API Key', type='password')

    # Create a button for submitting the OpenAI API key
    if st.button('Submit API Key'):
        if openai_key:
            # Store the API key in the session state
            st.session_state["OPENAI_API_KEY"] = openai_key
            st.session_state["page"] = "app"  # switch to the app page
        else:
            st.error("Please enter an API key.")


# set up a function for the app page
def app_page():
    # set openAI API key
    #os.environ["OPENAI_API_KEY"] = "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"

    st.session_state["number"] = 0

    # load the document
    loader = PyPDFLoader('../../data/HandwerkskammerBotInfo.pdf')
    historyjson = '../../data/history.json'
    faq = loader.load()

    # split the documents into chunks
    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(faq)

    # select which embeddings we want to use
    embeddings = OpenAIEmbeddings()

    # create the vectorestore to use as the index
    db = Chroma.from_documents(texts, embeddings)

    # expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    # create a chain to answer questions
    qa = ConversationalRetrievalChain.from_llm(
        llm=OpenAI(model_name='gpt-3.5-turbo'), chain_type="stuff", retriever=retriever, return_source_documents=False)

    # If there's no chat history in the session state yet, create a new list
    st.session_state["chat_history"] = []
    chat_history = [] 

    cont = True

    st.set_page_config(page_title="HandwerksGPT", page_icon="🔧", layout="wide")
    st.header("🔧Fragen an die Handwerkskammer")

    query = st.text_area("Stellen Sie Fragen an die Handwerkskammer zum Berufsausbildungsvertrag",
                         on_change=clear_submit)

    button = st.button("Submit")

    if button:
        st.session_state["submit"] = True
        # Output Columns
        answer_col = st.columns(1)
        try:
            # Load chat history from JSON file if it exists
            if os.path.exists(historyjson):
                with open(historyjson, 'r') as f:
                    chat_history = [tuple(item) for item in json.load(f)]
            else:
                chat_history = []
            # Get the answer from GPT-4
            answer = qa({'question': query, 'chat_history': chat_history})

            # Append the user's question and GPT-4's answer to the chat history
            chat_history.append((query, answer["answer"]))

            # Save chat history to JSON file
            with open(historyjson, 'w') as f:
                json.dump(chat_history, f)

                # Display the chat history
                for i in range(0, len(chat_history)):
                    st.markdown(f"**Frage:** {chat_history[i][0]}")
                    st.markdown(f"**Antwort:** {chat_history[i][1]}")
            st.session_state["number"] = st.session_state["number"] + 1
            print(st.session_state['number'])

        except OpenAIError as e:
            st.error(e._message)

# main
def main():
    if "started" not in st.session_state:
        st.session_state['started'] = True
        historyjson = '../../data/history.json'
        if os.path.isfile(historyjson):
            os.remove(historyjson)
    # initialize the session state variables
    if "OPENAI_API_KEY" not in st.session_state:
        st.session_state["OPENAI_API_KEY"] = ""
    if "page" not in st.session_state:
        st.session_state["page"] = "intro"

    # Determine which page to display based on the session state
    if st.session_state["page"] == "intro":
        intro_page()
    elif st.session_state["page"] == "app":
        # set the OPENAI_API_KEY environment variable
        os.environ["OPENAI_API_KEY"] = st.session_state["OPENAI_API_KEY"]
        app_page()


if __name__ == "__main__":
    main()


