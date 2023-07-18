import langchain.chains.combine_documents.map_reduce
from langchain import PromptTemplate
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
    st.title("Willkommen zu HandwerksGPT!")
    st.markdown("Bitte geben Sie Ihren OpenAI API Schlüssel ein um weiter zu machen. Wenn nach Submit nichts passiert drücken Sie erneut auf Submit:")

    # Create an input box for the OpenAI API key
    openai_key = st.text_input('OpenAI API Schlüssel', type='password')

    # Create a button for submitting the OpenAI API key
    if st.button('Submit API Key'):
        if openai_key:
            # Store the API key in the session state
            st.session_state["OPENAI_API_KEY"] = openai_key
            st.session_state["page"] = "app"  # switch to the app page
        else:
            st.error("Bitte geben Sie den API Schlüssel ein.")


# set up a function for the app page
def app_page():
    # set openAI API key
    #os.environ["OPENAI_API_KEY"] = "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"

    st.session_state["number"] = 0
    print(os.getcwd())

    # load the document
    loader = PyPDFLoader('src/HandwerkskammerBotInfo.pdf')
    historyjson = '../data/history.json'
    faq = loader.load()

    # split the documents into chunks
    text_splitter = TokenTextSplitter(chunk_size=3000, chunk_overlap=400)
    texts = text_splitter.split_documents(faq)

    # select which embeddings we want to use
    embeddings = OpenAIEmbeddings()

    # create the vectorestore to use as the index
    db = Chroma.from_documents(texts, embeddings)

    # expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})



    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    prompt_template = """Du bist eine freundliche, hilfreiche KI, die Fragen zum Berufsausbildungsvertrag beantwortet. Dein Name ist HandwerksGPT. Benutze den folgenden Kontext, um die Frage am Ende zu beantworten.
    Wenn du die Antwort anhand des Kontext nicht beantworten kannst sag, dass du die Frage nicht beantworten kannst, erfinde keine Antworten. 

    Kontext: {context}

    Frage: {question}
    Hilfreiche Antwort:"""
    QA_PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"])

    # create the llm
    llm = OpenAI(
        model_name='gpt-4',
        temperature=0.1,
        max_tokens=1000
    )

    # create a chain to answer questions
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        qa_prompt=QA_PROMPT
    )

    # If there's no chat history in the session state yet, create a new list
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
                    st.markdown(f"**Frage {i}:** {chat_history[i][0]}")
                    st.markdown(f"**Antwort {i}:** {chat_history[i][1]}")

        except OpenAIError as e:
            st.error(e._message)


# main
def main():
    # Check if the history file has been created or whether this is a new session
    if "started" not in st.session_state:
        st.session_state['started'] = True
        historyjson = '../data/history.json'
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


