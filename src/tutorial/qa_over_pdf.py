from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

import os

# set openAI API key
os.environ["OPENAI_API_KEY"] = "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"

# load the document
loader = PyPDFLoader('C:/Users/lange/Documents/PyCharmProjects/chatbot-fuer-die-handwerkskammer-hannover/data/FAQ.pdf')
faq = loader.load()

# split the documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=50)
texts = text_splitter.split_documents(faq)

# select which embeddings we want to use
embeddings = OpenAIEmbeddings()

# create the vectorestore to use as the index
db = Chroma.from_documents(texts, embeddings)

# expose this index in a retriever interface
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":1})

# create a chain to answer questions
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)

query = 'Wo finde ich die Betriebsnummer der Handwerkskammer?'
result = qa({"query": query})
print(result)

