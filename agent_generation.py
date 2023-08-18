from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader
from glob import glob
from langchain.vectorstores import Chroma

from dotenv import load_dotenv
load_dotenv('.env')
loaders = [TextLoader(x) for x in glob("converted_texts/*.txt")]

docs = []
for loader in loaders:
    docs.extend(loader.load())


text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
documents = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents, embeddings)
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), vectorstore.as_retriever(), memory=memory)
chat_history = []

while True:
    query = input("You: ")
    if query == "exit":
        break
    chat_history.append(query)
    result = qa({"question": query, "chat_history": chat_history})
    print("Bot: ", result["answer"])