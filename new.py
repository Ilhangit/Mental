import bs4
import pandas as pd
from dotenv import load_dotenv
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, DataFrameLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.prompts import PromptTemplate

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

# PDF'yi yükle
pdf_loader = PyPDFLoader("data/DSM.pdf")
pdf_docs = pdf_loader.load()

# CSV'yi yükle (soru-cevap formatı)
csv_data = pd.read_csv("data/Mental_wellness_data.csv")

# CSV'deki her satırı LangChain dökümanına çevir
qa_docs = [
    Document(page_content=f"Question: {row['question']}\nAnswer: {row['answer']}")
    for _, row in csv_data.iterrows()
]

# PDF ve CSV verilerini birleştir
all_docs = pdf_docs + qa_docs

# Metni parçalara ayır
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(all_docs)

# Vektör veritabanı oluştur
vectorstore = Chroma.from_documents(splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# RAG promptunu çek
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are a mental health assistant trained to provide compassionate and accurate answers.
    If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    Use the following context from mental health resources and Q&A data to answer the question:

    Context: {context}

    Question: {question}

    Answer in a supportive and informative way.
    """
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG zinciri
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_prompt
    | llm
    | StrOutputParser()
)

if __name__ == '__main__':
    question = input("Enter a question: ")
    for chunk in rag_chain.stream(question):
        print(chunk, end="", flush=True)