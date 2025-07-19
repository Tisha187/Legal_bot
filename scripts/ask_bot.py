import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain

# Load Chroma DB
CHROMA_DIR = "../chroma_db"
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)

# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=st.secrets["GOOGLE_API_KEY"]
)


custom_prompt = PromptTemplate(
    template="""
    You are a legal assistant. Use the following legal context to answer the user's question.
    If the context does not contain relevant information, then answer from your own knowledge.

    Context:
    {context}

    Question: {question}
    Answer:""",
    input_variables=["context", "question"]
)


# RAG QA Chain using Gemini + Chroma
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 2}),
    chain_type="stuff",
    chain_type_kwargs={
        "prompt": custom_prompt,
        "document_variable_name": "context"  # THIS is the fix!
    },
    return_source_documents=True
)

# Function to ask question
def ask_legal_bot(query):
    result = qa_chain.invoke({"question": query})
    answer = result["answer"]
    sources = [doc.metadata.get("source", "N/A") for doc in result.get("source_documents", [])]
    return answer, sources

if __name__ == "__main__":
    test_query = "What is Section 302 of IPC?"
    response = ask_legal_bot(test_query)
    print("Test Query:", test_query)
    print("Bot Response:", response)




