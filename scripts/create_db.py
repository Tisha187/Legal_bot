import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Folder containing your PDFs
DATA_DIR = "../data/raw_pdfs"
CHROMA_DIR = "../chroma_db"

def load_documents(folder_path):
    all_docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            print(f" Loading: {filename}")
            loader = UnstructuredPDFLoader(path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = filename  
            all_docs.extend(docs)
    return all_docs

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_documents(documents)

def create_vector_database(docs):
    print("🔄 Creating vector store with HuggingFace embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    vectorstore.persist()
    print("Vector DB created and saved to chroma_db/")

if __name__ == "__main__":
    print(" Loading PDFs from:", DATA_DIR)
    documents = load_documents(DATA_DIR)

    print(f" Total documents loaded: {len(documents)}")
    chunks = split_documents(documents)

    print("🔍 Searching for 'section 302' in chunks...")
    found = False
    for chunk in chunks:
        if "section 302" in chunk.page_content.lower():
            print("\n✅ Found match:\n")
            print(chunk.page_content[:500])
            found = True
            break

    if not found:
        print(" 'Section 302' not found in any chunk!")

    print(f" Total text chunks created: {len(chunks)}")
    create_vector_database(chunks)
