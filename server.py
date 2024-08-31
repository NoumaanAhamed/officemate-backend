import os
import shutil
from typing import List
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq

# Load environment variables from .env
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
documents_dir = os.path.join(current_dir, "documents")

# Global variables to store embeddings, retrievers, and rag_chains
company_data = {}

class ChatRequest(BaseModel):
    message: str

def increment_request_counter():
    company_data["request_count"] = company_data.get("request_count", 0) + 1

def load_pdf_documents(pdf_directory: str) -> List[Document]:
    documents = []
    for filename in os.listdir(pdf_directory):
        if filename.endswith('.pdf'):
            file_path = os.path.join(pdf_directory, filename)
            print(f"Loading PDF: {file_path}")
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
    return documents

def create_vector_store(docs: List[Document], embeddings, store_name: str):
    persistent_directory = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_directory):
        print(f"\n--- Creating vector store {store_name} ---")
        Chroma.from_documents(
            docs, embeddings, persist_directory=persistent_directory)
        print(f"--- Finished creating vector store {store_name} ---")
    else:
        print(f"Vector store {store_name} already exists. Updating with new documents.")
        db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
        db.add_documents(docs)

def setup_retriever(embeddings, store_name: str):
    persistent_directory = os.path.join(db_dir, store_name)
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
    return db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

def setup_llm():
    # return ChatOpenAI(model="gpt-4o-mini")
    return ChatGroq(model="gemma2-9b-it")

def create_contextualize_q_prompt():
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )
    return ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

def create_qa_prompt():
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. Use "
        "the following pieces of retrieved context to answer the "
        "question. If you don't know the answer, just say that you "
        "don't know. Use three sentences maximum and keep the answer "
        "concise."
        "\n\n"
        "{context}"
    )
    return ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

def setup_rag_chain(llm, retriever, contextualize_q_prompt, qa_prompt):
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    retrieval_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return retrieval_chain, history_aware_retriever

@app.post("/upload_company_documents")
async def upload_company_documents(files: List[UploadFile] = File(...)):
    company_dir = os.path.join(documents_dir, "company")
    os.makedirs(company_dir, exist_ok=True)
    
    for file in files:
        file_path = os.path.join(company_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    
    # Process uploaded documents
    await load_documents()
    
    return {"message": "Company documents uploaded and processed"}

@app.post("/upload_employee_documents/{employee_id}")
async def upload_employee_documents(employee_id: str, files: List[UploadFile] = File(...)):
    employee_dir = os.path.join(documents_dir, "employees", employee_id)
    os.makedirs(employee_dir, exist_ok=True)
    
    for file in files:
        file_path = os.path.join(employee_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    
    # Process uploaded documents
    await load_documents()
    
    return {"message": f"Employee documents uploaded and processed for employee {employee_id}"}

@app.post("/load_documents")
async def load_documents():
    global company_data
    
    # Set up the embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Load company documents
    company_dir = os.path.join(documents_dir, "company")
    company_documents = load_pdf_documents(company_dir)

    # Load employee documents
    employee_documents = []
    employees_dir = os.path.join(documents_dir, "employees")
    for employee in os.listdir(employees_dir):
        employee_dir = os.path.join(employees_dir, employee)
        employee_documents.extend(load_pdf_documents(employee_dir))

    # Combine all documents
    all_documents = company_documents + employee_documents
    print(f"Total number of documents: {len(all_documents)}")

    # Create or update the vector store
    create_vector_store(all_documents, embeddings, "enterprise")

    # Set up the retriever
    retriever = setup_retriever(embeddings, "enterprise")

    # Set up the language model
    llm = setup_llm()

    # Create prompts
    contextualize_q_prompt = create_contextualize_q_prompt()
    qa_prompt = create_qa_prompt()

    # Set up the RAG chain
    rag_chain, history_aware_retriever = setup_rag_chain(llm, retriever, contextualize_q_prompt, qa_prompt)

    # Store the rag_chain, retriever, and initialize chat history
    company_data["rag_chain"] = rag_chain
    company_data["retriever"] = history_aware_retriever
    company_data["chat_history"] = []

    return {"message": "All documents loaded and processed"}

@app.post("/chat")
async def chat(chat_request: ChatRequest, background_tasks: BackgroundTasks):
    if "rag_chain" not in company_data or "retriever" not in company_data:
        raise HTTPException(status_code=400, detail="Documents not loaded. Please load documents first.")

    increment_request_counter()

    rag_chain = company_data["rag_chain"]
    retriever = company_data["retriever"]
    chat_history = company_data["chat_history"]

    # First, retrieve the relevant documents
    retrieved_documents = retriever.invoke({"input": chat_request.message, "chat_history": chat_history})

    # Then, use the RAG chain to generate the answer
    result = rag_chain.invoke({"input": chat_request.message, "chat_history": chat_history})
    
    # Update chat history
    chat_history.append(HumanMessage(content=chat_request.message))
    chat_history.append(SystemMessage(content=result['answer']))

    # Prepare the retrieved chunks for the response
    retrieved_chunks = [
        {
            "content": doc.page_content,
            "metadata": doc.metadata
        } for doc in retrieved_documents
    ]

    return {
        "response": result['answer'],
        "retrieved_chunks": retrieved_chunks
    }

@app.get("/analytics")
def get_analytics():
    return JSONResponse({"request_count": company_data.get("request_count", 0)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)