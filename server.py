import os
import shutil
from typing import List, Dict
from fastapi.responses import JSONResponse, FileResponse
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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

# Load environment variables from .env
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
documents_dir = os.path.join(current_dir, "documents")

# Global variables to store embeddings, retrievers, and rag_chains for each company
company_data = {}

class ChatRequest(BaseModel):
    company_id: str
    message: str

def load_pdf_documents(pdf_directory: str) -> List[Document]:
    """
    Load documents from PDF files in the specified directory.
    """
    documents = []
    for filename in os.listdir(pdf_directory):
        if filename.endswith('.pdf'):
            file_path = os.path.join(pdf_directory, filename)
            print(f"Loading PDF: {file_path}")
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
    return documents

def create_vector_store(docs: List[Document], embeddings, store_name: str):
    """
    Create a vector store from the given documents if it doesn't exist.
    """
    persistent_directory = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_directory):
        print(f"\n--- Creating vector store {store_name} ---")
        Chroma.from_documents(
            docs, embeddings, persist_directory=persistent_directory)
        print(f"--- Finished creating vector store {store_name} ---")
    else:
        print(f"Vector store {store_name} already exists. No need to initialize.")

def setup_retriever(embeddings, store_name: str):
    """
    Set up and return a retriever for the vector store.
    """
    persistent_directory = os.path.join(db_dir, store_name)
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
    
    return db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

def setup_llm():
    """
    Set up and return the language model.
    """
    return ChatOpenAI(model="gpt-4o-mini")

def create_contextualize_q_prompt():
    """
    Create and return the prompt for contextualizing questions.
    """
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
    """
    Create and return the prompt for answering questions.
    """
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
    """
    Set up and return the RAG chain.
    """
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

@app.get("/")
def greet():
    return {"message":"Hello World!"}

@app.post("/upload_documents/{company_id}")
async def upload_documents(company_id: str, files: List[UploadFile] = File(...)):
    company_dir = os.path.join(documents_dir, company_id)
    os.makedirs(company_dir, exist_ok=True)
    
    for file in files:
        file_path = os.path.join(company_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    
    # Process uploaded documents
    await load_documents(company_id)
    
    return {"message": f"Documents uploaded and processed for company ID: {company_id}"}


@app.post("/load_documents/{company_id}")
async def load_documents(company_id: str):
    global company_data
    
    # Set up the embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Load PDF documents
    pdf_directory = os.path.join(documents_dir, company_id)
    if not os.path.exists(pdf_directory):
        raise HTTPException(status_code=404, detail=f"Directory not found for company ID: {company_id}")
    
    documents = load_pdf_documents(pdf_directory)

    # Create or load the vector store
    store_name = company_id
    create_vector_store(documents, embeddings, store_name)

    # Set up the retriever
    retriever = setup_retriever(embeddings, store_name)

    # Set up the language model
    llm = setup_llm()

    # Create prompts
    contextualize_q_prompt = create_contextualize_q_prompt()
    qa_prompt = create_qa_prompt()

    # Set up the RAG chain
    rag_chain = setup_rag_chain(llm, retriever, contextualize_q_prompt, qa_prompt)

    # Store the rag_chain for this company
    company_data[company_id] = {
        "rag_chain": rag_chain,
        "chat_history": []
    }

    return {"message": f"Documents loaded and processed for company ID: {company_id}"}

@app.post("/chat")
async def chat(chat_request: ChatRequest):
    company_id = chat_request.company_id
    user_message = chat_request.message
    
    if company_id not in company_data:
        raise HTTPException(status_code=400, detail=f"Documents not loaded for company ID: {company_id}. Please load documents first.")

    rag_chain = company_data[company_id]["rag_chain"]
    chat_history = company_data[company_id]["chat_history"]

    result = rag_chain.invoke({"input": user_message, "chat_history": chat_history})
    
    # Update chat history
    chat_history.append(HumanMessage(content=user_message))
    chat_history.append(SystemMessage(content=result['answer']))

    return {"response": result['answer']}

@app.get("/chatbot_script/{company_id}")
async def get_chatbot_script(company_id: str):
    script_content = f"""
    (function() {{
        var chatbotContainer = document.createElement('div');
        chatbotContainer.id = 'chatbot-container';
        chatbotContainer.style.position = 'fixed';
        chatbotContainer.style.bottom = '20px';
        chatbotContainer.style.right = '20px';
        chatbotContainer.style.width = '300px';
        chatbotContainer.style.height = '400px';
        chatbotContainer.style.border = '1px solid #ccc';
        chatbotContainer.style.borderRadius = '10px';
        chatbotContainer.style.overflow = 'hidden';
        chatbotContainer.style.display = 'flex';
        chatbotContainer.style.flexDirection = 'column';

        var chatHeader = document.createElement('div');
        chatHeader.style.padding = '10px';
        chatHeader.style.backgroundColor = '#f1f1f1';
        chatHeader.style.borderBottom = '1px solid #ccc';
        chatHeader.innerHTML = '<h3 style="margin: 0;">Chatbot</h3>';

        var chatMessages = document.createElement('div');
        chatMessages.style.flexGrow = '1';
        chatMessages.style.overflowY = 'auto';
        chatMessages.style.padding = '10px';

        var chatInput = document.createElement('input');
        chatInput.type = 'text';
        chatInput.placeholder = 'Type your message...';
        chatInput.style.width = '100%';
        chatInput.style.padding = '10px';
        chatInput.style.border = 'none';
        chatInput.style.borderTop = '1px solid #ccc';

        chatbotContainer.appendChild(chatHeader);
        chatbotContainer.appendChild(chatMessages);
        chatbotContainer.appendChild(chatInput);

        document.body.appendChild(chatbotContainer);

        chatInput.addEventListener('keypress', function(e) {{
            if (e.key === 'Enter') {{
                var message = this.value;
                if (message.trim() !== '') {{
                    addMessage('You', message);
                    sendMessage(message);
                    this.value = '';
                }}
            }}
        }});

        function addMessage(sender, message) {{
            var messageElement = document.createElement('p');
            messageElement.innerHTML = '<strong>' + sender + ':</strong> ' + message;
            chatMessages.appendChild(messageElement);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }}

        function sendMessage(message) {{
            fetch('http://localhost:8000/chat', {{
                method: 'POST',
                headers: {{
                    'Content-Type': 'application/json',
                }},
                body: JSON.stringify({{
                    company_id: '{company_id}',
                    message: message
                }})
            }})
            .then(response => {{
                if (!response.ok) {{
                    return response.text().then(text => {{ throw new Error(text) }});
                }}
                return response.json();
            }})
            .then(data => {{
                addMessage('Chatbot', data.response);
            }})
            .catch((error) => {{
                console.error('Error:', error);
                addMessage('Chatbot', 'Error: ' + error.message);
            }});
        }}
    }})();
    """
    
    # Create a JavaScript file with the content
    script_path = f"static/chatbot_script_{company_id}.js"
    with open(script_path, "w") as f:
        f.write(script_content)
    
    return FileResponse(script_path, media_type="application/javascript")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)