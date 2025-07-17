import gradio as gr
import os
import tempfile
import shutil
import hashlib
import time
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

OWNER_API_KEY = os.environ.get("GOOGLE_API_KEY")
FREE_QUOTA = 5

def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def get_loader(path):
    if path.endswith(".pdf"):
        return PyPDFLoader(path)
    elif path.endswith(".docx"):
        return UnstructuredWordDocumentLoader(path)
    else:
        return TextLoader(path, encoding="utf-8")

def process_files(files, embeddings):
    docs = []
    temp_dir = tempfile.mkdtemp()
    file_hashes = []

    for file_path in files:
        temp_file_path = os.path.join(temp_dir, os.path.basename(file_path))
        shutil.copy2(file_path, temp_file_path)

        try:
            loader = get_loader(temp_file_path)
            loaded_docs = loader.load()

            # Add source metadata manually
            for doc in loaded_docs:
                doc.metadata["source"] = os.path.basename(file_path)

            docs.extend(loaded_docs)
            file_hashes.append(file_hash(file_path))
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    # Use improved splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)

    persist_dir = tempfile.mkdtemp()
    db = Chroma.from_documents(split_docs, embedding=embeddings, persist_directory=persist_dir)

    shutil.rmtree(temp_dir)
    return db, persist_dir, file_hashes

def get_embeddings(api_key):
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

def safe_rmtree(directory, retries=3, delay=1):
    for _ in range(retries):
        try:
            shutil.rmtree(directory)
            return
        except PermissionError:
            time.sleep(delay)
        except Exception as e:
            print(f"Failed to delete {directory}: {e}")
            return

def chat_rag(user_message, history, files, user_api_key, session_state):
    if session_state is None or session_state.get("reset", False):
        session_state = {
            "db": None,
            "persist_dir": None,
            "query_count": 0,
            "api_key": OWNER_API_KEY,
            "processed_hashes": [],
            "reset": False,
        }

    current_file_paths = [f.name for f in files] if files else []
    current_file_hashes = [file_hash(f.name) for f in files] if files else []

    if current_file_hashes and current_file_hashes != session_state.get("processed_hashes"):
        # Prefer user API key if provided
        api_key_for_embedding = user_api_key or OWNER_API_KEY
        if not api_key_for_embedding:
            return "Please provide an API key to process the documents.", session_state

        embeddings = get_embeddings(api_key_for_embedding)
        db, persist_dir, file_hashes = process_files(current_file_paths, embeddings)

        old_dir = session_state.get("persist_dir")

        session_state.update({
            "db": db,
            "persist_dir": persist_dir,
            "processed_hashes": file_hashes,
            "api_key": api_key_for_embedding,
        })

        if old_dir and os.path.exists(old_dir):
            safe_rmtree(old_dir)

    if session_state["query_count"] < FREE_QUOTA:
        api_key = OWNER_API_KEY
        info = f"Using free quota ({session_state['query_count'] + 1}/{FREE_QUOTA})"
    else:
        if not user_api_key:
            return "Free quota exceeded. Please enter your Gemini API key.", session_state
        api_key = user_api_key
        info = "Using your API key."

    if not session_state.get("db"):
        return "Please upload at least one document to start.", session_state

    try:
        query = user_message.strip()
        if not query:
            return "Please enter a valid question.", session_state

        session_state["query_count"] += 1

        retriever = session_state["db"].as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        )
        relevant_docs = retriever.invoke(query)

        if not relevant_docs:
            return "No relevant documents found for your query.", session_state

        # Prepare context for Gemini generation
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        prompt = (
            f"You are a helpful assistant. Use the following context to answer the user's question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

        # Use Gemini model to generate answer
        llm = GoogleGenerativeAI(
            model="models/gemini-2.0-flash", google_api_key=api_key
        )
        answer = llm.invoke(prompt)

        # Build final response with generation and doc references
        response = f"{info}\n\n**Answer:**\n{answer}\n\n**Sources:**\n"
        for i, doc in enumerate(relevant_docs, 1):
            response += f"- Document {i} (Source: {doc.metadata.get('source', 'unknown')})\n"

        return response, session_state

    except Exception as e:
        return f"An error occurred during generation: {str(e)}", session_state

# Gradio UI with Reset Button
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# RAG Chatbot with Gemini\nUpload documents and ask questions!")

    session_state = gr.State()

    with gr.Row():
        with gr.Column(scale=1):
            files = gr.File(
                label="Upload your documents", file_count="multiple", type="filepath"
            )
            user_api_key = gr.Textbox(
                label="Your Gemini API Key",
                placeholder="Required after free quota is used",
                type="password",
            )
            reset_btn = gr.Button("Reset Chat & State")
            gr.Markdown(
                "**Note:** Your documents are processed only in-session and are not stored permanently."
            )
        with gr.Column(scale=2):
            chatbot = gr.ChatInterface(
                fn=chat_rag,
                additional_inputs=[files, user_api_key, session_state],
                additional_outputs=[session_state],
                title="Document Chat",
                description=(
                    f"Ask questions about your uploaded documents. "
                    f"You get {FREE_QUOTA} queries for free, then enter your Gemini API key to continue."
                ),
            )

    # Reset logic
    reset_btn.click(fn=lambda: {"reset": True}, outputs=[session_state])

demo.launch(debug=True)
