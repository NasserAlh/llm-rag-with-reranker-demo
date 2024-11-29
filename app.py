import os
import tempfile
import chromadb
import ollama
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile
from typing import Generator

# Load environment variables
load_dotenv()

system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""


def process_document(uploaded_file: UploadedFile) -> list[Document]:
    """Processes an uploaded PDF file by converting it to text chunks.

    Takes an uploaded PDF file, saves it temporarily, loads and splits the content
    into text chunks using recursive character splitting.

    Args:
        uploaded_file: A Streamlit UploadedFile object containing the PDF file

    Returns:
        A list of Document objects containing the chunked text from the PDF

    Raises:
        IOError: If there are issues reading/writing the temporary file
    """
    # Store uploaded file as a temp file
    temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
    temp_file.write(uploaded_file.read())

    loader = PyMuPDFLoader(temp_file.name)
    docs = loader.load()
    os.unlink(temp_file.name)  # Delete temp file

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
    )
    return text_splitter.split_documents(docs)


def get_vector_collection() -> chromadb.Collection:
    """Gets or creates a ChromaDB collection for vector storage.

    Creates an Ollama embedding function using the nomic-embed-text model and initializes
    a persistent ChromaDB client. Returns a collection that can be used to store and
    query document embeddings.

    Returns:
        chromadb.Collection: A ChromaDB collection configured with the Ollama embedding
            function and cosine similarity space.
    """
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )

    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
    return chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )


def add_to_vector_collection(all_splits: list[Document], file_name: str):
    """Adds document splits to a vector collection for semantic search.

    Takes a list of document splits and adds them to a ChromaDB vector collection
    along with their metadata and unique IDs based on the filename.

    Args:
        all_splits: List of Document objects containing text chunks and metadata
        file_name: String identifier used to generate unique IDs for the chunks

    Returns:
        None. Displays a success message via Streamlit when complete.

    Raises:
        ChromaDBError: If there are issues upserting documents to the collection
    """
    collection = get_vector_collection()
    documents, metadatas, ids = [], [], []

    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
    st.success("Data added to the vector store!")


def query_collection(prompt: str, n_results: int = 10):
    """Queries the vector collection with a given prompt to retrieve relevant documents.

    Args:
        prompt: The search query text to find relevant documents.
        n_results: Maximum number of results to return. Defaults to 10.

    Returns:
        dict: Query results containing documents, distances and metadata from the collection.

    Raises:
        ChromaDBError: If there are issues querying the collection.
    """
    collection = get_vector_collection()
    results = collection.query(query_texts=[prompt], n_results=n_results)
    return results


def call_llm(context: str, prompt: str, temperature: float, top_p: float, num_ctx: int, conversation_history: list = None):
    """Calls the language model with context and prompt to generate a response.

    Uses Ollama to stream responses from a language model by providing context and a
    question prompt. The model uses a system prompt to format and ground its responses appropriately.

    Args:
        context: String containing the relevant context for answering the question
        prompt: String containing the user's question
        temperature: Controls randomness in output generation
        top_p: Controls diversity via nucleus sampling
        num_ctx: Maximum context length
        conversation_history: List of previous messages in the conversation

    Yields:
        String chunks of the generated response as they become available from the model

    Raises:
        OllamaError: If there are issues communicating with the Ollama API
    """
    system_prompt = """You are a helpful AI assistant with access to both document context and conversation history. When answering questions:
    1. Use information from both the provided context and previous conversation when relevant
    2. If you're unsure or neither the context nor conversation history contains the answer, say so
    3. Keep responses clear and concise
    4. Format responses using markdown when helpful
    5. Maintain consistency with your previous responses"""

    # Initialize messages with system prompt
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history if available
    if conversation_history:
        messages.extend(conversation_history)
    
    # Add current context and question
    messages.append({
        "role": "user",
        "content": f"Context: {context}\nQuestion: {prompt}"
    })

    response = ollama.chat(
        model="llama3.2:3b",
        messages=messages,
        options={
            "temperature": temperature,
            "top_p": top_p,
            "num_ctx": num_ctx,
            "stop": ["\n\n Human:"]
        },
        stream=True
    )
    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break


def call_deepseek_llm(context: str, prompt: str, temperature: float = 0.7, conversation_history: list = None) -> Generator:
    """Calls the DeepSeek language model with context and prompt using OpenAI SDK.
    
    Args:
        context: String containing the relevant context
        prompt: String containing the user's question
        temperature: Controls randomness in output generation
        conversation_history: List of previous messages in the conversation
        
    Returns:
        Generator yielding chunks of the model's response
    """
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com/v1")

    # Initialize messages with system prompt
    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history if available
    if conversation_history:
        messages.extend(conversation_history)

    # Add current context and prompt
    current_prompt = f"Context:\n{context}\n\nQuestion: {prompt}"
    messages.append({"role": "user", "content": current_prompt})

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=temperature,
            stream=True
        )

        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    except Exception as e:
        yield f"Error calling DeepSeek API: {str(e)}"


def re_rank_cross_encoders(documents: list[str]) -> tuple[str, list[int]]:
    """Re-ranks documents using a cross-encoder model for more accurate relevance scoring.

    Uses the MS MARCO MiniLM cross-encoder model to re-rank the input documents based on
    their relevance to the query prompt. Returns the concatenated text of the top 3 most
    relevant documents along with their indices.

    Args:
        documents: List of document strings to be re-ranked.

    Returns:
        tuple: A tuple containing:
            - relevant_text (str): Concatenated text from the top 3 ranked documents
            - relevant_text_ids (list[int]): List of indices for the top ranked documents

    Raises:
        ValueError: If documents list is empty
        RuntimeError: If cross-encoder model fails to load or rank documents
    """
    relevant_text = ""
    relevant_text_ids = []

    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    ranks = encoder_model.rank(prompt, documents, top_k=3)
    for rank in ranks:
        relevant_text += documents[rank["corpus_id"]]
        relevant_text_ids.append(rank["corpus_id"])

    return relevant_text, relevant_text_ids


if __name__ == "__main__":
    # Must be the first Streamlit command
    st.set_page_config(page_title="RAG Question Answer")

    # Initialize session state for conversation history and other states
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'model' not in st.session_state:
        st.session_state.model = "deepseek"
    
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.7
    
    if 'top_p' not in st.session_state:
        st.session_state.top_p = 0.7
    
    if 'num_ctx' not in st.session_state:
        st.session_state.num_ctx = 4096

    # Callback functions for state updates
    def clear_chat():
        st.session_state.messages = []
        st.rerun()

    def update_model():
        st.session_state.model = st.session_state.model_select

    # Sidebar configuration
    with st.sidebar:
        st.header("Model Configuration")
        
        # Model selection
        st.selectbox(
            "Select Model",
            ["deepseek", "mistral", "llama2", "neural-chat"],
            key="model_select",
            on_change=update_model,
            index=["deepseek", "mistral", "llama2", "neural-chat"].index(st.session_state.model)
        )

        # Model parameters
        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.1,
            key="temp_slider",
            help="Controls randomness in the output. Higher values make the output more random."
        )

        st.session_state.top_p = st.slider(
            "Top P",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.top_p,
            step=0.1,
            key="top_p_slider",
            help="Controls diversity via nucleus sampling"
        )

        st.session_state.num_ctx = st.slider(
            "Context Length",
            min_value=512,
            max_value=16384,
            value=st.session_state.num_ctx,
            step=512,
            key="num_ctx_slider",
            help="Maximum number of tokens to consider for context"
        )

        # Add clear chat button
        st.button("Clear Chat History", on_click=clear_chat)

    st.title("RAG Question Answer")

    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file:
        # Process document
        with st.spinner("Processing document..."):
            splits = process_document(uploaded_file)
            add_to_vector_collection(splits, uploaded_file.name)
            st.success("Document processed successfully!")

    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get the user question
    if prompt := st.chat_input("Ask a question about the document"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get relevant documents
        results = query_collection(prompt)
        documents = results["documents"][0]  # documents are already strings

        # Re-rank documents
        relevant_text, relevant_text_ids = re_rank_cross_encoders(documents)

        # Create the assistant message
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Use the model selected in sidebar
            if st.session_state.model == "deepseek":
                response_generator = call_deepseek_llm(
                    relevant_text, 
                    prompt, 
                    temperature=st.session_state.temperature,
                    conversation_history=st.session_state.messages
                )
            else:
                response_generator = call_llm(
                    relevant_text,
                    prompt,
                    temperature=st.session_state.temperature,
                    top_p=st.session_state.top_p,
                    num_ctx=st.session_state.num_ctx,
                    conversation_history=st.session_state.messages,
                )

            # Stream the response
            for chunk in response_generator:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Display document sources
        with st.expander("View Sources"):
            for idx in relevant_text_ids:
                st.markdown(f"Document chunk {idx}:")
                st.markdown(documents[idx])
