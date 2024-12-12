import streamlit as st
import traceback
from typing import List, Optional, Dict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

class RAGChatbot:
    def __init__(self, 
                 embedding_model="mxbai-embed-large:latest",
                 llm_model="llama3.2:1b",
                 chunk_size=500,
                 chunk_overlap=100):
        """
        Initialize RAG Chatbot with configurable parameters
        
        Args:
            embedding_model (str): Ollama embedding model
            llm_model (str): Ollama language model
            chunk_size (int): Text chunk size for document splitting
            chunk_overlap (int): Overlap between text chunks
        """
        self.embedding_model = embedding_model
        self.llm = Ollama(model=llm_model, temperature=0.3)
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        
        # Initialize Pinecone
        self.index_name = "hybrid-search-langchain-pinecone"
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        # Create the index if it doesn't exist
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=1024,  
                metric="dotproduct",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        
        # Initialize BM25 Encoder
        self.bm25_encoder = BM25Encoder().default()
    
    def _load_and_split_pdf(self, file_path: str) -> List[Document]:
        """
        Load and split PDF documents
        
        Args:
            file_path (str): Path to the PDF file
        
        Returns:
            List of processed document chunks
        """
        try:
            # Load PDF
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Split documents
            split_docs = RecursiveCharacterTextSplitter(
                chunk_size=500, 
                chunk_overlap=100,
                separators=["\n\n", "\n", " ", ""]
            ).split_documents(documents)
            
            return split_docs
        
        except Exception as e:
            st.error(f"Error processing PDF {file_path}: {e}")
            return []
    
    def create_vector_store(self, files: List[str]) -> Optional[PineconeHybridSearchRetriever]:
            """
            Create Pinecone vector store from uploaded files
            
            Args:
                files (List[str]): List of file paths
            
            Returns:
                PineconeHybridSearchRetriever or None
            """
            all_documents = []
            for file in files:
                all_documents.extend(self._load_and_split_pdf(file))
            
            if not all_documents:
                return None
            
            try:
                # Prepare texts and metadata for Pinecone
                texts = [doc.page_content for doc in all_documents]
                metadatas = [{"source": file} for file in files for _ in all_documents]  # Example metadata
                
                # Add documents to Pinecone
                retriever = PineconeHybridSearchRetriever(
                    embeddings=self.embeddings,
                    sparse_encoder=self.bm25_encoder,
                    index=self.pc.Index(self.index_name)
                )
                
                # Add texts and metadata to the Pinecone index
                retriever.add_texts(texts, metadatas=metadatas)
                
                return retriever
            
            except Exception as e:
                st.error(f"Error creating vector store: {e}")
                return None

    
    def generate_response(self, 
                           query: str, 
                           retriever: PineconeHybridSearchRetriever, 
                           chat_history: List[Dict[str, str]] = None
                           ) -> str:
        """
        Generate response using RAG approach
        
        Args:
            query (str): User query
            retriever (PineconeHybridSearchRetriever): Retriever for document retrieval
            chat_history (List[Dict]): Previous conversation context
        
        Returns:
            Generated response
        """
        # Retrieve relevant documents
        retrieved_docs = retriever.invoke(query)
        
        # Prepare context
        context = "\n\n".join([
            f"Document {i+1}: {doc.page_content}" 
            for i, doc in enumerate(retrieved_docs)
        ])
        
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant specializing in document analysis. 
            Use the provided context to answer questions accurately. 
            If the context doesn't contain enough information, 
            clearly state what additional information is needed.
            
            Context:
            {context}
            
            Chat History:
            {chat_history}"""),
            ("user", "Question: {question}")
        ])
        
        # Prepare chat history
        formatted_history = "\n".join([
            f"Human: {msg['human']}\nAI: {msg['ai']}" 
            for msg in (chat_history or [])
        ])
        
        # Create RAG chain
        chain = prompt | self.llm | StrOutputParser()
        
        # Generate response
        return chain.invoke({
            'context': context, 
            'chat_history': formatted_history,
            'question': query
        })

def main():
    # Streamlit UI Configuration
    st.set_page_config(
        page_title="PDF Chat Assistant", 
        page_icon="📄", 
        layout="wide"
    )
    
    # Custom CSS for chatbot-like interface
    st.markdown("""
    <style>
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        background-color: #f9f9f9;
    }
    .user-message {
        background-color: #e6f2ff;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        text-align: right;
    }
    .ai-message {
        background-color: #f0f0f0;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Title and description
    st.title("📄 PDF Chat Assistant")
    st.markdown("Upload PDFs and chat with your documents!")
    
    # Initialize RAG Chatbot
    rag = RAGChatbot()
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Document Upload")
        uploaded_files = st.file_uploader(
            "Choose PDF files", 
            type=['pdf'], 
            accept_multiple_files=True
        )
        
        # Model selection
        st.subheader("Model Settings")
        embedding_model = st.selectbox(
            "Embedding Model", 
            ["mxbai-embed-large:latest"]
        )
        llm_model = st.selectbox(
            "Language Model", 
            ["llama3:2b"]
        )
    
    # Initialize session state for chat
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None
    
    # Process uploaded files
    if uploaded_files:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_paths = []
            for uploaded_file in uploaded_files:
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                file_paths.append(temp_path)
            
            # Create vector store
            st.session_state.retriever = rag.create_vector_store(file_paths)
            
            if st.session_state.retriever:
                st.sidebar.success("📄 Documents processed successfully!")
    
    # Chat interface
    if st.session_state.retriever:
        # Display chat messages
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents"):
            # Add user message to chat history
            st.session_state.messages.append({
                "role": "user", 
                "content": prompt
            })
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Prepare chat history for context
                    chat_history = [
                        {
                            'human': msg['content'] if msg['role'] == 'user' else None,
                            'ai': msg['content'] if msg['role'] == 'assistant' else None
                        } 
                        for msg in st.session_state.messages
                    ]
                    
                    response = rag.generate_response(
                        prompt, 
                        st.session_state.retriever,
                        chat_history
                    )
                    st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response
            })
    else:
        st.info("👉 Upload PDFs in the sidebar to start chatting!")

if __name__ == "__main__":
    main()
