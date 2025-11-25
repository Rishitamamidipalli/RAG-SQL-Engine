import os
import logging
# import atexit
import threading
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv
import requests
import re

import streamlit as st

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


LANGCHAIN_AVAILABLE = True
import uuid

load_dotenv()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HomeLoanRAGSystem:
    """RAG system for SQL-focused querying using a Qdrant vector database"""
    
    _instance = None
    _lock = threading.Lock()
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern to prevent multiple instances"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(HomeLoanRAGSystem, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, 
                 qdrant_url: str = os.getenv("QDRANT_URL"),
                 collection_name: str = os.getenv("QDRANT_COLLECTION_NAME"),
                 api_key: str = os.getenv("QDRANT_API_KEY"),
                 embedding_model: str = "sentence-transformers/distiluse-base-multilingual-cased-v2"
):
        """
        Initialize the RAG system
        
        Args:
            qdrant_url: URL for Qdrant Cloud instance
            collection_name: Name of the collection in Qdrant
            api_key: API key for Qdrant Cloud authentication
            embedding_model: Sentence transformer model for embeddings
        """
        # Only initialize once
        if self._initialized:
            return
            
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.api_key = api_key
        self.embedding_model_name = embedding_model
        # HCL API configuration
        self.hcl_api_key = os.getenv("HCL_API_KEY")
        self.hcl_deployment_name = "gpt-4.1"  # As specified in the documentation
        self.hcl_api_version = "2024-12-01-preview"  # As specified in the documentation
        
        # Initialize components
        self.client = None
        self.embedding_model = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,  # Increased for better context
            chunk_overlap=20,  # Proportional overlap
            length_function=lambda x: len(x.split()), 
            separators=["\n\n", "\n", ".", "!", "?"]  
)
        
        self._initialize_components()
        
        # Register cleanup function
        # atexit.register(self._cleanup)
        
        # Mark as initialized
        self._initialized = True
    
    def _initialize_components(self):
        """Initialize Qdrant client and embedding model"""
        try:
            # Clean up any existing client first
            # self._cleanup_client()
            
            # Initialize Qdrant Cloud client
            self.client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.api_key,
                prefer_grpc=True  # Recommended for better performance with Qdrant Cloud
            )
            logger.info(f"Connected to Qdrant Cloud at {self.qdrant_url}")
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Loaded embedding model: {self.embedding_model_name}")
            
            # Create collection if it doesn't exist
            self._create_collection_if_not_exists()
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG components: {str(e)}")
            # self._cleanup_client()
            self.client = None
            self.embedding_model = None

    def _generate_with_hcl(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000) -> Optional[str]:
        """Call HCL's AI Cafe API to generate text for the given prompt.
        
        Uses HCL_API_KEY from environment variables.
        """
        if not self.hcl_api_key:
            logger.error("HCL_API_KEY is not set in environment")
            return None
            
        try:
            # HCL API endpoint
            url = f"https://aicafe.hcl.com/AICafeService/api/v1/subscription/openai/deployments/{self.hcl_deployment_name}/chat/completions"
            
            # Query parameters
            params = {
                "api-version": self.hcl_api_version
            }
            
            # Headers
            headers = {
                "Content-Type": "application/json",
                "api-key": self.hcl_api_key
            }
            
            # Request body
            data = {
                "model": self.hcl_deployment_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a SQL generation assistant that only outputs SQL code."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "maxTokens": max_tokens,
                "temperature": temperature
            }
            
            # Make the request
            logger.info(f"Calling HCL AI Cafe API with model: {self.hcl_deployment_name}")
            response = requests.post(url, headers=headers, json=data, params=params, timeout=60)
            
            if response.status_code != 200:
                logger.error(f"HCL API error {response.status_code}: {response.text}")
                return None
                
            # Parse response
            response_data = response.json()
            
            # Extract content from response
            if response_data and "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0].get("message", {}).get("content", "")
                return content.strip()
            else:
                logger.error("HCL API returned unexpected response format")
                logger.debug(f"Response: {response_data}")
                return None
                
        except Exception as e:
            logger.error(f"HCL API request failed: {e}")
            return None
            

    def _extract_sql(self, text: str) -> Optional[str]:
        """Extract only SQL from a model response.

        Strategies:
        - Prefer fenced code blocks ```sql ... ```
        - Otherwise, find first SQL keyword (SELECT|WITH|INSERT|UPDATE|DELETE|CREATE)
          and return until the last semicolon or end of string.
        """
        if not text:
            return None
        # 1) Fenced SQL block
        m = re.search(r"```sql\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            content = m.group(1).strip()
            return content if content else None
        # 2) Any fenced code block
        m = re.search(r"```\s*(.*?)```", text, flags=re.DOTALL)
        if m:
            content = m.group(1).strip()
            # Best-effort: ensure it starts with SQL-ish keyword
            if re.match(r"^(SELECT|WITH|INSERT|UPDATE|DELETE|CREATE)\b", content, flags=re.IGNORECASE):
                return content
        # 3) First SQL keyword occurrence
        m = re.search(r"\b(SELECT|WITH|INSERT|UPDATE|DELETE|CREATE)\b", text, flags=re.IGNORECASE)
        if m:
            start = m.start()
            candidate = text[start:].strip()
            # Trim to last semicolon if present
            semi = candidate.rfind(";")
            if semi != -1:
                candidate = candidate[: semi + 1]
            return candidate.strip()
        return None

    def _create_collection_if_not_exists(self):
        """Create Qdrant collection if it doesn't exist"""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                # Get embedding dimension
                sample_embedding = self.embedding_model.encode(["sample text"])
                embedding_dim = len(sample_embedding[0])
                
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            logger.error(f"Failed to create collection: {str(e)}")
    
    def ingest_pdf(self, pdf_path: str) -> bool:
        """
        Ingest a PDF document into the vector database
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.client or not self.embedding_model:
            logger.error("RAG system not properly initialized")
            return False
        
        try:
            # Load PDF
            if not os.path.exists(pdf_path):
                logger.error(f"PDF file not found: {pdf_path}")
                return False
            
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} pages from PDF")
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Split into {len(chunks)} chunks")
            
            # Generate embeddings and store in Qdrant
            points = []
            for i, chunk in enumerate(chunks):
                # Generate embedding
                embedding = self.embedding_model.encode(chunk.page_content).tolist()
                # logger.debug(f"Generated embedding for chunk {i}, dim: {len(embedding)}")
                # Create point
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "content": chunk.page_content,
                        "source": pdf_path,
                        "page": chunk.metadata.get("page", 0),
                        "chunk_index": i
                    }
                )
                points.append(point)
            
            # Upload to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Successfully ingested {len(points)} chunks from {pdf_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to ingest PDF: {str(e)}")
            return False
    
    def search_similar_documents(self, query: str, top_k: int = 5) -> List[dict]:
        """
        Search for similar documents based on query
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of similar documents with content and metadata
        """
        if not self.client or not self.embedding_model:
            logger.error("RAG system not properly initialized")
            return []
        
        # Debug: Check collection info
        try:
            collection_info = self.client.get_collection(self.collection_name)
            logger.info(f"Collection {self.collection_name} has {collection_info.vectors_count} vectors")
        except Exception as e:
            logger.error(f"Collection {self.collection_name} not found or error: {e}")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Search in Qdrant
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True
            )
            
            # Format results
            results = []
            for result in search_results:
                results.append({
                    "content": result.payload["content"],
                    "source": result.payload["source"],
                    "page": result.payload.get("page", 0),
                    "score": result.score,
                    "chunk_index": result.payload.get("chunk_index", 0)
                })
            
            logger.info(f"Found {len(results)} similar documents for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search documents: {str(e)}")
            return []
    
    def generate_rag_response(self, query: str, top_k: int = 3) -> str:
        """
        Generate SQL response using RAG with HCL API - retrieve relevant documents and generate SQL
        
        Args:
            query: User query
            top_k: Number of relevant documents to retrieve
            
        Returns:
            Generated SQL based on retrieved documents
        """
        try:
            # Retrieve relevant documents
            relevant_docs = self.search_similar_documents(query, top_k=top_k)
            
            if not relevant_docs:
                return "-- Unable to find sufficient context in the knowledge base to generate SQL."
            
            # Prepare context from retrieved documents
            context = "\n\n".join([doc["content"] for doc in relevant_docs])
            
            # Create detailed SQL-focused RAG prompt for HCL API
            rag_prompt = f"""You are a SQL generation assistant. Use the provided database schema context to generate a single SQL query that answers the user's request.

Rules:
- Output only the SQL statement, no explanations or markdown.
- Use ANSI SQL compatible with PostgreSQL.
- Prefer table and column names exactly as shown in the context.
- If the context lacks exact names, make minimal, sensible assumptions and include them as SQL comments at the top like `-- Assumption: ...`.
- If multiple interpretations are possible, choose the most likely one based on the context.

Database Schema Context:
{context}

User Request: {query}

Generate the SQL query only, without any explanations:"""

            # Generate SQL using HCL API
            if not self.hcl_api_key:
                logger.error("HCL_API_KEY is not set in environment")
                return "-- Error: HCL_API_KEY is not configured in .env file"
                
            logger.info("Generating SQL using HCL API")
            response = self._generate_with_hcl(rag_prompt)
            if response:
                sql_only = self._extract_sql(response)
                if sql_only:
                    return sql_only
                # If model ignored instructions, coerce best-effort
                coerced = self._extract_sql("```\n" + response + "\n```")
                if coerced:
                    return coerced
                    
            # If we get here, HCL API failed to generate valid SQL
            logger.error("HCL API failed or returned no valid SQL")
            return "-- Error: Failed to generate SQL. Please try rephrasing the question."
            
        except Exception as e:
            logger.error(f"Failed to generate RAG response: {str(e)}")
            return "-- Error generating SQL. Please try again later."
    
    def is_initialized(self) -> bool:
        """Check if the RAG system is properly initialized"""
        return self.client is not None and self.embedding_model is not None
    
    
    def get_collection_info(self) -> dict:
        """Get information about the current collection"""
        if not self.client:
            return {"error": "Client not initialized"}
        
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": collection_info.vectors_count,
                "status": collection_info.status
            }
        except Exception as e:
            return {"error": str(e)}
