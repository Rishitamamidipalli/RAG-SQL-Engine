# SQL RAG Assistant

A powerful Retrieval-Augmented Generation (RAG) system that converts natural language queries into SQL statements using vector embeddings and the HCL AI Cafe API (GPT-4.1).

## 🎯 Features

- **Natural Language to SQL**: Ask questions in plain English, get SQL queries
- **Vector-Based Retrieval**: Uses Qdrant vector database for semantic search
- **RAG Architecture**: Combines document retrieval with LLM generation for accurate SQL
- **HCL AI Integration**: Powered by GPT-4.1 via HCL AI Cafe API
- **Web Interface**: Interactive Streamlit application
- **PDF Knowledge Base**: Ingest PDF documents as your knowledge base

## 🏗️ Architecture

```
User Query
    ↓
Vector Embedding (Sentence Transformers)
    ↓
Semantic Search (Qdrant)
    ↓
Context Retrieval
    ↓
RAG Prompt Construction
    ↓
HCL AI Cafe API (GPT-4.1)
    ↓
SQL Extraction & Validation
    ↓
Generated SQL Query
```

## 📋 Prerequisites

- Python 3.8+
- Qdrant Cloud account
- HCL AI Cafe API credentials
- PDF documents for knowledge base

## 🚀 Installation

### 1. Clone or download the project
```bash
cd USAA
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Create a `.env` file in the project root:
```env
# Qdrant Configuration
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION_NAME=USAA

# HCL AI Cafe Configuration
HCL_API_KEY=your_hcl_api_key

# AWS Configuration (optional)
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_REGION=us-east-1
S3_BUCKET_NAME=your_bucket
S3_PREFIX=your_prefix
```

## 📚 Usage

### 1. Ingest PDF Knowledge Base
```bash
python simple_ingest.py
```

This script will:
- Load your PDF file (`Sample_RAG_Metadata.pdf`)
- Split it into chunks
- Generate embeddings
- Store vectors in Qdrant

### 2. Run the Web Application
```bash
streamlit run App.py
```

The app will open at `http://localhost:8501`

### 3. Query the System
1. Enter a natural language question in the text area
2. Click "Generate SQL"
3. View the generated SQL query
4. Optionally view the retrieved context

## 📁 Project Structure

```
USAA/
├── App.py                    # Streamlit web interface
├── rag_system.py            # Core RAG system implementation
├── simple_ingest.py         # PDF ingestion script
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (not in git)
├── .gitignore              # Git ignore rules
├── README.md               # This file
├── Sample_RAG_Metadata.pdf # Knowledge base PDF
└── api_documentation.txt   # HCL API reference
```

## 🔧 Configuration

### Qdrant Setup
1. Create a Qdrant Cloud account at https://qdrant.tech/
2. Create a new cluster
3. Copy the URL and API key to `.env`

### HCL AI Cafe Setup
1. Get API credentials from HCL AI Cafe
2. Add `HCL_API_KEY` to `.env`
3. The system uses GPT-4.1 model by default

### PDF Knowledge Base
1. Place your PDF file in the project root
2. Update `PDF_PATH` in `simple_ingest.py` if needed
3. Run `python simple_ingest.py` to ingest

## 🎨 How It Works

### 1. Document Ingestion
- PDFs are loaded using LangChain's PyPDFLoader
- Documents are split into chunks (200 tokens with 20 token overlap)
- Embeddings are generated using Sentence Transformers
- Vectors are stored in Qdrant with metadata

### 2. Query Processing
- User query is embedded using the same model
- Semantic search finds top-k relevant documents
- Retrieved context is formatted into a prompt
- HCL AI Cafe API generates SQL based on context

### 3. SQL Extraction
- Response is parsed to extract SQL statements
- Supports multiple extraction strategies:
  - Fenced code blocks (```sql ... ```)
  - Direct SQL keyword detection
  - Fallback to best-effort parsing

## 📊 Example Queries

```
"List all transactions made using Credit Card"
→ SELECT * FROM Transaction_History WHERE Payment_Method = 'Credit Card';

"Find customers who live in Hyderabad"
→ SELECT * FROM Customer_Info WHERE City = 'Hyderabad';

"Show total transaction amount by payment method"
→ SELECT Payment_Method, SUM(Transaction_Amount) FROM Transaction_History GROUP BY Payment_Method;
```

## 🐛 Troubleshooting

### HCL API Error
- Verify `HCL_API_KEY` is set in `.env`
- Check API key is valid and has active credits
- Ensure network connectivity to `aicafe.hcl.com`

### Qdrant Connection Error
- Verify `QDRANT_URL` and `QDRANT_API_KEY` are correct
- Check Qdrant Cloud cluster is running
- Ensure collection exists or will be auto-created

### PDF Ingestion Failed
- Verify PDF file path is correct
- Check PDF is not corrupted
- Ensure sufficient disk space

### No Context Retrieved
- Verify PDF was successfully ingested
- Check Qdrant collection has vectors
- Try different query phrasing

## 📝 Logging

The system logs detailed information to help with debugging:
- Check terminal output for Streamlit app logs
- Check console for ingestion script logs
- All errors are logged with timestamps

## 🔐 Security

- Never commit `.env` file to git
- Keep API keys confidential
- Use environment variables for all secrets
- Review `.gitignore` before committing

## 🤝 Contributing

To extend this project:

1. **Add new LLM providers**: Modify `_generate_with_hcl()` pattern
2. **Improve SQL extraction**: Enhance `_extract_sql()` method
3. **Add database support**: Extend schema context retrieval
4. **Optimize embeddings**: Try different Sentence Transformer models

## 📚 Dependencies

- **streamlit**: Web framework
- **qdrant-client**: Vector database client
- **sentence-transformers**: Embedding generation
- **langchain**: Document loading and processing
- **requests**: HTTP client for HCL API
- **python-dotenv**: Environment variable management

## 📄 License

This project is part of the USAA system.

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section
2. Review logs for error messages
3. Verify all environment variables are set
4. Check API credentials and quotas

## 🎯 Future Enhancements

- [ ] Support for multiple document formats (CSV, JSON, SQL dumps)
- [ ] Query caching and optimization
- [ ] Multi-language support
- [ ] Advanced SQL validation
- [ ] Query execution and result display
- [ ] User authentication
- [ ] Query history and analytics
- [ ] Custom embedding models
- [ ] Batch query processing

---

**Last Updated**: November 2025
**Version**: 1.0.0
