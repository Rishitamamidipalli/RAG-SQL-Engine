# RAG SQL Engine

A sophisticated Retrieval-Augmented Generation (RAG) system that converts natural language queries into accurate SQL statements using comprehensive database schema knowledge and HCL AI Cafe API (GPT-4.1).

## 🚀 Features

- **Natural Language to SQL**: Convert plain English questions into accurate SQL queries
- **Multi-PDF Knowledge Base**: Comprehensive database schema across 4 PDFs with 11 tables
- **Enhanced Vector Retrieval**: Optimized chunking and search for complete table schemas
- **HCL AI Integration**: Uses GPT-4.1 via HCL AI Cafe API for intelligent SQL generation
- **Interactive Web Interface**: Streamlit app with context visualization
- **Cross-Table Queries**: Handles complex joins and relationships automatically

## 📊 Knowledge Base Structure

The system includes **4 comprehensive PDFs** with **11 interconnected database tables**:

### PDF Files
1. **Sample_RAG_Metadata.pdf** - Basic customer and transaction tables
2. **01_Customer_Management.pdf** - Customer profiles, preferences, and segments  
3. **02_Transaction_Management.pdf** - Transaction details, payment methods, and merchants
4. **03_Analytics_and_Reporting.pdf** - Analytics, behavior metrics, and business KPIs

### Database Tables (11 Total)
- **Customer_Info** - Customer profiles and demographics
- **Customer_Preferences** - Communication preferences and settings
- **Customer_Segments** - Customer segmentation and lifetime value
- **Transaction_History** - Detailed transaction records
- **Payment_Methods** - Customer payment method details
- **Merchant_Master** - Merchant information and statistics
- **Customer_Spending_Analysis** - Monthly spending patterns
- **Category_Spending_Breakdown** - Spending by merchant category
- **Customer_Behavior_Metrics** - Behavioral and engagement metrics
- **Business_Metrics** - Overall business KPIs

## 🏗️ Architecture

```
Natural Language Query
    ↓
Vector Embedding (distiluse-base-multilingual-cased-v2)
    ↓
Enhanced Semantic Search (Qdrant Cloud)
    ↓
Multi-Document Context Retrieval (5 chunks)
    ↓
RAG Prompt Construction
    ↓
HCL AI Cafe API (GPT-4.1)
    ↓
SQL Generation & Validation
    ↓
Clean SQL Query Output
```

## 📋 Prerequisites

- Python 3.8+
- Qdrant Cloud account and API key
- HCL AI Cafe API credentials
- Knowledge base PDFs (included in project)

## 🛠️ Setup

### 1. Environment Setup
```bash
cd USAA
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the project root:
```env
# Qdrant Cloud Configuration
QDRANT_URL=https://your-cluster-url.us-east-1-0.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_COLLECTION_NAME=USAA

# HCL AI Cafe API Configuration
HCL_API_KEY=your-hcl-api-key
```

### 3. Ingest Knowledge Base
The system automatically processes all PDFs in the `knowledge_base/` folder:
```bash
python simple_ingest.py
```

This will:
- Load all 4 PDFs with optimized chunking (1000 chars, 200 overlap)
- Create vector embeddings using sentence transformers
- Store them in Qdrant Cloud with metadata
- Test the system with sample queries

### 4. Launch the Application
```bash
streamlit run App.py
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

## 🎯 Usage

1. **Open the Streamlit interface** in your browser
2. **Check the sidebar** for system status and vector count
3. **Enter a natural language question** about your data
4. **View results**: The system displays both generated SQL and retrieved context snippets

## 📝 Example Queries

### Simple Queries
```
"Show all customers with their email and city information"
→ SELECT Customer_Name, Email, City FROM Customer_Info;

"List transactions made using UPI payment method"  
→ SELECT * FROM Transaction_History WHERE Payment_Method = 'UPI';

"Find customers who belong to the Premium segment"
→ SELECT ci.* FROM Customer_Info ci JOIN Customer_Segments cs ON ci.Customer_ID = cs.Customer_ID WHERE cs.Segment_Category = 'Premium';
```

### Complex Analytics Queries
```
"Show total transaction amount grouped by merchant category"
→ SELECT mm.Category, SUM(th.Transaction_Amount) FROM Transaction_History th JOIN Merchant_Master mm ON th.Merchant_ID = mm.Merchant_ID GROUP BY mm.Category;

"List customers with high churn probability"
→ SELECT cbm.Customer_ID, cbm.Risk_Score FROM Customer_Behavior_Metrics cbm WHERE cbm.Risk_Score >= 70;

"Find VIP customers who are subscribed to newsletters"
→ SELECT ci.Customer_Name FROM Customer_Info ci JOIN Customer_Preferences cp ON ci.Customer_ID = cp.Customer_ID JOIN Customer_Segments cs ON ci.Customer_ID = cs.Customer_ID WHERE cp.Newsletter_Subscription = TRUE AND cs.Segment_Category = 'VIP';
```

## 📁 Project Structure

```
USAA/
├── rag_system.py              # Core RAG SQL Engine (RAGSQLEngine class)
├── simple_ingest.py           # Automated PDF ingestion script  
├── App.py                     # Streamlit web interface
├── knowledge_base/            # PDF documents folder
│   ├── Sample_RAG_Metadata.pdf
│   ├── 01_Customer_Management.pdf
│   ├── 02_Transaction_Management.pdf
│   └── 03_Analytics_and_Reporting.pdf
├── requirements.txt           # Python dependencies
├── .env                      # Environment variables (create this)
├── .gitignore               # Git ignore rules
└── README.md                # This documentation
```

## ⚙️ Key Configuration

### Enhanced Chunking Strategy
- **Chunk Size**: 1000 characters (captures complete table schemas)
- **Overlap**: 200 characters (maintains context continuity)  
- **Separators**: Prioritizes headings (`\n## `, `\n### `) then paragraphs

### Search Parameters
- **Retrieval**: 5 chunks per query (configurable in App.py)
- **No Score Threshold**: Includes all potentially relevant chunks
- **Deduplication**: Removes very similar chunks based on content hash

### API Configuration
- **Model**: gpt-4.1 (HCL AI Cafe)
- **Temperature**: 0.1 (deterministic SQL generation)
- **Max Tokens**: 500 (sufficient for complex queries)

## 🐛 Troubleshooting

### HCL API Issues
- ✅ Verify `HCL_API_KEY` is set in `.env`
- ✅ Check network connectivity to `aicafe.hcl.com`
- ✅ Ensure API endpoint format: `https://aicafe.hcl.com/AICafeService/api/v1/subscription/openai/deployments/gpt-4.1/chat/completions`

### Qdrant Connection Issues
- ✅ Verify `QDRANT_URL` and `QDRANT_API_KEY` are correct
- ✅ Check Qdrant Cloud cluster is running
- ✅ Collection `USAA` will be auto-created during ingestion

### Limited Context Retrieval
- ✅ Check sidebar for "Vectors in DB" count
- ✅ Re-run `python simple_ingest.py` if vectors are 0
- ✅ Try different query phrasing for better matches

## 🚀 Getting Started

### Quick Start Guide
1. **Setup environment and install dependencies**
2. **Configure `.env` with your API keys**
3. **Run ingestion**: `python simple_ingest.py`
4. **Launch app**: `streamlit run App.py`
5. **Start querying with natural language!**

### Expected Results
- **Vector Database**: ~50-100 chunks from 4 PDFs
- **Query Response Time**: 2-5 seconds
- **Context Retrieval**: 5 relevant chunks per query
- **SQL Accuracy**: High accuracy for schema-covered queries

## 📚 Key Dependencies

- **streamlit**: Interactive web framework
- **qdrant-client**: Vector database client  
- **sentence-transformers**: Embedding generation (distiluse-base-multilingual-cased-v2)
- **langchain**: Document loading and text splitting
- **requests**: HTTP client for HCL API calls
- **python-dotenv**: Environment variable management

## 🔐 Security & Best Practices

- ✅ Never commit `.env` file to version control
- ✅ Keep API keys confidential and rotate regularly
- ✅ Use environment variables for all sensitive configuration
- ✅ Review `.gitignore` before committing code

## 🎯 System Capabilities

- **Multi-Document Retrieval**: Searches across all 4 PDFs seamlessly
- **Complex Join Generation**: Creates accurate multi-table SQL queries
- **Schema-Aware Processing**: Understands table relationships and foreign keys
- **Context Visualization**: Shows retrieved chunks for transparency
- **Real-time Processing**: Fast vector search and SQL generation

## 🆘 Support & Debugging

### Common Issues
1. **"Collection USAA has None vectors"** → Re-run `python simple_ingest.py`
2. **"HCL API connection failed"** → Check `HCL_API_KEY` and network connectivity
3. **"Only getting 1 snippet"** → Check vector count in sidebar, may need re-ingestion

### Debug Logs to Check
- `INFO:rag_system:Found X unique chunks for query...`
- `INFO:rag_system:Collection USAA has X vectors`
- `INFO:rag_system:Generating SQL using HCL API`

---

## 📊 Project Stats

- **Total Tables**: 11 interconnected database tables
- **Total PDFs**: 4 comprehensive schema documents  
- **Vector Chunks**: ~50-100 optimized chunks
- **Query Types**: Simple selects to complex multi-table joins
- **Response Time**: 2-5 seconds end-to-end

**Status**: ✅ Production Ready  
**Last Updated**: November 2025  
**Version**: 2.0.0 (RAG SQL Engine)
