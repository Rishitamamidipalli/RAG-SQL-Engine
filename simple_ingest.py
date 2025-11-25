import os
from pathlib import Path
from rag_system import RAGSQLEngine

# ===== KNOWLEDGE BASE CONFIGURATION =====
KB_FOLDER = r"knowledge_base"  # Folder containing PDF files
# =========================================

def main():
    print("=" * 60)
    print("Knowledge Base Ingestion System")
    print("=" * 60)
    
    # Initialize RAG system with Qdrant Cloud
    print(f"\n🔧 Initializing RAG system...")
    rag_system = RAGSQLEngine()
    
    if not rag_system.is_initialized():
        print("❌ Error: Failed to initialize RAG system.")
        print("Check that all required packages are installed: pip install -r requirements.txt")
        return
    
    print("✅ RAG system initialized successfully!")
    
    # Get all PDFs from knowledge_base folder
    kb_path = Path(KB_FOLDER)
    if not kb_path.exists():
        print(f"❌ Error: {KB_FOLDER} folder not found!")
        print(f"   Please create the folder and add PDF files to it.")
        return
    
    pdf_files = sorted(kb_path.glob("*.pdf"))
    pdfs_to_ingest = [str(f) for f in pdf_files]
    
    if not pdfs_to_ingest:
        print(f"❌ Error: No PDF files found in {KB_FOLDER} folder!")
        print(f"   Please add PDF files to the {KB_FOLDER} folder.")
        return
    
    print(f"\n📚 Found {len(pdfs_to_ingest)} PDF files to ingest:")
    for pdf in pdfs_to_ingest:
        print(f"   - {pdf}")
    
    # Ingest all PDFs
    print(f"\n🔄 Starting ingestion process...")
    print("This may take a few minutes...\n")
    
    successful_ingestions = 0
    for pdf_file in pdfs_to_ingest:
        if not os.path.exists(pdf_file):
            print(f"⚠️  Skipping {pdf_file} - file not found")
            continue
        
        print(f"📄 Ingesting: {pdf_file}")
        success = rag_system.ingest_pdf(pdf_file)
        
        if success:
            print(f"✅ Successfully ingested: {pdf_file}\n")
            successful_ingestions += 1
        else:
            print(f"❌ Failed to ingest: {pdf_file}\n")
    
    # Show final collection info
    print("\n" + "=" * 60)
    print("Collection Statistics")
    print("=" * 60)
    info = rag_system.get_collection_info()
    if "error" not in info:
        print(f"✅ Collection Name: {info.get('name', 'N/A')}")
        print(f"📊 Total Vectors: {info.get('vectors_count', 0)}")
        print(f"📈 Status: {info.get('status', 'N/A')}")
    else:
        print(f"❌ Error: {info['error']}")
    
    # Test search and SQL generation
    print("\n" + "=" * 60)
    print("Testing RAG System")
    print("=" * 60)
    
    test_queries = [
        "Show all customers with their email and city information",
        "List transactions made using UPI payment method",
        "Find customers who belong to the Premium segment",
        "Show total transaction amount grouped by merchant category",
        "List customers with high churn probability from Customer_Behavior_Metrics",
    ]
    
    print(f"\n🔍 Running {len(test_queries)} test queries...\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"{i}. Query: {query}")
        
        # Search for relevant documents
        results = rag_system.search_similar_documents(query, top_k=2)
        if results:
            print(f"   ✅ Found {len(results)} relevant documents")
            for j, result in enumerate(results, 1):
                print(f"      {j}. Score: {result['score']:.3f} | Source: {result['source']}")
        else:
            print("   ⚠️  No relevant documents found")
        
        # Generate SQL
        try:
            sql_answer = rag_system.generate_rag_response(query, top_k=2)
            print(f"   🧠 Generated SQL:\n      {sql_answer}")
        except Exception as e:
            print(f"   ❌ Failed to generate SQL: {e}")
        
        print()
    
    # Final summary
    print("=" * 60)
    if successful_ingestions > 0:
        print(f"✅ Successfully ingested {successful_ingestions}/{len(pdfs_to_ingest)} PDFs")
        print("🎉 Your RAG system is ready to use!")
        print("\n📝 Next steps:")
        print("   1. Run the Streamlit app: streamlit run App.py")
        print("   2. Ask natural language questions")
        print("   3. Get SQL queries generated from your knowledge base")
    else:
        print("❌ No PDFs were successfully ingested")
        print("Please check the error messages above for troubleshooting")
    print("=" * 60)

if __name__ == "__main__":
    main()
