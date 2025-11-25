import os
from rag_system import HomeLoanRAGSystem

# ===== EDIT THIS PATH TO YOUR PDF FILE =====
PDF_PATH = r"Sample_RAG_Metadata.pdf"  # Change this to your PDF file name/path
# ============================================

def main():
    print("=== Knowledge Base Ingestion ===")
    
    # Check if PDF file exists
    if not os.path.exists(PDF_PATH):
        print(f"❌ Error: PDF file not found: {PDF_PATH}")
        print("Please make sure the PDF file exists and update the PDF_PATH variable in this script.")
        return
    
    print(f"📄 PDF file: {PDF_PATH}")
    print(f"🔧 Initializing RAG system...")
    
    # Initialize RAG system with Qdrant Cloud
    rag_system = HomeLoanRAGSystem()
    
    if not rag_system.is_initialized():
        print("❌ Error: Failed to initialize RAG system.")
        print("Check that all required packages are installed: pip install -r requirements.txt")
        return
    
    print("✅ RAG system initialized successfully!")
    
    # Ingest the PDF
    print(f"📚 Ingesting PDF: {PDF_PATH}")
    print("This may take a few minutes...")
    
    success = rag_system.ingest_pdf(PDF_PATH)
    
    if success:
        print("✅ PDF ingested successfully!")
        
        # Show collection info
        info = rag_system.get_collection_info()
        print(f"📊 Collection info: {info}")
        
        # Test search
        print("\n🔍 Testing search functionality...")
        test_queries = [
            
"List the top 10 customers with the highest transaction amount.",
"Find all customers who live in Hyderabad and made transactions using UPI.",
"Show the total transaction amount by each payment method."
        ]
        
        for i, query in enumerate(test_queries, 3):
            print(f"\n{i}. Test query: {query}")
            results = rag_system.search_similar_documents(query, top_k=1)
            if results:
                print(f"   ✅ Found {len(results)} relevant documents")
                for j, result in enumerate(results):
                    print(f"      {j+1}. Score: {result['score']:.3f}")
                    print(f"         Preview: {result['content'][:100]}...")
            else:
                print("   ❌ No relevant documents found")

            # Generate and print SQL answer using RAG + LLM
            try:
                sql_answer = rag_system.generate_rag_response(query, top_k=1)
                print("\n   🧠 Generated SQL:\n" + sql_answer)
            except Exception as e:
                print(f"   ❌ Failed to generate SQL: {e}")
        
        print("\n🎉 Setup complete! Your RAG system is ready to use.")
        print("You can now run your Streamlit app: streamlit run main.py")
        
    else:
        print("❌ Failed to ingest PDF")
        print("Check the error messages above for troubleshooting.")

if __name__ == "__main__":
    main()
