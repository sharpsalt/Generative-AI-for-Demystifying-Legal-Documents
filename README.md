# Generative AI for Demystifying Legal Documents

A comprehensive Retrieval-Augmented Generation (RAG) system designed to make legal documents more accessible and understandable through intelligent document analysis and question-answering capabilities.

## Overview

This project implements a hybrid RAG system that combines vector search (ChromaDB) and graph database (Neo4j) technologies to provide intelligent analysis of legal documents, with a specific focus on insurance policies. The system uses advanced embedding models and language models to extract, process, and query legal document content.

## Features

- **Dual Database Architecture**: Combines ChromaDB for vector similarity search and Neo4j for graph-based relationships
- **Advanced Document Processing**: PDF text extraction with intelligent chunking strategies
- **Hybrid Retrieval**: Leverages both vector similarity and graph traversal for comprehensive document retrieval
- **Legal Document Analysis**: Specialized for insurance policy analysis with structured response generation
- **Scalable Architecture**: Supports multiple document types and can be extended for various legal domains

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PDF Documents │───▶│  Text Processor │───▶│   Embedding     │
└─────────────────┘    └─────────────────┘    │    Generator    │
                                              └─────────────────┘
                                                       │
                       ┌─────────────────┐            │
                       │    ChromaDB     │◀───────────┘
                       │ (Vector Store)  │
                       └─────────────────┘
                                │
                       ┌─────────────────┐
                       │     Neo4j       │
                       │ (Graph Database)│
                       └─────────────────┘
                                │
                       ┌─────────────────┐
                       │ Language Model  │
                       │   (Phi-3 Mini)  │
                       └─────────────────┘
```

## Technology Stack

- **Embedding Model**: Qwen/Qwen3-Embedding-0.6B
- **Language Model**: Microsoft Phi-3-mini-128k-instruct
- **Vector Database**: ChromaDB
- **Graph Database**: Neo4j
- **Document Processing**: PyPDF, LangChain
- **Framework**: PyTorch, Transformers, Sentence-Transformers

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/sharpsalt/Generative-AI-for-Demystifying-Legal-Documents.git
cd Generative-AI-for-Demystifying-Legal-Documents
```

2. **Create and activate conda environment**
```bash
conda create -n legal_rag python=3.10
conda activate legal_rag
```


3. **Install dependencies**
```bash
pip install torch torchvision torchaudio
pip install transformers sentence-transformers
pip install chromadb neo4j
pip install pypdf langchain
pip install numpy pandas jupyter
```

4. **Set up Neo4j Database**
   - Create a Neo4j Aura instance or local installation
   - Note down your connection credentials

5. **Configure environment variables**
```bash
export NEO4J_URI="your_neo4j_uri"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="your_password"
export HF_TOKEN="your_huggingface_token"  # Optional
```

## Usage

### Basic Setup

```python
from insurance_rag_system import InsuranceRAGSystem

# Initialize the system
rag_system = InsuranceRAGSystem(
    neo4j_uri="neo4j+s://your-instance.databases.neo4j.io",
    neo4j_username="neo4j",
    neo4j_password="your_password",
    hf_token="your_hf_token"  # Optional
)
```

### Document Processing

```python
# Process PDF documents
pdf_files = ['path/to/insurance_policy.pdf', 'path/to/legal_document.pdf']
rag_system.process_documents(pdf_files, chunk_size=256, chunk_overlap=32)
```

### Querying the System

```python
# Ask questions about your documents
queries = [
    "What is the definition of Burglary in the insurance policy?",
    "What are the exclusions under trip cancellation benefits?",
    "Who qualifies as a family member under this policy?"
]

for query in queries:
    response = rag_system.query(query, retrieval_method="hybrid", k=3)
    print(f"Query: {query}")
    print(f"Response: {response}")
```

### Retrieval Methods

The system supports three retrieval methods:

1. **ChromaDB Only**: `retrieval_method="chromadb"`
2. **Neo4j Only**: `retrieval_method="neo4j"`
3. **Hybrid**: `retrieval_method="hybrid"` (recommended)

## Configuration Options

### Model Configuration

```python
rag_system = InsuranceRAGSystem(
    # Database settings
    neo4j_uri="your_neo4j_uri",
    neo4j_username="neo4j",
    neo4j_password="your_password",
    neo4j_database="neo4j",
    
    embedding_model="Qwen/Qwen3-Embedding-0.6B",
    llm_model="microsoft/Phi-3-mini-128k-instruct",
    
    chromadb_path="doc_db",
    
    # Authentication
    hf_token="your_hf_token"
)
```

### Document Processing Parameters

```python
rag_system.process_documents(
    pdf_files=["document.pdf"],
    chunk_size=512,        # Larger chunks for more context
    chunk_overlap=64       # More overlap for better continuity
)
```

## Project Structure

```
legal-document-rag/
├── README.md
├── final(1).ipynb                    # Main notebook with implementation
├── insurance_rag_system.py           # Core RAG system class
├── requirements.txt                  # Python dependencies
├── data/                            # Document storage
│   └── sample_policies/
├── doc_db/                          # ChromaDB persistence
├── configs/                         # Configuration files
└── examples/                        # Usage examples
```

## Key Components

### 1. Document Processing Pipeline
- PDF text extraction using PyPDF
- Intelligent text chunking with RecursiveCharacterTextSplitter
- Embedding generation using Qwen3-Embedding model

### 2. Dual Storage System
- **ChromaDB**: Stores document embeddings for fast similarity search
- **Neo4j**: Maintains document relationships and enables graph traversal

### 3. Hybrid Retrieval Engine
- Combines vector similarity scores with graph-based relationships
- Supports configurable retrieval strategies
- Automatic result deduplication and ranking

### 4. Language Model Integration
- Uses Microsoft Phi-3-mini for response generation
- Structured prompt engineering for legal document analysis
- JSON-formatted responses with proper citations

## Performance Optimization

- **GPU Acceleration**: Automatically detects and uses CUDA when available
- **Batch Processing**: Efficient embedding generation for large document sets
- **Memory Management**: Optimized model loading and inference
- **Persistent Storage**: ChromaDB persistence for faster startup times

## Legal Document Support

Currently optimized for:
- Insurance policies
- Travel insurance documents
- Policy terms and conditions
- Claims procedures

Can be extended for:
- Contracts
- Legal agreements
- Regulatory documents
- Compliance materials

### Core Methods

#### `process_documents(pdf_paths, chunk_size=256, chunk_overlap=32)`
Processes PDF documents and stores them in both databases.

#### `query(query_text, retrieval_method="hybrid", k=3)`
Queries the system and returns intelligent responses.

#### `retrieve_from_chromadb(query_text, k=3)`
Retrieves documents using vector similarity search.

#### `retrieve_from_neo4j(query_text, k=3)`
Retrieves documents using graph-based search.

#### `hybrid_retrieve(query_text, k=3)`
Combines both retrieval methods for optimal results.

## Troubleshooting

### Common Issues

1. **Neo4j Connection Failed**
   - Verify your Neo4j credentials and URI
   - Check network connectivity
   - Ensure Neo4j instance is running

2. **CUDA Out of Memory**
   - Reduce batch size in embedding generation
   - Use CPU for embedding model: `device="cpu"`

3. **Model Download Issues**
   - Ensure stable internet connection
   - Verify Hugging Face token if using gated models
   - Check available disk space

### Performance Tips

- Use SSD storage for ChromaDB persistence
- Allocate sufficient GPU memory
- Process documents in batches for large datasets
- Consider using smaller embedding models for faster inference

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

```bibtex
@software{legal_document_rag,
  title={Generative AI for Demystifying Legal Documents},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/legal-document-rag}
}
```

## Acknowledgments

- Qwen team for the embedding model
- Microsoft for the Phi-3 language model
- ChromaDB and Neo4j communities
- LangChain for document processing utilities

## Contact

For questions and support, please open an issue on GitHub or contact [srijanv0@gmail,com].

---

**Disclaimer**: This system is designed to assist with legal document analysis but should not be used as a substitute for professional legal advice. Always consult with qualified legal professionals for important legal matters.
