# Retrieval-Augmented Generation (RAG) System 
**Project Overview**

This project implements a Retrieval-Augmented Generation (RAG) system using a local Large Language Model (LLM) and a vector database. The system allows users to ask questions about a collection of business, AI, cybersecurity, GDPR, HR, and Generative AI documents and receive context-aware answers.

The objective is to reduce hallucinations and improve answer accuracy by retrieving relevant information from uploaded documents before generating responses.

Documents Used

The knowledge base consists of the following documents:

  1. Artificial Intelligence, Machine Learning and Deep Learning
  2. Oxford IB Diploma Programme Business Management Course Companion (2014 Edition)
  3. GDPR and Cyber Security for Business Information Systems
  4. Indian Institute of Management Human Resource Policy Manual Staff 2023
  5. Snowflake Special Edition Generative AI and LLMs For Dummies

**Features**

  - Document ingestion and processing
  - Text chunking and embedding generation
  - Vector similarity search
  - Retrieval-Augmented Generation (RAG)
  - Local LLM inference using Ollama
  - Context-aware question answering
  - Business and AI knowledge retrieval

**System Architecture**
  1. Documents are loaded and processed.
  2. Documents are split into smaller chunks.
  3. Embeddings are generated for each chunk.
  4. Embeddings are stored in a vector database.
  5. User submits a question.
  6. Relevant document chunks are retrieved.
  7. Retrieved context is passed to the LLM.
  8. The LLM generates an answer based on the retrieved information.

**Technologies Used**

Python,
LangChain,
Ollama,
ChromaDB,
Local Large Language Models,
Retrieval-Augmented Generation (RAG).

**Example Questions**
1. What is Artificial Intelligence?
2. What is SWOT analysis?
3. What are the principles of GDPR?
4. What is employee onboarding?
5. What is prompt engineering?
6. How does a Large Language Model work?
7. What is the difference between Machine Learning and Deep Learning?

**Running the Project**(Everything is Bash)

Install Dependencies

    pip install -r requirements.txt

Start Ollama

    ollama serve

Run the RAG System

    python3 RAG.py

**Evaluation**

The system is evaluated using 20 test questions across multiple document domains.

Evaluation criteria include:
  - Retrieval relevance
  - Answer accuracy
  - Completeness
  - Context grounding

**Learning Outcomes**

Through this project, the following concepts were explored:
  - Retrieval-Augmented Generation
  - Vector databases
  - Semantic search
  - Prompt engineering
  - Local LLM deployment
  - AI-powered knowledge management systems

