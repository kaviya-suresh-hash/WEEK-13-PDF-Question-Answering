# PDF Question Answering System Using RAG

## Project Overview
This project implements a PDF Question Answering System using Retrieval-Augmented Generation (RAG). It extracts information from PDF documents, stores document embeddings in ChromaDB, retrieves relevant content, and generates answers using Ollama.

## Technologies Used
- Python 3.10
- LangChain
- ChromaDB
- Sentence Transformers
- Ollama
- Gemma3:1b
- PyPDF
- Pandas
- Jupyter Notebook

## Class Activity
1. Loaded and extracted text from a sample PDF.
2. Split the extracted text into chunks.
3. Generated embeddings using Sentence Transformers.
4. Stored embeddings in ChromaDB.
5. Retrieved relevant document chunks for user questions.
6. Generated answers using Ollama.
7. Executed the complete system in Jupyter Notebook.

## Home Activity

### PDF Details
- Topic: Introduction to Artificial Intelligence
- Total Pages: 17

### Chunk Size Experiments

| Chunk Size | Overlap | Number of Chunks |
|---|---|---|
| 300 | 50 | 92 |
| 500 | 100 | 57 |
| 800 | 150 | 35 |

Selected Chunk Size: 500

Selected Chunk Overlap: 100

### Question Answering
- Prepared 15 questions.
- Retrieved Top-5 relevant chunks for every question.
- Generated answers with RAG.
- Generated answers without RAG.
- Compared both answers.

### Output Files
- queries.csv
- retrieved_chunks.csv
- home_answers.csv
- home_extracted_text.txt

## Observations
1. PDF text extraction was successful.
2. Different chunk sizes produced different numbers of chunks.
3. ChromaDB stored 57 document chunks.
4. Top-5 chunks were retrieved for all 15 questions.
5. RAG answers use retrieved PDF context.
6. Without RAG, answers are generated without document retrieval.
7. Both answers were saved for comparison.

## Project Structure

```text
PDF_Question_Answering/
│
├── dataset/
│   ├── sample.pdf
│   ├── home_activity.pdf
│   ├── extracted_text.txt
│   ├── home_extracted_text.txt
│   ├── chunks.csv
│   ├── queries.csv
│   ├── answers.csv
│   ├── retrieved_chunks.csv
│   └── home_answers.csv
│
├── notebook/
│   └── Week13_PDF_Question_Answering.ipynb
│
├── chroma_db/
├── home_chroma_db/
├── pdf_qa.py
├── home_activity.py
├── README.md
└── requirements.txt

```

## Conclusion
The PDF Question Answering System was successfully implemented using LangChain, ChromaDB, Sentence Transformers and Ollama. The system retrieves relevant information from PDF documents and generates context-based answers using RAG.