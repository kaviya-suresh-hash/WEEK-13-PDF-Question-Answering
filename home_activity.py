import os
import shutil
import pandas as pd
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load Home Activity PDF
pdf_path = "dataset/home_activity.pdf"
reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text + "\n"

print("Home Activity PDF Extracted Successfully!")
print("Total Pages:", len(reader.pages))
print("Total Characters:", len(text))

# Save extracted text
with open("dataset/home_extracted_text.txt", "w", encoding="utf-8") as file:
    file.write(text)

print("Extracted Text Saved Successfully!")

# Experiment with different chunk sizes
splitter_300 = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=50
)
chunks_300 = splitter_300.split_text(text)

splitter_500 = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100
)
chunks_500 = splitter_500.split_text(text)

splitter_800 = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=150
)
chunks_800 = splitter_800.split_text(text)

print("\nChunk Size Experiment Results")
print("300 Size:", len(chunks_300), "Chunks")
print("500 Size:", len(chunks_500), "Chunks")
print("800 Size:", len(chunks_800), "Chunks")

# Load Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("\nHome Activity Embeddings Model Loaded Successfully!")

# Remove old ChromaDB to prevent duplicate documents
if os.path.exists("home_chroma_db"):
    shutil.rmtree("home_chroma_db")

# Store documents in ChromaDB
home_vector_store = Chroma.from_texts(
    texts=chunks_500,
    embedding=embeddings,
    collection_name="home_activity_documents",
    persist_directory="home_chroma_db"
)

print("Home Activity Documents Stored Successfully!")
print("Total Documents:", home_vector_store._collection.count())

# Prepare 15 Questions
queries = [
    "What is Artificial Intelligence?",
    "What are the main objectives of Artificial Intelligence?",
    "What are the characteristics of AI?",
    "What are the limitations and ethics of AI?",
    "What are AI search strategies?",
    "What is a state space in AI?",
    "What are the different types of knowledge representation?",
    "What is uncertain knowledge?",
    "What is Bayesian reasoning?",
    "What is machine learning?",
    "What are the three types of machine learning?",
    "What is an expert system?",
    "What are rule-based expert systems?",
    "What is Natural Language Processing?",
    "What are intelligent agents and PEAS?"
]

queries_df = pd.DataFrame({
    "Query_ID": range(1, 16),
    "Question": queries
})

queries_df.to_csv("dataset/queries.csv", index=False)

print("15 Questions Saved Successfully!")
# Retrieve Top-5 Chunks for Each Question

retrieved_chunks_data = []

for index, row in queries_df.iterrows():
    question = row["Question"]

    results = home_vector_store.similarity_search(
        question,
        k=5
    )

    print("\nQuestion:", question)
    print("Retrieved Chunks:", len(results))

    for chunk_number, result in enumerate(results, start=1):
        retrieved_chunks_data.append({
            "Query_ID": row["Query_ID"],
            "Question": question,
            "Chunk_Number": chunk_number,
            "Retrieved_Chunk": result.page_content
        })

# Save Retrieved Chunks
retrieved_df = pd.DataFrame(retrieved_chunks_data)

retrieved_df.to_csv(
    "dataset/retrieved_chunks.csv",
    index=False
)

print("\nTop-5 Chunks Retrieved Successfully!")
print("Total Retrieved Chunks:", len(retrieved_df))
print("Retrieved Chunks Saved Successfully!")
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Load Ollama LLM
llm = OllamaLLM(
    model="gemma3:1b",
    temperature=0
)

# RAG Prompt
prompt = ChatPromptTemplate.from_template("""
Answer the question using only the given context.
If the answer is not available in the context,
say "The answer is not available in the PDF."

Context:
{context}

Question:
{question}

Answer:
""")

print("\nOllama Model Loaded Successfully!")
# Compare Answers With and Without RAG

answers_data = []

for index, row in queries_df.iterrows():

    question = row["Question"]

    # Without RAG
    without_rag_answer = llm.invoke(question)

    # Retrieve Top-5 Chunks
    results = home_vector_store.similarity_search(
        question,
        k=5
    )

    context = "\n\n".join(
        result.page_content for result in results
    )

    # With RAG
    rag_chain = prompt | llm
    rag_answer = rag_chain.invoke({
        "context": context,
        "question": question
    })

    answers_data.append({
        "Query_ID": row["Query_ID"],
        "Question": question,
        "Without_RAG_Answer": without_rag_answer,
        "With_RAG_Answer": rag_answer
    })

    print("\nQuestion:", question)
    print("Without RAG:", without_rag_answer)
    print("With RAG:", rag_answer)

print("\nAnswer Comparison Completed!")
# Save Answer Comparison Results

answers_df = pd.DataFrame(answers_data)

answers_df.to_csv(
    "dataset/home_answers.csv",
    index=False
)

print("\nHome Activity Answers Saved Successfully!")
print("Total Questions:", len(answers_df))