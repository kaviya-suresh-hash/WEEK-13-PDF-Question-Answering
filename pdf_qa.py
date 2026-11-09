import os
import pandas as pd
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Step 1: Extract PDF Text
pdf_path = "dataset/sample.pdf"

reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text + "\n"

print("PDF Text Extracted Successfully!")
print("Total Characters:", len(text))

# Save extracted text
with open("dataset/extracted_text.txt", "w", encoding="utf-8") as file:
    file.write(text)

# Step 2: Split Text into Chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_text(text)

print("Total Chunks:", len(chunks))

# Save chunks into CSV
chunks_df = pd.DataFrame({
    "Chunk_ID": range(1, len(chunks) + 1),
    "Text": chunks
})

chunks_df.to_csv("dataset/chunks.csv", index=False)

# Step 3: Generate Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embeddings Model Loaded Successfully!")

# Step 4: Store Chunks in ChromaDB
if os.path.exists("chroma_db"):
    import shutil
    shutil.rmtree("chroma_db")

vector_store = Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    collection_name="pdf_qa_documents",
    persist_directory="chroma_db"
)

print("Documents Stored in ChromaDB Successfully!")

# Step 5: Load Ollama LLM
llm = OllamaLLM(
    model="gemma3:1b",
    temperature=0
)

print("Ollama LLM Loaded Successfully!")

# Step 6: Create Question Answering Prompt
prompt = ChatPromptTemplate.from_template("""
Answer the question using only the context given below.

If the answer is not available in the context, say:
"The answer is not available in the PDF."

Context:
{context}

Question:
{question}

Answer:
""")

# Step 7: Accept User Query
query = input("\nEnter your question: ")

# Step 8: Retrieve Relevant Chunks
results = vector_store.similarity_search(query, k=5)

retrieved_context = "\n\n".join(
    document.page_content for document in results
)

print("\nRetrieved Context:")
print(retrieved_context)

# Step 9: Generate Answer
formatted_prompt = prompt.format(
    context=retrieved_context,
    question=query
)

answer = llm.invoke(formatted_prompt)

print("\nGenerated Answer:")
print(answer)

# Step 10: Save Answer into CSV
answers_df = pd.DataFrame([{
    "Question": query,
    "Retrieved_Context": retrieved_context,
    "Answer": answer
}])

answers_df.to_csv(
    "dataset/answers.csv",
    index=False
)

print("\nAnswer Saved Successfully!")