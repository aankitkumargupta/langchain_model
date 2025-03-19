from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = "Delhi is the capital of India"
documents = [
    "Delhi is capital of India",
    "Kolkata is capital of West Bengal",
    "Paris is capital of Italy",
]
result = embedding.embed_documents(documents)
# vector = embedding.embed_query(text)

print(str(result))
