# from langchain_openai import OpenAIEmbeddings
# from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

from sklearn.metr  ics.pairwise import cosine_similarity

# import numpy as np

# load_dotenv()

# embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=300)

documents = [
    "The Eiffel Tower is one of the most iconic landmarks in Paris, attracting millions of tourists each year.",
    "Machine learning algorithms can analyze vast amounts of data to identify patterns and make predictions.",
    "A balanced diet rich in vitamins and minerals is essential for maintaining good health and energy levels.",
    "The latest advancements in electric vehicle technology are driving the shift toward sustainable transportation.",
    "Shakespeare's plays, such as Hamlet and Macbeth, continue to influence modern literature and theater.",
]

query = "Tell me about Hamlet"

doc_embeddings = embedding.embed_documents(documents)

query_embedding = embedding.embed_query(query)

# print(cosine_similarity([query_embedding], doc_embeddings))
score = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(list(enumerate(score)), key=lambda x: x[1])[-1]

print(query)
print(documents[index])
print("Similariy Score is:", score)
