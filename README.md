# LangChain Model Demos

This repository contains various scripts demonstrating the use of different language models and embedding models using the LangChain framework. Each script showcases a specific model or functionality, such as chat models, embeddings, and document similarity.

## Prerequisites

Before running any scripts, ensure you have the necessary dependencies installed. You can install them using:

```sh
pip install langchain-openai langchain-anthropic langchain-google-genai langchain-huggingface scikit-learn python-dotenv
```

Additionally, ensure you have API keys for the respective models and that they are set up in your environment variables using a `.env` file.

## 1. OpenAI Models

### 1.1 Chat Model (OpenAI)
**File:** `1_chatmodel_openai.py`  
**Description:** Demonstrates how to use OpenAI's GPT-4 chat model for text-based conversations.

### 1.2 Embedding Query (OpenAI)
**File:** `1_embedding_openai_query.py`  
**Description:** Uses OpenAI's `text-embedding-3-large` model to generate a 32-dimensional embedding for a given text query.

### 1.3 LLM Demonstration (OpenAI)
**File:** `1_llm_demo.py`  
**Description:** Invokes OpenAI's GPT-3.5-turbo-instruct model for text-based response generation.

## 2. Anthropic and OpenAI Embedding for Documents

### 2.1 Chat Model (Anthropic)
**File:** `2_chatmodel_anthropic.py`  
**Description:** Uses Anthropic's GPT-4 equivalent model for chat-based interactions.

### 2.2 Embedding for Documents (OpenAI)
**File:** `2_embedding_openai_docs.py`  
**Description:** Generates embeddings for multiple documents using OpenAI's `text-embedding-3-large` model and returns their vector representation.

## 3. Google and Hugging Face Embeddings

### 3.1 Chat Model (Google)
**File:** `3_chatmodel_google.py`  
**Description:** Uses Google's `gemini-1.5-pro` model for conversational AI interactions.

### 3.2 Local Hugging Face Embeddings
**File:** `3_embedding_hf_local.py`  
**Description:** Uses `sentence-transformers/all-MiniLM-L6-v2` to generate embeddings locally for a set of documents.

## 4. Hugging Face API and Document Similarity

### 4.1 Chat Model (Hugging Face API)
**File:** `4_chatmodel_hf_api.py`  
**Description:** Uses Hugging Face's `TinyLlama-1.1B-Chat-v1.0` model for generating text responses via API.

### 4.2 Document Similarity with Embeddings
**File:** `4_document_similarity.py`  
**Description:** Uses Hugging Face's `sentence-transformers/all-MiniLM-L6-v2` embeddings to compute similarity between a query and a set of documents using cosine similarity.

## 5. Local Hugging Face Chat Model

### 5.1 Chat Model (Local Hugging Face Pipeline)
**File:** `5_chatmodel_hf_local.py`  
**Description:** Uses Hugging Face's `TinyLlama-1.1B-Chat-v1.0` model locally via a pipeline for text generation.

## Usage

To run any script, use the following command:

```sh
python <script_name>.py
```

Ensure that API keys and necessary credentials are correctly set up in your `.env` file before running the scripts.

## Conclusion

This repository provides a hands-on demonstration of various language models and embedding techniques using LangChain. It covers OpenAI, Anthropic, Google, and Hugging Face models for chat, text generation, and document similarity tasks.
