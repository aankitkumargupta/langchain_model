from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline


llm = HuggingFacePipeline(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(max_new_tokens=100, temperature=0.5),
)
model = ChatHuggingFace(llm=llm)
result = model.invoke("what is capital of India")
print(result.content)
