from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model="claude-3-7-sonnet-20250219")

result = model.invoke("what is the capital of Delhi")
print(result)
