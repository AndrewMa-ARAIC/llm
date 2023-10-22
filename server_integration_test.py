import os

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# According to these links it seems that setting the openai_api_base as a param or an
# environment variable will allow us to use drop-in replacements for the OpenAI API
# https://clehaxze.tw/gemlog/2023/09-25-using-llama-cpp-python-server-with-langchain.gmi
# https://github.com/langchain-ai/langchain/issues/10415#issuecomment-1720879489

# link to the langchain 0.302 docs
# https://api.python.langchain.com/en/latest/chat_models/\
# langchain.chat_models.openai.ChatOpenAI.html#langchain.chat_models.openai.ChatOpenAI

# In order to serve the model locally, you need to run the following command:
# python3 -m llama_cpp.server --model <path to model> --n_ctx=<context length>
#
# I made an environment variable for the context length named N_CTX
#
# Further instructions can be found here:
# https://llama-cpp-python.readthedocs.io/en/latest/

# Mistral-7B docs suggests that it works for context length up to 8000 tokens

chat = ChatOpenAI(
    temperature=0,
    openai_api_key="YOUR_API_KEY",
    openai_api_base="http://localhost:8000/v1",
    max_tokens=2000,
)

system_template = (
    "You are MistralOrca, a large language model trained by Alignment Lab AI."
)
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

# get a chat completion from the formatted messages
result = chat(
    chat_prompt.format_prompt(text="What is the capital of Taiwan?").to_messages()
)

print(result.content)
