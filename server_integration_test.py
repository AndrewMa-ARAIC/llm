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

import mlflow
import pandas as pd

import time

import os
import textwrap

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

def read_txt():
    with open(os.path.join(os.getcwd(), "input.txt"), "r") as f:
        text = f.read()
    return text

MODEL = os.getenv("MODEL", "UNKNOWN MODEL")
MAX_TOKENS = 2000

text = read_txt()
print(f"input.txt:\n{textwrap.fill(text, max_lines=5)}")
print()
print("Parameters:")
print(f'MODEL = {MODEL}')
print(f'N_CTX = {os.environ["N_CTX"]}')
print(f'max tokens = {MAX_TOKENS}')

chat = ChatOpenAI(
    temperature=0,
    openai_api_key="YOUR_API_KEY",
    openai_api_base="http://localhost:8000/v1",
    max_tokens=MAX_TOKENS,
    streaming=True,
    callback_manager=callback_manager,
)

system_template = (
    "You are MistralOrca, a large language model."
)
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

mlflow.set_experiment("local_llm")
with mlflow.start_run():
    start_time = time.time()
    
    print()
    print("Running inference...")
    # get a chat completion from the formatted messages
    result = chat(
        chat_prompt.format_prompt(
            text=text,
        ).to_messages()
    )
    print()
    print()
    print("Inference complete.")
    
    end_time = time.time()
    duration = end_time - start_time
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    formatted_duration = f"{minutes}m {seconds}s"

    # Create a pandas DataFrame with the details to be logged
    log_data = {
        "Model Used": [MODEL],
        "Duration": [formatted_duration],
        "Input": [text],
        "Output": [result.content],
    }
    df = pd.DataFrame(log_data)
    
    # Log the DataFrame as a table in mlflow
    mlflow.log_table(data=df, artifact_file="run_details.json")
