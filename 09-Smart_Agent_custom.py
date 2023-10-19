
# %%
import os
import random
from langchain.chat_models import AzureChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import ConversationalChatAgent, AgentExecutor, Tool
from langchain.memory import CosmosDBChatMessageHistory
from langchain.callbacks.manager import CallbackManager

#custom libraries that we will use later in the app
from common.utils import DocSearchTool, CSVTabularTool, SQLDbTool, ChatGPTTool, BingSearchTool, run_agent
from common.callbacks import StdOutCallbackHandler
from common.prompts import CUSTOM_CHATBOT_PREFIX, CUSTOM_CHATBOT_SUFFIX 
from common.prompts import MSSQL_PROMPT, MSSQL_AGENT_PREFIX, MSSQL_AGENT_FORMAT_INSTRUCTIONS

from dotenv import load_dotenv
load_dotenv("credentials.env")

from IPython.display import Markdown, HTML, display 

def printmd(string):
    # display(Markdown(string))
    print(string)

MODEL_DEPLOYMENT_NAME = "gpt-4-32k" # Reminder: gpt-35-turbo models will create parsing errors and won't follow instructions correctly 

# %%
os.environ["OPENAI_API_BASE"] = os.environ["AZURE_OPENAI_ENDPOINT"]
os.environ["OPENAI_API_KEY"] = os.environ["AZURE_OPENAI_API_KEY"]
os.environ["OPENAI_API_VERSION"] = os.environ["AZURE_OPENAI_API_VERSION"]
os.environ["OPENAI_API_TYPE"] = "azure"

# %% [markdown]
# ### Get the Tools - Doc Search, CSV Agent, SQL Agent and  Web Search
# 
# In the file `common/utils.py` we created Agent Tools Classes for each of the Functionalities that we developed in prior Notebooks. This means that we are not using `qa_with_sources` chain anymore as we did until notebook 5. Agents that Reason, Act and Reflect is the best way to create bots that comunicate with sources.

# %%
cb_handler = StdOutCallbackHandler()
cb_manager = CallbackManager(handlers=[cb_handler])

llm = AzureChatOpenAI(deployment_name=MODEL_DEPLOYMENT_NAME, temperature=0.5, max_tokens=500)

# Uncomment the below line if you want to see the responses being streamed/typed
# llm = AzureChatOpenAI(deployment_name=MODEL_DEPLOYMENT_NAME, temperature=0.5, max_tokens=500, streaming=True, callback_manager=cb_manager)

# %%
vector_only_indexes = ["cogsrch-index-custom-vector"]
custom_search = DocSearchTool(llm=llm, vector_only_indexes = vector_only_indexes,
                           k=10, similarity_k=10, reranker_th=1,
                           sas_token=os.environ['BLOB_SAS_TOKEN'],
                           callback_manager=cb_manager, return_direct=True,
                           # This is how you can edit the default values of name and description
                           name="@internal",
                           description="useful when the questions includes the term: @internal.\n",
                           verbose=True)

# # %%
# # Test the Documents Search Tool with a question we know it doesn't have the knowledge for
# printmd(custom_search.run("Kdo je ministrem obrany CR?"))
# %%
tools = [custom_search]
# tools =[]

# %% [markdown]
# **Note**: Notice that since both the CSV file and the SQL Database have the same exact data, we are only going to use the SQLDBTool since it is faster and more reliable

# %% [markdown]
# ### Initialize the brain agent

# %%
cosmos = CosmosDBChatMessageHistory(
    cosmos_endpoint=os.environ['AZURE_COSMOSDB_ENDPOINT'],
    cosmos_database=os.environ['AZURE_COSMOSDB_NAME'],
    cosmos_container=os.environ['AZURE_COSMOSDB_CONTAINER_NAME'],
    connection_string=os.environ['AZURE_COMOSDB_CONNECTION_STRING'],
    session_id="Agent-Test-Session" + str(random.randint(1, 1000)),
    user_id="Agent-Test-User" + str(random.randint(1, 1000))
    )
# prepare the cosmosdb instance
cosmos.prepare_cosmos()

# %%
llm_a = AzureChatOpenAI(deployment_name=MODEL_DEPLOYMENT_NAME, temperature=0.0, max_tokens=500)
agent = ConversationalChatAgent.from_llm_and_tools(llm=llm_a, tools=tools, system_message=CUSTOM_CHATBOT_PREFIX, human_message=CUSTOM_CHATBOT_SUFFIX, format_instructions = MSSQL_AGENT_FORMAT_INSTRUCTIONS)
# memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=10, chat_memory=cosmos)
memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=10)
agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, memory=memory)

# This question should not use any tool, the brain agent should answer it without the use of any tool
# printmd(run_agent("hi, how are you doing today?", agent_chain))
# # agent_chain.memory.clear()
# printmd(run_agent("what was the last question?", agent_chain))


# print("AGENT:" + "-"*80)
# for m in agent_chain.agent.llm_chain.prompt.messages:
#     print(str(type(m)))
#     if 'MessagesPlaceholder' in str(type(m)):
#         print(m)
#     else:
#         printmd(m.prompt.template)
# print("AGENT:" + "-"*80)

# printmd(run_agent("@internal, Kdo je ministr obrany CR?", agent_chain))
printmd(run_agent("@internal, Jaké jsou benefity v J&T?", agent_chain))

# print("Memory buffer:" + "-"*20)
# print(agent_chain.memory.buffer)
# # print("Clearing memory")
# # print("Memory buffer:" + "-"*20)
# # agent_chain.memory.clear()
# # print(agent_chain.memory.buffer)

# %%
