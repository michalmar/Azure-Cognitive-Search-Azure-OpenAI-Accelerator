
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

llm = AzureChatOpenAI(deployment_name=MODEL_DEPLOYMENT_NAME, temperature=0.5, max_tokens=1000)

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



# Set up the prompt with input variables for tools, user input and a scratchpad for the model to record its workings
template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer without any modification
Thought: you should always think about what to do, you should stay in the language of input question
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action is the original input question
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to speak as a pirate when giving your final answer. Use lots of "Arg"s

Question: {input}
{agent_scratchpad}"""

from langchain.prompts import BaseChatPromptTemplate, ChatPromptTemplate
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage, SystemMessage
from langchain.tools import BaseTool
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
import re
from langchain import SerpAPIWrapper, LLMChain

# Set up a prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[BaseTool]
    
    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
            
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]
    
prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"]
)

class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        
        # Parse out the action and action input
        regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        
        # If it can't parse the output it raises an error
        # You can add your own logic here to handle errors in a different way i.e. pass to a human, give a canned response
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
    
output_parser = CustomOutputParser()


# Initiate our LLM - default is 'gpt-3.5-turbo'
# llm = ChatOpenAI(temperature=0)
llm = AzureChatOpenAI(deployment_name=MODEL_DEPLOYMENT_NAME, temperature=0.5, max_tokens=500)

# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Using tools, the LLM chain and output_parser to make an agent
tool_names = [tool.name for tool in tools]

agent = LLMSingleActionAgent(
    llm_chain=llm_chain, 
    output_parser=output_parser,
    # We use "Observation" as our stop sequence so it will stop when it receives Tool output
    # If you change your prompt template you'll need to adjust this as well
    stop=["\nObservation:"], 
    allowed_tools=tool_names
)
# Initiate the agent that will respond to our queries
# Set verbose=True to share the CoT reasoning the LLM goes through
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)


agent_executor.run("Kdo je ministr obrany?")



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
llm_a = AzureChatOpenAI(deployment_name=MODEL_DEPLOYMENT_NAME, temperature=0.5, max_tokens=500)
agent = ConversationalChatAgent.from_llm_and_tools(llm=llm_a, tools=tools, system_message=CUSTOM_CHATBOT_PREFIX, human_message=CUSTOM_CHATBOT_SUFFIX, format_instructions = MSSQL_AGENT_FORMAT_INSTRUCTIONS)
memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=10, chat_memory=cosmos)
# memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=10)
agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, memory=memory)

# This question should not use any tool, the brain agent should answer it without the use of any tool
# printmd(run_agent("hi, how are you doing today?", agent_chain))
# # agent_chain.memory.clear()
# printmd(run_agent("what was the last question?", agent_chain))
print("AGENT:" + "-"*80)
for m in agent_chain.agent.llm_chain.prompt.messages:
    print(str(type(m)))
    if 'MessagesPlaceholder' in str(type(m)):
        print(m)
    else:
        printmd(m.prompt.template)
print("AGENT:" + "-"*80)

printmd(run_agent("@internal, Kdo je ministr obrany CR?", agent_chain))

# print("Memory buffer:" + "-"*20)
# print(agent_chain.memory.buffer)
# # print("Clearing memory")
# # print("Memory buffer:" + "-"*20)
# # agent_chain.memory.clear()
# # print(agent_chain.memory.buffer)

# %%
