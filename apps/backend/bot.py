# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import re
import asyncio
import random
from concurrent.futures import ThreadPoolExecutor
from langchain.chat_models import AzureChatOpenAI
from langchain.utilities import BingSearchAPIWrapper
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationBufferMemory
from langchain.memory import CosmosDBChatMessageHistory
from langchain.agents import ConversationalChatAgent, AgentExecutor, Tool
from typing import Any, Dict, List, Optional, Union
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.schema import AgentAction, AgentFinish, LLMResult

#custom libraries that we will use later in the app
from utils import DocSearchTool, CSVTabularTool, SQLDbTool, ChatGPTTool, BingSearchTool, run_agent, get_answer
from prompts import WELCOME_MESSAGE, CUSTOM_CHATBOT_PREFIX, CUSTOM_CHATBOT_SUFFIX


######################
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

from utils import parse_pdf, read_pdf_files, text_to_base64
from prompts import COMBINE_QUESTION_PROMPT, COMBINE_PROMPT, COMBINE_PROMPT_TEMPLATE
from utils import (
    get_search_results,
    model_tokens_limit,
    num_tokens_from_docs,
    num_tokens_from_string
)
#####################

from botbuilder.core import ActivityHandler, TurnContext
from botbuilder.schema import ChannelAccount, Activity, ActivityTypes

# Env variables needed by langchain
os.environ["OPENAI_API_BASE"] = os.environ.get("AZURE_OPENAI_ENDPOINT")
os.environ["OPENAI_API_KEY"] = os.environ.get("AZURE_OPENAI_API_KEY")
os.environ["OPENAI_API_VERSION"] = os.environ.get("AZURE_OPENAI_API_VERSION")
os.environ["OPENAI_API_TYPE"] = "azure"


# Callback hanlder used for the bot service to inform the client of the thought process before the final response
class BotServiceCallbackHandler(BaseCallbackHandler):
    """Callback handler to use in Bot Builder Application"""
    
    def __init__(self, turn_context: TurnContext) -> None:
        self.tc = turn_context

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        asyncio.run(self.tc.send_activity(f"LLM Error: {error}\n"))

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        asyncio.run(self.tc.send_activity(f"Tool: {serialized['name']}"))

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        if "Action Input" in action.log:
            action = action.log.split("Action Input:")[1]
            asyncio.run(self.tc.send_activity(f"\u2611 Searching: {action} ..."))
            asyncio.run(self.tc.send_activity(Activity(type=ActivityTypes.typing)))

            
# Bot Class
class MyBot(ActivityHandler):
    
    def __init__(self):
        # self.model_name = os.environ.get("AZURE_OPENAI_MODEL_NAME") 
        self.model_name = "gpt-35-turbo-16k"
    
    # Function to show welcome message to new users
    async def on_members_added_activity(self, members_added: ChannelAccount, turn_context: TurnContext):
        for member_added in members_added:
            if member_added.id != turn_context.activity.recipient.id:
                await turn_context.send_activity(WELCOME_MESSAGE)
    
    
    # See https://aka.ms/about-bot-activity-message to learn more about the message and other activity types.
    async def on_message_activity(self, turn_context: TurnContext):
             
        # Extract info from TurnContext - You can change this to whatever , this is just one option 
        session_id = turn_context.activity.conversation.id
        user_id = turn_context.activity.from_property.id + "-" + turn_context.activity.channel_id
        input_text_metadata = dict()
        input_text_metadata["local_timestamp"] = turn_context.activity.local_timestamp.strftime("%I:%M:%S %p, %A, %B %d of %Y")
        input_text_metadata["local_timezone"] = turn_context.activity.local_timezone
        input_text_metadata["locale"] = turn_context.activity.locale

        # Setting the query to send to OpenAI
        input_text = turn_context.activity.text + "\n\n metadata:\n" + str(input_text_metadata)    
            
        # Set Callback Handler
        cb_handler = BotServiceCallbackHandler(turn_context)
        cb_manager = CallbackManager(handlers=[cb_handler])

        # Set brain Agent with persisten memory in CosmosDB
        # cosmos = CosmosDBChatMessageHistory(
        #                 cosmos_endpoint=os.environ['AZURE_COSMOSDB_ENDPOINT'],
        #                 cosmos_database=os.environ['AZURE_COSMOSDB_NAME'],
        #                 cosmos_container=os.environ['AZURE_COSMOSDB_CONTAINER_NAME'],
        #                 connection_string=os.environ['AZURE_COMOSDB_CONNECTION_STRING'],
        #                 session_id=session_id,
        #                 user_id=user_id
        #             )
        # cosmos.prepare_cosmos()
        # memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=30, chat_memory=cosmos)
        memory = ConversationBufferMemory(memory_key="chat_history",input_key="question")

        if (turn_context.activity.text == "/reset"):
            if (memory is not None):
                memory.clear()
            await turn_context.send_activity("Memory cleared")
        else:
            await turn_context.send_activity(Activity(type=ActivityTypes.typing))
           

            

            vector_only_indexes = ["cogsrch-index-custom-vector"]
            QUESTION = turn_context.activity.text
            # QUESTION = "Jaké mám benefity v J&T?"
            print("QUESTION:", QUESTION)
            vector_indexes = vector_only_indexes
            embedder = OpenAIEmbeddings(deployment="text-embedding-ada-002", chunk_size=1) 
            ordered_results = get_search_results(QUESTION, vector_indexes, 
                                                    k=7,
                                                    reranker_threshold=1,
                                                    vector_search=True, 
                                                    similarity_k=7,
                                                    query_vector = embedder.embed_query(QUESTION)
                                                    )
            

            print(f"Found {len(ordered_results)} files.")
            COMPLETION_TOKENS = 1000
            MODEL = self.model_name
            llm = AzureChatOpenAI(deployment_name=MODEL, temperature=0.5, max_tokens=COMPLETION_TOKENS)
            top_docs = []
            for key,value in ordered_results.items():
                location = value["location"] if value["location"] is not None else ""
                top_docs.append(Document(page_content=value["chunk"], metadata={"source": location+os.environ['BLOB_SAS_TOKEN']}))
                    
            print("Number of chunks:",len(top_docs))

            # Calculate number of tokens of our docs
            chain_type = "stuff"
            if(len(top_docs)>0):
                tokens_limit = model_tokens_limit(MODEL) # this is a custom function we created in common/utils.py
                prompt_tokens = num_tokens_from_string(COMBINE_PROMPT_TEMPLATE) # this is a custom function we created in common/utils.py
                context_tokens = num_tokens_from_docs(top_docs) # this is a custom function we created in common/utils.py
                
                requested_tokens = prompt_tokens + context_tokens + COMPLETION_TOKENS
                
                chain_type = "map_reduce" if requested_tokens > 0.9 * tokens_limit else "stuff"  
                
                print("System prompt token count:",prompt_tokens)
                print("Max Completion Token count:", COMPLETION_TOKENS)
                print("Combined docs (context) token count:",context_tokens)
                print("--------")
                print("Requested token count:",requested_tokens)
                print("Token limit for", MODEL, ":", tokens_limit)
                print("Chain Type selected:", chain_type)
                    
            else:
                print("NO RESULTS FROM AZURE SEARCH")

            if chain_type == "stuff":
                chain = load_qa_with_sources_chain(llm, chain_type=chain_type, 
                                                prompt=COMBINE_PROMPT)
            elif chain_type == "map_reduce":
                chain = load_qa_with_sources_chain(llm, chain_type=chain_type, 
                                                question_prompt=COMBINE_QUESTION_PROMPT,
                                                combine_prompt=COMBINE_PROMPT,
                                                return_intermediate_steps=True)

            # response = chain({"input_documents": top_docs, "question": QUESTION, "language": "Czech"})
            # response = get_answer(llm=llm, docs=top_docs,  query=QUESTION, language="Czech", chain_type=chain_type)
            # print(response['output_text'])

            response = get_answer(llm=llm, docs=top_docs, query=QUESTION, language="Czech", chain_type=chain_type, memory=memory)

            
            # printmd(response['output_text'])
            # print("Answer done.")
            # print(memory.buffer_as_str)


            answer = response['output_text']
            await turn_context.send_activity(answer)


            
            # # Set LLM 
            # llm = AzureChatOpenAI(deployment_name=self.model_name, temperature=0.5, max_tokens=1000, callback_manager=cb_manager)

            # # Initialize our Tools/Experts
            # # text_indexes = ["cogsrch-index-files", "cogsrch-index-csv"]
            # # doc_search = DocSearchTool(llm=llm, indexes=text_indexes,
            # #                    k=10, similarity_k=4, reranker_th=1,
            # #                    sas_token=os.environ['BLOB_SAS_TOKEN'],
            # #                    callback_manager=cb_manager, return_direct=True)
            # # vector_only_indexes = ["cogsrch-index-books-vector"]
            # # book_search = DocSearchTool(llm=llm, vector_only_indexes = vector_only_indexes,
            # #                    k=10, similarity_k=10, reranker_th=1,
            # #                    sas_token=os.environ['BLOB_SAS_TOKEN'],
            # #                    callback_manager=cb_manager, return_direct=True,
            # #                    name="@booksearch",
            # #                    description="useful when the questions includes the term: @booksearch.\n")
            # vector_only_indexes = ["cogsrch-index-custom-vector"]
            # custom_search = DocSearchTool(llm=llm, vector_only_indexes = vector_only_indexes,
            #                    k=10, similarity_k=10, reranker_th=1,
            #                    sas_token=os.environ['BLOB_SAS_TOKEN'],
            #                    callback_manager=cb_manager, return_direct=True,
            #                    # This is how you can edit the default values of name and description
            #                    name="@internal",
            #                    description="useful when the questions includes the term: @internal.\n",
            #                    verbose=True)
            # www_search = BingSearchTool(llm=llm, k=5, callback_manager=cb_manager, return_direct=True)
            # # sql_search = SQLDbTool(llm=llm, k=10, callback_manager=cb_manager, return_direct=True)
            # chatgpt_search = ChatGPTTool(llm=llm, callback_manager=cb_manager, return_direct=True)

            # # tools = [www_search, sql_search, doc_search, chatgpt_search, book_search]
            # tools = [www_search,  chatgpt_search, custom_search]

            # # Set brain Agent with persisten memory in CosmosDB
            # cosmos = CosmosDBChatMessageHistory(
            #                 cosmos_endpoint=os.environ['AZURE_COSMOSDB_ENDPOINT'],
            #                 cosmos_database=os.environ['AZURE_COSMOSDB_NAME'],
            #                 cosmos_container=os.environ['AZURE_COSMOSDB_CONTAINER_NAME'],
            #                 connection_string=os.environ['AZURE_COMOSDB_CONNECTION_STRING'],
            #                 session_id=session_id,
            #                 user_id=user_id
            #             )
            # cosmos.prepare_cosmos()
            # memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=30, chat_memory=cosmos)
            # agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=tools,system_message=CUSTOM_CHATBOT_PREFIX,human_message=CUSTOM_CHATBOT_SUFFIX)
            # agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, memory=memory)

            # if (turn_context.activity.text == "/reset"):
            #     if (agent_chain is not None):
            #         agent_chain.memory.clear()
            #     await turn_context.send_activity("Memory cleared")
            # else:
            #     await turn_context.send_activity(Activity(type=ActivityTypes.typing))
                
            #     # Please note below that running a non-async function like run_agent in a separate thread won't make it truly asynchronous. It allows the function to be called without blocking the event loop, but it may still have synchronous behavior internally.
            #     loop = asyncio.get_event_loop()
            #     answer = await loop.run_in_executor(ThreadPoolExecutor(), run_agent, input_text, agent_chain)
                
            #     await turn_context.send_activity(answer)



