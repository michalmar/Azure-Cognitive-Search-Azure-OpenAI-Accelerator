# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import asyncio
import os
import random
import re
import requests

from botbuilder.core import ActivityHandler, TurnContext
from botbuilder.schema import Activity, ActivityTypes, ChannelAccount
from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader

from botbuilder.schema import (
    ConversationAccount,
    Attachment,
)
from botbuilder.schema.teams import (
    FileDownloadInfo,
    FileConsentCard,
    FileConsentCardResponse,
    FileInfoCard,
)
from botbuilder.schema.teams.additional_properties import ContentType


from langchain.memory import (ConversationBufferMemory,
                              ConversationBufferWindowMemory,
                              CosmosDBChatMessageHistory)
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               MessagesPlaceholder,
                               SystemMessagePromptTemplate)

from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document

from prompts import (CUSTOM_CHATBOT_PREFIX, WELCOME_MESSAGE)
from prompts import COMBINE_QUESTION_PROMPT, COMBINE_PROMPT


from typing import Any, Dict, List, Optional, Awaitable, Callable, Tuple, Type, Union
from collections import OrderedDict
import uuid

#####################


# Env variables needed by langchain
os.environ["OPENAI_API_BASE"] = os.environ.get("AZURE_OPENAI_ENDPOINT")
os.environ["OPENAI_API_KEY"] = os.environ.get("AZURE_OPENAI_API_KEY")
os.environ["OPENAI_API_VERSION"] = os.environ.get("AZURE_OPENAI_API_VERSION")
os.environ["OPENAI_API_TYPE"] = "azure"

      
# Bot Class
class MyBot(ActivityHandler):
    memory = None
    prompt = ChatPromptTemplate(
                messages=[
                    SystemMessagePromptTemplate.from_template(
                        CUSTOM_CHATBOT_PREFIX
                    ),
                    # The `variable_name` here is what must align with memory
                    MessagesPlaceholder(variable_name="chat_history"),
                    HumanMessagePromptTemplate.from_template("{question}")
                ]
            )
    
    memory_dict = {}
    memory_cleared_messages = [
        "Historie byla smazána.",
        "Historie byla vynulována.",
        "Historie byla obnovena na výchozí hodnoty.",
        "Došlo k resetování historie.",
        "Historie byla resetována na počáteční stav."
    ]

    # allowed content types for file upload
    ALLOWED_CONTENT_TYPES = [
        "text/plain",  # ".txt"
        "text/markdown",  # ".md"
    ]

    # FAISS db 
    db = None


    def __init__(self):
        self.model_name = os.environ.get("AZURE_OPENAI_MODEL_NAME") 
        self.llm = AzureChatOpenAI(deployment_name=self.model_name, temperature=0.7, max_tokens=600)

    def get_search_results(self, query: str, indexes: list, 
                       k: int = 5,
                       reranker_threshold: int = 1,
                       sas_token: str = "",
                       vector_search: bool = False,
                       similarity_k: int = 3, 
                       query_vector: list = []) -> List[dict]:
    
        # headers = {'Content-Type': 'application/json','api-key': os.environ["AZURE_SEARCH_KEY"]}
        # params = {'api-version': os.environ['AZURE_SEARCH_API_VERSION']}

        agg_search_results = dict()
        
        for index in indexes:
            # search_payload = {
            #     "search": query,
            #     "queryType": "semantic",
            #     "semanticConfiguration": "my-semantic-config",
            #     "count": "true",
            #     "speller": "lexicon",
            #     "queryLanguage": "en-us",
            #     "captions": "extractive",
            #     "answers": "extractive",
            #     "top": k
            # }
            # if vector_search:
            #     search_payload["vectors"]= [{"value": query_vector, "fields": "chunkVector","k": k}]
            #     search_payload["select"]= "id, title, chunk, name, location"
            # else:
            #     search_payload["select"]= "id, title, chunks, language, name, location, vectorized"
            

            # resp = requests.post(os.environ['AZURE_SEARCH_ENDPOINT'] + "/indexes/" + index + "/docs/search",
            #                 data=json.dumps(search_payload), headers=headers, params=params)

            # search_results = resp.json()
            docs = self.db.similarity_search_with_score(query)
            agg_search_results[index] = docs
        
        content = dict()
        ordered_content = OrderedDict()
        
        for index,search_results in agg_search_results.items():
            for doc in search_results:
                result = doc[0] # Document object
                relevance_score = doc[1] # Relevance score
                if relevance_score > reranker_threshold: # Show results that are at least N% of the max possible score=4
                    tmp_id = self.generate_doc_id()
                    content[tmp_id]={
                                            "title": result.metadata["source"], # result['title'], 
                                            "name": result.metadata["source"], # result['name'], 
                                            "location": "none", # result['location'] + sas_token if result['location'] else "",
                                            "caption": "none", # result['@search.captions'][0]['text'],
                                            "index": index
                                        }
                    content[tmp_id]["chunk"]= result.page_content #result['chunk']
                    content[tmp_id]["score"]= relevance_score # Uses the reranker score

                    # if vector_search:
                    #     content[tmp_id]["chunk"]= result['chunk']
                    #     content[tmp_id]["score"]= result['@search.score'] # Uses the Hybrid RRF score
                
                    # else:
                    #     content[tmp_id]["chunks"]= result['chunks']
                    #     content[tmp_id]["language"]= result['language']
                    #     content[tmp_id]["score"]= relevance_score # Uses the reranker score
                    #     content[tmp_id]["vectorized"]= result['vectorized']
                    
        # After results have been filtered, sort and add the top k to the ordered_content
        if vector_search:
            topk = similarity_k
        else:
            topk = k*len(indexes)
            
        count = 0  # To keep track of the number of results added
        for id in sorted(content, key=lambda x: content[x]["score"], reverse=True):
            ordered_content[id] = content[id]
            count += 1
            if count >= topk:  # Stop after adding 5 results
                break

        return ordered_content

    def generate_index(self):
        return str(uuid.uuid4())    
    def generate_doc_id(self):
        return str(uuid.uuid4())
    
    def format_response(self, response):
        # return re.sub(r"(\n\s*)+\n+", "\n\n", response).strip()
        return response.strip()
    # Function to show welcome message to new users
    async def on_members_added_activity(self, members_added: ChannelAccount, turn_context: TurnContext):
        for member_added in members_added:
            if member_added.id != turn_context.activity.recipient.id:
                await turn_context.send_activity(WELCOME_MESSAGE + "\n\n" + self.model_name)
    
    
    # See https://aka.ms/about-bot-activity-message to learn more about the message and other activity types.
    async def on_message_activity(self, turn_context: TurnContext):
             
        # Extract info from TurnContext - You can change this to whatever , this is just one option 
        session_id = turn_context.activity.conversation.id
        user_id = turn_context.activity.from_property.id + "-" + turn_context.activity.channel_id

        QUESTION = turn_context.activity.text

        if session_id not in self.memory_dict:
            self.memory_dict[session_id] = ConversationBufferWindowMemory(memory_key="chat_history",input_key="question", return_messages=True, k=3)

        message_with_file_download = (
            False
            if not turn_context.activity.attachments
            # else turn_context.activity.attachments[0].content_type == "text/plain"
            else turn_context.activity.attachments[0].content_type in self.ALLOWED_CONTENT_TYPES
        )

        if message_with_file_download:
            await turn_context.send_activity(Activity(type=ActivityTypes.typing))
            # Save an uploaded file locally
            file = turn_context.activity.attachments[0]
            file_download = FileDownloadInfo.deserialize(file.content)
            file_path = "./" + file.name
            response = requests.get(file.content_url, allow_redirects=True)
            # response = requests.get(file_download.download_url, allow_redirects=True)
            open(file_path, "wb").write(response.content)

            # # read the file
            # with open(file_path, "r") as f:
            #     file_content_text = f.read()


            # load the file int FAISS db
            chunk_size = 1000
            loader = TextLoader(file_path)
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
            docs = text_splitter.split_documents(documents)

            embeddings = OpenAIEmbeddings()
            warning_msg = ""
            if (len(docs) > 16):
                warning_msg = f"Only the first 16 chunks will be loaded (got {len(docs) } chunks for chunk size = {chunk_size})"
                docs = docs[:16]
            
            self.db = FAISS.from_documents(docs, embeddings)

            await turn_context.send_activity("document loaded \n" + warning_msg + "\nYou can now ask questions about the document")
            return

            
        if (QUESTION.startswith("/file")):
            # perform vector search
            await turn_context.send_activity(Activity(type=ActivityTypes.typing))

            # remove the /file prefix
            QUESTION = QUESTION[5:].strip()
            
            # query = "What did the president say about Ketanji Brown Jackson"
            # docs = self.db.similarity_search_with_score(query)
            vector_indexes = [self.generate_index()]

            ordered_results = self.get_search_results(QUESTION, vector_indexes, 
                                                    k=10,
                                                    reranker_threshold=0.8, #1
                                                    vector_search=True, 
                                                    similarity_k=10,
                                                    #query_vector = embedder.embed_query(QUESTION)
                                                    query_vector= []
                                                    )
            # COMPLETION_TOKENS = 1000
            # llm = AzureChatOpenAI(deployment_name=MODEL, temperature=0.5, max_tokens=COMPLETION_TOKENS)

            top_docs = []
            for key,value in ordered_results.items():
                location = value["location"] if value["location"] is not None else ""
                # top_docs.append(Document(page_content=value["chunk"], metadata={"source": location+os.environ['BLOB_SAS_TOKEN']}))
                top_docs.append(Document(page_content=value["chunk"], metadata={"source": value["name"]}))
                    
            # print("Number of chunks:",len(top_docs))

            chain_type = "stuff"
            
            if chain_type == "stuff":
                chain = load_qa_with_sources_chain(self.llm, chain_type=chain_type, 
                                                prompt=COMBINE_PROMPT)
            elif chain_type == "map_reduce":
                chain = load_qa_with_sources_chain(self.llm, chain_type=chain_type, 
                                                question_prompt=COMBINE_QUESTION_PROMPT,
                                                combine_prompt=COMBINE_PROMPT,
                                                return_intermediate_steps=True)


            response = chain({"input_documents": top_docs, "question": QUESTION, "language": "English"})
            text_output = self.format_response(response['output_text'])
            await turn_context.send_activity(text_output)
            return

        if (QUESTION == "/reset"):
            # self.memory.clear()
            self.memory_dict[session_id].clear()
            # randomly pick one of the memory_cleared_messages
            await turn_context.send_activity(random.choice(self.memory_cleared_messages))
            # await turn_context.send_activity("Memory cleared")
        elif (QUESTION == "/help"):
            await turn_context.send_activity(WELCOME_MESSAGE + "\n\n" + self.model_name)
        else:
            
            chatgpt_chain = LLMChain(
                llm=self.llm,
                prompt=self.prompt,
                verbose=False,
                memory=self.memory_dict[session_id]
            )
            await turn_context.send_activity(Activity(type=ActivityTypes.typing))
            answer = chatgpt_chain.run(QUESTION)
            await turn_context.send_activity(answer)





