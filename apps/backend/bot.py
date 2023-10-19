# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import asyncio
import os
import random
import re

from botbuilder.core import ActivityHandler, TurnContext
from botbuilder.schema import Activity, ActivityTypes, ChannelAccount
from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI


from langchain.memory import (ConversationBufferMemory,
                              ConversationBufferWindowMemory,
                              CosmosDBChatMessageHistory)
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               MessagesPlaceholder,
                               SystemMessagePromptTemplate)

from prompts import (CUSTOM_CHATBOT_PREFIX, WELCOME_MESSAGE)

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


    def __init__(self):
        self.model_name = os.environ.get("AZURE_OPENAI_MODEL_NAME") 
        self.llm = AzureChatOpenAI(deployment_name=self.model_name, temperature=0.7, max_tokens=600)

    
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





