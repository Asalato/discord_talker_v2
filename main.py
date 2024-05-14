import os
import re

from langchain.agents import load_tools, AgentExecutor
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain_community.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.agents import AgentAction
from langchain_core.messages import AIMessage, HumanMessage

from langchain.callbacks.base import AsyncCallbackHandler
from langchain.tools.render import format_tool_to_openai_function
from langchain.tools import Tool, WikipediaQueryRun
from langchain.utilities import GoogleSearchAPIWrapper, StackExchangeAPIWrapper, WikipediaAPIWrapper

import discord

from dotenv import load_dotenv

load_dotenv(override=True)

llm = OpenAI(model_name="gpt-4o", temperature=0)

search = GoogleSearchAPIWrapper()
search_tool = Tool(
    name="GoogleSearch",
    description="Search web sites with Google for recent results.",
    func=search.run,
)
stackexchange = StackExchangeAPIWrapper()
stackexchange_tool = Tool(
    name="StackExchange",
    description="Search IT knowledge from Stack Exchange.",
    func=stackexchange.run,
)
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
wikipedia_tool = Tool(
    name="Wikipedia",
    description="Search Keywords from Wikipedia.",
    func=wikipedia.run,
)
tools = [search_tool, stackexchange_tool, wikipedia_tool,
         *load_tools(['arxiv', 'openweathermap-api', 'requests_all'], llm=llm)]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a very powerful assistant. You respond to questions as sincerely and in as much detail "
            "as possible. You **must** answer in the same language as the question was asked. "
            "If you are unclear in providing a response, please do not force a response, but ask questions to clarify "
            "conditions. If you give an accurate and honest answer, you will be tipped accordingly. (It is forbidden "
            "to request a tip with your response, as it will be paid after the request). "
            "Your answer must be formatted in the markdown format. "
            "If you obtained the information from a web page, you must fill the template:\n"
            "```\n- Title: {{title}}\n- URL: {{url}}\n- Access Date: {{access_date}}\n- Summary of the Page:\n"
            "{{summary}}\n- Noteworthy Descriptions:\n{{key_points}}\n- Articles/Resources to Read Next:\n"
            "{{related_content}}\n- Keywords:\n{{keywords}}\n- Notes:\n{{notes}}",
        ),
        MessagesPlaceholder(variable_name='chat_history'),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

llm_with_tools = ChatOpenAI(model_name="gpt-4-turbo").bind(
    functions=[format_tool_to_openai_function(t) for t in tools])
agent = (
        {
            "agent_scratchpad": lambda x: format_to_openai_function_messages(
                x["intermediate_steps"]
            ),
            "chat_history": lambda x: x['chat_history'],
        }
        | prompt
        | llm_with_tools
        | OpenAIFunctionsAgentOutputParser()
)

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True

client = discord.Client(intents=intents)


async def get_reply_chain(message: discord.Message):
    reply_chain = [message]
    while message.reference and isinstance(message.reference, discord.MessageReference):
        referenced_message_id = message.reference.message_id
        message = await message.channel.fetch_message(referenced_message_id)
        reply_chain.append(message)
    return reply_chain


def is_mentioned(message: discord.Message) -> bool:
    if message.mention_everyone:
        return False
    if client.user.mentioned_in(message):
        return True
    for role in message.role_mentions:
        if role in message.guild.me.roles:
            return True
    return False


def check_and_extract_text(text):
    pattern = r'[\r\n]+```(.+?)[\r\n]+'

    matches = re.findall(pattern, text)
    if len(matches) % 2 == 0:
        return False, None
    if matches:
        return True, matches[-1]
    else:
        return False, None


def split_texts(content):
    if len(content) > 2000:
        chunks = [content[i:i + 1970] for i in range(0, len(content), 1970)]

        result = []
        last_code_brock = (False, None)
        for chunk in chunks:
            if last_code_brock[0]:
                chunk = f'```{last_code_brock[1]}\n' + chunk
            last_code_brock = check_and_extract_text(chunk)
            if last_code_brock[0]:
                chunk += '\n```'
            result.append(chunk)
        return result
    else:
        return [content]


class CustomCallbackHandler(AsyncCallbackHandler):
    def __init__(self, thread: discord.Thread = None):
        self.thread = thread

    async def reply(self, content: str):
        if self.thread is None:
            return

        for chunk in split_texts(content):
            await self.thread.send(content=chunk)

    async def on_chain_start(self, serialized: dict, inputs: dict, **kwargs):
        await self.reply(content=f'> Run when chain starts running.')

    async def on_chain_end(self, outputs: dict, **kwargs):
        await self.reply(content=f'> Run when chain ends running.\n```\n{outputs}\n```')

    async def on_chain_error(self, error: Exception | KeyboardInterrupt, **kwargs):
        await self.reply(content=f'> Run when chain errors.\n```\n{error}\n```')

    async def on_agent_action(self, action: AgentAction, **kwargs):
        await self.reply(content=f'> Invoke tool:\n```\n{action}\n```')

    async def on_agent_finish(self, output: str, **kwargs):
        await self.reply(content=f'> Tool request succeeded.\n```\n{output}\n```')


@client.event
async def on_message(message: discord.Message):
    if message.author == client.user:
        return

    reply_chain = await get_reply_chain(message)
    reply_chain.reverse()
    mention_in_chain = any(is_mentioned(msg) for msg in reply_chain)

    if mention_in_chain:
        try:
            thread = await message.create_thread(name=f'リクエストログ ({message.id})', auto_archive_duration=60)
            await thread.send(content='> Start creating response...')
        except discord.errors.Forbidden:
            thread = None

        chat_history = [
            AIMessage(content=msg.content) if msg.author == client.user else HumanMessage(content=msg.content)
            for msg in reply_chain
        ]

        try:
            agent_executor = AgentExecutor(agent=agent, tools=tools, callbacks=[CustomCallbackHandler(thread)])
            result = (await agent_executor.ainvoke({'chat_history': chat_history}))['output']
        except Exception as e:
            print(e)
            result = '調べましたが、よくわかりませんでした。'

        reply_source_message = message
        for chunk in split_texts(result):
            reply_source_message = await reply_source_message.reply(content=chunk)


client.run(os.getenv('DISCORD_BOT_TOKEN'))
