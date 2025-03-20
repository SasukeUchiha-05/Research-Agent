from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
import os
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent ,AgentExecutor
from tools import search_tool,wiki_tool,save_tool,save_to_txt

load_dotenv()

class ResearchResponse(BaseModel):
    topic:str
    summary:str
    source:list[str]
    tools_used:list[str]

api_key = os.getenv("OPENROUTER_API_KEY")

# llm2 = ChatAnthropic()
# llm = ChatOpenAI(model="google/gemma-3-1b-it:free" , openai_api_key=api_key)
llm = ChatOpenAI(
    model="google/gemini-2.0-pro-exp-02-05:free",
    openai_api_key=api_key,  # OpenRouter API key
    openai_api_base="https://openrouter.ai/api/v1",  # OpenRouter endpoint
)
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use all necessary tools.
            Record **all** tools you used in the response.
            
            You must use at least two different tools for your response. If Wikipedia provides an answer, verify it with the search tool as well.
            
            If the user asks to save the research, you **must** use the 'save_text_to_file' tool **and execute it**.

            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool,wiki_tool,save_tool]

agent = create_tool_calling_agent(
    llm =llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools,verbose=True)

query = input("What can i Help You research?: ")

raw_response = agent_executor.invoke({"query":query })

print(raw_response)
print("-------------------------------------------------------------------")

output_text = raw_response.get("output","").strip()

if output_text.startswith("```json"):
    output_text = output_text[7:].strip()
if output_text.endswith("```"):
    output_text = output_text[:-3].strip()
    
print(output_text)
print("-------------------------------------------------------------------")
# try:
#     output_dict = json.loads(output_text)
#     print(output_dict)
# except json.JSONDecodeError as e:
#     print("Error in parsing response",e)

# formatted_response = {
#     "topic": output_dict.get("properties", {}).get("topic", {}).get("title", "Unknown"),
#     "summary": output_dict.get("summary", "No summary provided."),
#     "source": [output_dict.get("properties", {}).get("topic", {}).get("source", "No source provided.")],
#     "tools_used": tools  # Since it's missing, provide a default value
# }
# print("---------------------------------------------------------")
# print(output_dict)
try:
    structured_response = parser.parse(output_text)
except Exception as e:
    print("Error in parsing response",e)
    structured_response = None
    print("Raw Response :",raw_response)
# structured_response = parser.parse(raw_response.get("output")[0]["text"])
print(structured_response)

if structured_response and "save_text_to_file" in structured_response.tools_used:
    print("Calling `save_text_to_file` manually...")
    save_to_txt(str(structured_response))  # Save the structured response to file
    print(" Data saved successfully!")
else:
    print("`save_text_to_file` tool was not executed.")

# response = llm.invoke("What is langChain?")
# print(response)