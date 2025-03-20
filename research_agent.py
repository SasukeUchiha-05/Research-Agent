from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
import os
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool, save_to_txt

load_dotenv()

# Define response schema
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    source: list[str]
    tools_used: list[str]

# Load API key
api_key = os.getenv("OPENROUTER_API_KEY")

# Define LLM
llm = ChatOpenAI(
    model="google/gemini-2.0-pro-exp-02-05:free",
    openai_api_key=api_key,
    openai_api_base="https://openrouter.ai/api/v1",
)

# Define parser
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Define prompt (keeping your original format)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use all necessary tools.
            Record **all** tools you used in the response.
            
            You must use at least two different tools for your response. If Wikipedia provides an answer, verify it with the search tool as well.
            
            If the user asks to save the research, you **must** use the 'save_text_to_file' tool **and execute it**.

            Wrap the output in this format and provide no other text\n{format_instructions}
        """),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# Define tools
tools = [search_tool, wiki_tool, save_tool]

# Create agent
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

# Create agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Function to run the research agent
def run_research_agent(query: str) -> ResearchResponse:
    """
    Runs the AI research agent and returns structured research data.

    Args:
        query (str): The research topic/question.

    Returns:
        ResearchResponse: Structured research results.
    """
    try:
        raw_response = agent_executor.invoke({
            "query": query,
            "agent_scratchpad": ""  # ✅ If not needed, you can remove this
        })

        output_text = raw_response.get("output", "").strip()

        # Clean JSON formatting
        if output_text.startswith("```json"):
            output_text = output_text[7:].strip()
        if output_text.endswith("```"):
            output_text = output_text[:-3].strip()

        # Parse response
        structured_response = parser.parse(output_text)

        # Save if necessary
        if "save_text_to_file" in structured_response.tools_used:
            save_to_txt(str(structured_response))
            print("Data saved successfully!")

        return structured_response

    except Exception as e:
        print(f"Error in research agent: {e}")
        return ResearchResponse(
            topic=query,
            summary="An error occurred during research.",
            source=[],
            tools_used=[]
        )
