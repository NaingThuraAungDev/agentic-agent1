import os
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.output_parsers import PydanticOutputParser
from tools import read_resume, find_candidate_info, save_candidate_to_db, fetch_github_stats

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# 1. SETUP BRAIN
load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# 2. BIND TOOLS
tools = [read_resume, save_candidate_to_db, fetch_github_stats]

# 3. DEFINE BEHAVIOR (System Prompt)
system_prompt = f"""You are an Expert Technical Recruiter. Your goal is to determine cv is accepted or rejected.

RULES:
1. READ the resume first.
2. Find the github url in the resume. 
    - (e.g., "github.com/username").

    [SCENARIO A: Link Found]
   - IF you see a link: STOP. Do NOT search the web.
   - IMMEDIATELY call 'fetch_github_stats' with that link.
   - ONCE you verified github link save in database with the candidate's name, github URL, CV filename, and accepted status, USE 'save_candidate_to_db'.

    - Finally, RESPOND with a summary of the candidate's profile including GitHub stats and application status.

    Wrap the output in this format and provide no other text\n{parser.get_format_instructions()}
"""

# Web search is only used if no GitHub link is found in the resume.
#    [SCENARIO B: Link Missing]
#    - IF NO link is found: CALL 'find_candidate_info' to search for "Name + GitHub".
#    - THEN call 'fetch_github_stats' with the result.

# 4. CREATE AGENT (The Graph)
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt,  
    response_format=parser.pydantic_object    
)

# 5. RUN IT

if __name__ == "__main__":
    print("🤖 Recruiter Agent Activated...")
    
    user_query = input("Which CV you want to process? ")
    messages = [{"role": "user", "content": user_query}]
    raw_response = agent.invoke({"messages": messages})

    # Print the final response from the agent
    try:
        structured_response = raw_response.get("structured_response")
        print("Topic:", structured_response.topic)
        print("Sources:", structured_response.sources)
        print("Tools used:", structured_response.tools_used)
        print("Summary:", structured_response.summary)
    except Exception as e:
        print("Error parsing response", e, "Raw Response - ", raw_response)