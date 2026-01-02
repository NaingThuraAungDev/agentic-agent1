import sqlite3
import requests
import pdfplumber
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from pydantic import BaseModel, Field

# --- (PDF Reader) ---
@tool
def read_resume(file_path: str) -> str:
    """Reads a PDF resume and returns the raw text."""
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error reading file: {e}"

# --- (Web Search) ---
# We use the pre-built DuckDuckGo tool for privacy and ease
search_tool = DuckDuckGoSearchRun()

@tool
def find_candidate_info(query: str) -> str:
    """Searches the web for missing candidate info (email, phone, github)."""
    return search_tool.invoke(query)


@tool
def fetch_github_stats(username_or_url: str) -> str:
    """
    Fetches profile stats from GitHub API. 
    Use this immediately if you see a GitHub URL in the resume.
    """
    print(f"🐙 [GitHub API] Connecting to: {username_or_url}")
    
    # Extract username from various URL formats
    username = username_or_url.rstrip("/").split("/")[-1]
    
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        print("🐙 [GitHub API] Data fetched successfully.")
        return (
            f"✅ GITHUB VERIFIED:\n"
            f"User: {data.get('login')}\n"
            f"Bio: {data.get('bio', 'No bio')}\n"
            f"Public Repos: {data.get('public_repos')}\n"
            f"Followers: {data.get('followers')}"
        )
    else:
        return f"❌ GitHub User '{username}' not found (Status: {response.status_code})"
    
# --- (Database) ---
# First, setup a dummy DB
conn = sqlite3.connect("cv_table.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS cv_table (name TEXT, github TEXT, cv_filename TEXT, status TEXT)")
conn.commit()

class CandidateProfile(BaseModel):
    name: str = Field(description="Full Name")
    github: str = Field(description="GitHub Username or URL")
    cv_filename: str = Field(description="Filename of the uploaded CV")
    status: str = Field(description="Application status (e.g., 'Accepted', 'Rejected')")

@tool(args_schema=CandidateProfile)
def save_candidate_to_db(name: str, github: str, cv_filename: str, status: str) -> str:
    """Saves the fully verified candidate profile to the database."""
    cursor.execute("INSERT INTO cv_table VALUES (?, ?, ?, ?)", (name, github, cv_filename, status))
    conn.commit()
    print(f"💾 [DB] Saved {name} to database.")
    return f"SUCCESS: Saved {name} to database."