import os
from dotenv import load_dotenv
from crewai_tools import SerpApiGoogleSearchTool

load_dotenv()

search_tool = SerpApiGoogleSearchTool(api_key=os.getenv("SERP_API_KEY"))
