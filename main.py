import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool

load_dotenv()

# Prevent OpenAI requirement crash
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "NA"

app = FastAPI()
search_tool = SerperDevTool()

# Use the active Groq model
sigiriya_llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)


SIGIRIYA_SITES = [
    "Sinha Padaya", 
    "Rajamaligawa", 
    "Sinhasanaya", 
    "Mirror Wall", 
    "Water Gardens", 
    "Cobra Hood Cave",
    "Sigiriya"
]

class ChatRequest(BaseModel):
    location: str
    user_query: str

@app.post("/chat")
async def sigiriya_chat(request: ChatRequest):
    # 1. HARD VALIDATION: Check if the location is in Sigiriya
    # If the location is "Dalada Maligawa" or anything else, it stops here.
    if request.location not in SIGIRIYA_SITES:
        return {
            "location": request.location, 
            "response": f"Sorry, I can only provide information for Sigiriya locations. Information for {request.location} is not available."
        }

    # 2. AGENT LOGIC (Only runs if location is valid)
    guide = Agent(
        role=f"{request.location} Expert",
        goal=f"Provide details ONLY about {request.location} inside the Sigiriya complex.",
        backstory=(
            f"You are a local guide at {request.location}. You do not know about "
            "any places outside of Sigiriya. Stick strictly to the history of this spot."
        ),
        tools=[search_tool],
        llm=sigiriya_llm,
        verbose=True,
        allow_delegation=False
    )

    task = Task(
        description=f"Explain '{request.user_query}' specifically for {request.location}.",
        expected_output="A historical explanation of the site.",
        agent=guide
    )

    crew = Crew(agents=[guide], tasks=[task])
    result = crew.kickoff()
    
    return {"location": request.location, "response": str(result.raw)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)