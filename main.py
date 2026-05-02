import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
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


# ── Location Knowledge Base ─────────────────────────────────────────
# Synced with Flutter app's 14 tourist attractions (11 original + 3 hidden gems)
SIGIRIYA_KNOWLEDGE = {
    "Sigiriya Entrance": {
        "description": "The main entrance gateway to the ancient Sigiriya rock fortress complex, where visitors begin their journey into King Kashyapa's 5th-century citadel.",
        "key_facts": [
            "Sigiriya was built by King Kashyapa I (477-495 CE) after he seized the throne from his father King Dhatusena.",
            "The entrance leads into a vast complex covering about 160 hectares.",
            "Visitors pass through the outer moat and ramparts that once protected the royal city.",
            "Declared a UNESCO World Heritage Site in 1982.",
            "The name 'Sigiriya' means 'Lion Rock' in Sinhalese, derived from 'Sinha' (lion) and 'Giri' (rock)."
        ],
        "suggested_questions": [
            "What is the history of Sigiriya?",
            "Who built this fortress and why?",
            "Why is it called Lion Rock?"
        ],
        "category": "Historical Site",
        "visit_order": 1
    },
    "Bridge over Moat": {
        "description": "The ancient bridge crossing the wide moat that surrounds the Sigiriya fortress, part of the elaborate defensive water system designed to protect King Kashyapa's citadel.",
        "key_facts": [
            "The moat surrounding Sigiriya is one of the oldest and most sophisticated defensive water systems in the ancient world.",
            "The moat was part of a triple-layered defense system: moat, rampart wall, and inner moat.",
            "The bridge provided the only controlled access point into the inner royal city.",
            "The moat system also served as a water management feature, channeling rainwater for irrigation.",
            "Crocodiles were reportedly kept in the moat as an additional defense measure."
        ],
        "suggested_questions": [
            "Why was the moat built around Sigiriya?",
            "How was this bridge used in ancient times?",
            "What defensive systems protected the fortress?"
        ],
        "category": "Historical Site",
        "visit_order": 2
    },
    "Water Garden": {
        "description": "The spectacular Water Gardens at the western foot of Sigiriya rock, one of the oldest landscaped gardens in the world, showcasing advanced 5th-century hydraulic engineering.",
        "key_facts": [
            "The Water Gardens are among the oldest landscaped gardens in the world, dating to the 5th century CE.",
            "Designed in three distinct sections: miniature water gardens, fountain gardens, and the large island garden.",
            "The gardens feature symmetrical pools, islands, and water channels with remarkable precision.",
            "Underground conduits connect the pools and channels in a sophisticated hydraulic network.",
            "The gardens were designed for both aesthetic beauty and practical cooling during Sri Lanka's hot seasons."
        ],
        "suggested_questions": [
            "How were these ancient gardens designed?",
            "What makes the Water Gardens unique?",
            "How did the hydraulic system work?"
        ],
        "category": "Nature Spot",
        "visit_order": 3
    },
    "Water Fountains": {
        "description": "The ancient stone fountains within the Sigiriya Water Gardens — among the oldest known fountains in the world — that still function during the rainy season using the original 5th-century hydraulic system.",
        "key_facts": [
            "These are believed to be the oldest surviving fountains in the world.",
            "The fountains still operate during the rainy season using the original 1,500-year-old hydraulic mechanism.",
            "They work on simple water pressure principles — underground limestone conduits carry water from higher elevations.",
            "Circular and square stone plates with perforations create the fountain spray patterns.",
            "The fountains demonstrate that ancient Sri Lankan engineers had a sophisticated understanding of hydraulic pressure."
        ],
        "suggested_questions": [
            "How do these ancient fountains still work?",
            "How old are these fountains?",
            "What technology was used to build them?"
        ],
        "category": "Nature Spot",
        "visit_order": 4
    },
    "Summer Palace": {
        "description": "The ruins of King Kashyapa's Summer Palace (Miniature Water Palace), a pleasure garden complex at the base of Sigiriya rock used for relaxation and royal entertainment.",
        "key_facts": [
            "The Summer Palace was a leisure complex used by King Kashyapa and the royal court.",
            "It featured bathing pools, pavilions, and shaded walkways surrounded by water features.",
            "The palace design used water channels to create natural air cooling — an ancient form of air conditioning.",
            "Stone seats and platforms were placed beside pools for the king to hold informal audiences.",
            "The layout reflects influence from both Indian and Sri Lankan royal garden traditions."
        ],
        "suggested_questions": [
            "What was the Summer Palace used for?",
            "How did the king relax here?",
            "What architectural features remain?"
        ],
        "category": "Historical Site",
        "visit_order": 5
    },
    "Caves with Inscriptions": {
        "description": "The ancient rock shelters and caves at Sigiriya containing drip-ledge inscriptions dating from the 3rd century BCE to the 1st century CE, predating King Kashyapa's fortress by centuries.",
        "key_facts": [
            "The caves contain Brahmi script inscriptions dating from the 3rd century BCE to the 1st century CE.",
            "These inscriptions prove that Sigiriya was used as a Buddhist monastery long before Kashyapa built his palace.",
            "Drip ledges were carved above cave mouths to divert rainwater away from the monks living inside.",
            "The inscriptions record donations made by ancient kings and nobles to the Buddhist monks.",
            "After Kashyapa's defeat in 495 CE, the site reverted to a Buddhist monastery until the 14th century."
        ],
        "suggested_questions": [
            "What do the ancient inscriptions say?",
            "How old are these caves?",
            "Who lived in these caves?"
        ],
        "category": "Historical Site",
        "visit_order": 6
    },
    "Lion's Paw": {
        "description": "The massive pair of lion paws carved into the rock at the northern face of Sigiriya — the remains of a colossal lion figure whose open mouth once served as the gateway to the upper palace.",
        "key_facts": [
            "Only the two enormous brick-and-plaster paws survive from what was once a full gigantic lion figure.",
            "The original lion's open mouth served as the dramatic entrance to the stairway leading to the summit palace.",
            "The name 'Sigiriya' (Lion Rock) derives directly from this lion gateway — 'Sinha Giri'.",
            "Built during King Kashyapa I's reign (477-495 CE) to project royal power and intimidation.",
            "The stairway between the paws was narrow by design, making it easily defensible against attackers."
        ],
        "suggested_questions": [
            "What did the original lion look like?",
            "Why was a lion chosen as the symbol?",
            "How was this structure built?"
        ],
        "category": "Main Attraction",
        "visit_order": 7
    },
    "Main Palace": {
        "description": "The Royal Palace complex (Rajamaligawa) at the summit of Sigiriya rock — King Kashyapa's sky palace covering approximately 1.6 hectares at the top of a 200-meter-high rock column.",
        "key_facts": [
            "The summit palace covers roughly 1.6 hectares (about 4 acres) at the top of the 200-meter rock.",
            "It contained a throne hall, royal chambers, cisterns carved from solid rock, and a swimming pool.",
            "A large stone throne (Sinhasanaya) at the summit offered panoramic views of the kingdom.",
            "The palace served as both a royal residence and an impregnable military fortress.",
            "After King Kashyapa's defeat by his brother Moggallana in 495 CE, the palace was abandoned and later converted into a Buddhist monastery."
        ],
        "suggested_questions": [
            "What rooms were in the Main Palace?",
            "How did the king live at the summit?",
            "What happened to the palace after Kashyapa?"
        ],
        "category": "Main Attraction",
        "visit_order": 8
    },
    "Boulder Gardens": {
        "description": "The Boulder Gardens on the southern slope of Sigiriya, where massive natural boulders were incorporated into the landscape and architectural design, with pathways, pavilions, and cave dwellings built among them.",
        "key_facts": [
            "The Boulder Gardens link the Water Gardens at the base to the terraced gardens higher up the rock.",
            "Ancient architects ingeniously incorporated massive natural boulders into the garden design rather than removing them.",
            "Many boulders contain caves that were used as monk residences, with carved drip ledges and sleeping platforms.",
            "The Cobra Hood Cave (Nagala Handiya), shaped like a cobra's hood, is one of the most famous boulders here.",
            "Brick walls, staircases, and pavilion foundations were built between and on top of the boulders."
        ],
        "suggested_questions": [
            "How were the boulders used in the design?",
            "What is the Cobra Hood Cave?",
            "Who lived among these boulders?"
        ],
        "category": "Nature Spot",
        "visit_order": 9
    },
    "Mirror Wall": {
        "description": "The highly polished Mirror Wall (Kadapat Pawura) — a section of plastered wall that was once so reflective the king could see his reflection while walking alongside it, now famous for ancient graffiti verses.",
        "key_facts": [
            "The wall was originally polished with a mixture of egg whites, beeswax, and lime to a mirror-like finish.",
            "Ancient visitors inscribed poetry and prose (Sigiri Graffiti) on the wall from the 7th to 11th century.",
            "Over 1,800 verses have been documented — among the oldest surviving examples of Sinhalese writing.",
            "The graffiti includes praise for the Sigiriya frescoes, love poems, and traveler observations.",
            "The wall runs along the pathway leading from the spiral staircase (to the frescoes) toward the Lion Platform."
        ],
        "suggested_questions": [
            "What is written on the Mirror Wall?",
            "How was the wall made so reflective?",
            "How old are the graffiti inscriptions?"
        ],
        "category": "Main Attraction",
        "visit_order": 10
    },
    "Sigiriya Museum": {
        "description": "The Sigiriya Archaeological Museum located near the entrance, housing artifacts, replicas, and exhibits that tell the complete story of Sigiriya from prehistoric times to the present.",
        "key_facts": [
            "The museum was established by the Central Cultural Fund of Sri Lanka.",
            "It contains original artifacts excavated from the Sigiriya complex including pottery, coins, and jewelry.",
            "Features full-scale replicas of the famous Sigiriya Frescoes (Apsara paintings) for close-up viewing.",
            "The museum displays a detailed scale model of the entire Sigiriya complex showing the original layout.",
            "Exhibits cover the site's history from the 3rd century BCE monastery period through Kashyapa's era to its rediscovery by British explorers in 1831."
        ],
        "suggested_questions": [
            "What artifacts are displayed here?",
            "What are the Sigiriya Frescoes?",
            "When was Sigiriya rediscovered?"
        ],
        "category": "Historical Site",
        "visit_order": 11
    },
    "Pahan Gala": {
        "description": "Pahan Gala (also known as Mapagala) — an important internal location within the Sigiriya complex, featuring an ancient defensive fortress with remarkable cyclopean stone masonry that formed a key part of King Kashyapa's 5th-century citadel's surrounding defensive network.",
        "key_facts": [
            "Pahan Gala is an important internal location within the Sigiriya complex, serving as a defensive fortress structure integral to the site's military architecture.",
            "The site features ancient cyclopean stone masonry — massive stone blocks fitted together without mortar, demonstrating advanced construction techniques.",
            "It formed part of the outer defensive network protecting the Sigiriya royal complex, including the upper palace summit, landscaped gardens, and the famous Lion Gate.",
            "The fortress predates or was contemporary with King Kashyapa's 5th-century construction, showcasing the broader strategic planning of the citadel.",
            "Pahan Gala provides crucial evidence that Sigiriya's military and architectural planning extended well beyond the main rock, incorporating surrounding structures into a unified defensive system."
        ],
        "suggested_questions": [
            "What role does Pahan Gala play within Sigiriya?",
            "What is cyclopean stone masonry?",
            "How did Pahan Gala function as a defensive structure?"
        ],
        "category": "Historical Site",
        "visit_order": 12
    },
    "Aligala Caves": {
        "description": "Aligala Cave — located on the eastern slope of Sigiriya Rock, this is the oldest known archaeological site within the Sigiriya complex, with excavations confirming human habitation dating back approximately 5,500 years to the Mesolithic period.",
        "key_facts": [
            "Aligala Cave is located on the eastern slope of Sigiriya Rock and holds the distinction of being the oldest known archaeological site within the Sigiriya complex.",
            "According to the official archaeological signboard erected by the Department of Archaeology and the Central Cultural Fund, excavations have uncovered evidence of human habitation dating back approximately 5,500 years to the Mesolithic period.",
            "Stone tools, animal remains, and floral evidence discovered here reveal that prehistoric humans used this cave as a shelter and living space.",
            "Human occupation at this site continued through the Protohistoric period (10th to 9th century BCE), as confirmed by archaeological findings.",
            "These discoveries make Aligala Cave one of the most important prehistoric sites in Sri Lanka, predating the fortress by thousands of years."
        ],
        "suggested_questions": [
            "How old is Aligala Cave?",
            "What was found during the excavations?",
            "Why is Aligala the most important prehistoric site in Sigiriya?"
        ],
        "category": "Historical Site",
        "visit_order": 13
    },
    "Rock Shelter": {
        "description": "The Rock Shelter (Murakuti) — an important internal location within the Sigiriya complex, these ancient protective shelters were built directly into the Sigiriya rock and served as guard stations and lookout points for security and observation of the fortress.",
        "key_facts": [
            "The Rock Shelters at Sigiriya are ancient Murakuti — protective structures built into the rock face of the fortress.",
            "They were strategically positioned as guard stations where soldiers and sentries stayed to provide round-the-clock security for the citadel.",
            "The shelters doubled as lookout points, giving guards elevated vantage positions to observe approaching threats from all directions.",
            "Their placement along the rock demonstrates King Kashyapa's sophisticated military planning, creating a layered defense system across the fortress.",
            "The Murakuti rock shelters are a key example of how Sigiriya's natural rock formations were ingeniously adapted for military and defensive purposes."
        ],
        "suggested_questions": [
            "What are the Murakuti rock shelters?",
            "How were these shelters used for defense?",
            "Where are the lookout points located on the rock?"
        ],
        "category": "Historical Site",
        "visit_order": 14
    }
}

SIGIRIYA_SITES = list(SIGIRIYA_KNOWLEDGE.keys())

# ── Location Alias Resolution ────────────────────────────────────────
# Maps alternate names / common variants to the canonical location key
LOCATION_ALIASES = {
    # Pahan Gala variants
    "pahangala": "Pahan Gala",
    "pahan gala": "Pahan Gala",
    "mapagala": "Pahan Gala",
    # Aligala Caves variants
    "aligala": "Aligala Caves",
    "aligala caves": "Aligala Caves",
    "aligala cave": "Aligala Caves",
    # Rock Shelter variants
    "rock shelter": "Rock Shelter",
    "rock shelters": "Rock Shelter",
    "rockshelter": "Rock Shelter",
}

def resolve_location(name: str) -> str:
    """Resolve a location name to its canonical key, checking aliases (case-insensitive)."""
    if name in SIGIRIYA_KNOWLEDGE:
        return name
    # Case-insensitive alias lookup
    return LOCATION_ALIASES.get(name.lower(), name)


# ── Request / Response Models ────────────────────────────────────────

class ChatRequest(BaseModel):
    location: str
    user_query: str

class LocationRequest(BaseModel):
    location: str


# ── Endpoints ────────────────────────────────────────────────────────

@app.get("/locations")
async def get_all_locations():
    """Returns all supported locations with their metadata."""
    locations = []
    for name, info in SIGIRIYA_KNOWLEDGE.items():
        locations.append({
            "name": name,
            "category": info["category"],
            "visit_order": info["visit_order"],
            "description": info["description"],
            "suggested_questions": info["suggested_questions"]
        })
    locations.sort(key=lambda x: x["visit_order"])
    return {"locations": locations}


@app.post("/location-info")
async def get_location_info(request: LocationRequest):
    """Called when user arrives at a new location. Returns welcome info and suggested questions."""
    location = resolve_location(request.location)
    if location not in SIGIRIYA_KNOWLEDGE:
        return {
            "location": request.location,
            "supported": False,
            "welcome_message": f"Sorry, I can only provide information for Sigiriya locations. Information for {request.location} is not available.",
            "suggested_questions": []
        }

    site = SIGIRIYA_KNOWLEDGE[location]
    return {
        "location": location,
        "supported": True,
        "category": site["category"],
        "visit_order": site["visit_order"],
        "welcome_message": (
            f"Welcome to {location}! 🏛️\n\n"
            f"I'm your AI tour guide powered by advanced AI. "
            f"Ask me anything about this fascinating place - its history, "
            f"significance, legends, or interesting facts!"
        ),
        "description": site["description"],
        "suggested_questions": site["suggested_questions"]
    }


@app.post("/chat")
async def sigiriya_chat(request: ChatRequest):
    """AI chat — answers questions strictly about the user's current location."""

    # 1. HARD VALIDATION: location must be a known Sigiriya site
    location = resolve_location(request.location)
    if location not in SIGIRIYA_KNOWLEDGE:
        return {
            "location": request.location,
            "response": f"Sorry, I can only provide information for Sigiriya locations. Information for {request.location} is not available."
        }

    # 2. Build location-specific context
    site = SIGIRIYA_KNOWLEDGE[location]
    facts = "\n".join(f"- {f}" for f in site["key_facts"])
    other_locations = [name for name in SIGIRIYA_SITES if name != location]
    forbidden = ", ".join(other_locations)

    # 3. Create a strictly location-bound agent
    guide = Agent(
        role=f"{location} Tour Guide",
        goal=(
            f"Give short, fun, engaging answers ONLY about {location}. "
            f"Make tourists feel excited and connected to what they are seeing right now."
        ),
        backstory=(
            f"You are a friendly, enthusiastic local guide standing right at {location} in Sigiriya.\n\n"
            f"About this spot: {site['description']}\n\n"
            f"Facts you know:\n{facts}\n\n"
            f"RESPONSE STYLE:\n"
            f"- Be warm, conversational, and fun — like a knowledgeable friend, not a textbook.\n"
            f"- Keep it SHORT: 2-3 sentences max. No long paragraphs.\n"
            f"- Use simple, clear language. Spark curiosity and wonder.\n"
            f"- Emoji are welcome to make responses feel lively on mobile.\n\n"
            f"STRICT RULES:\n"
            f"- ONLY talk about {location}. Never mention {forbidden}.\n"
            f"- If asked about other places, reply: 'I'm your guide for {location} — ask me anything about this amazing spot!'"
        ),
        tools=[search_tool],
        llm=sigiriya_llm,
        verbose=True,
        allow_delegation=False
    )

    task = Task(
        description=(
            f"A tourist at {location} asks: '{request.user_query}'\n\n"
            f"Answer in 2-3 short, friendly sentences about {location} only. "
            f"If unrelated to this site, redirect them back with enthusiasm."
        ),
        expected_output=(
            f"A short, friendly 2-3 sentence answer about {location}. "
            f"Engaging, easy to read on mobile, no long paragraphs."
        ),
        agent=guide
    )

    crew = Crew(agents=[guide], tasks=[task])
    result = crew.kickoff()

    return {"location": location, "response": str(result.raw)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)