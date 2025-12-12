import json
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Load GROQ_API_KEY from .env
load_dotenv()

# Load JSON knowledge
with open("mosur_data.json", "r", encoding="utf-8") as f:
    mosur_data = json.load(f)

# Prompt template
template = """
You are the AI chatbot for MosurTech.

Answer ONLY using the company data below.
If the answer is not found, say:
"I don't have that information. Please contact MosurTech support."

COMPANY DATA:
{data}

USER QUESTION:
{input}

ANSWER:
"""

prompt = PromptTemplate(
    input_variables=["data", "input"],
    template=template
)

# Groq LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2
)

print("""
👋 MosurTech AI Assistant (Groq Powered)

Ask me about:
- Services
- Pricing
- Timelines
- Process
- Past work
- Contact details

Type 'exit' to quit.
""")

# Chat loop
while True:
    user = input("You: ")
    if user.lower().strip() == "exit":
        print("Bot: Goodbye! 👋")
        break

    final_prompt = prompt.format(
        data=json.dumps(mosur_data, indent=2),
        input=user
    )

    response = llm.invoke(final_prompt)
    print("\nBot:", response.content, "\n")
