from groq import Groq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Groq Client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def generate_answer(query, docs):

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are an AI Research Assistant.

Your task is to answer the user's question using ONLY the provided context.

Rules:
1. Answer in 3-5 complete sentences.
2. Explain concepts clearly and professionally.
3. Do not make up information.
4. If the answer is not present in the context, respond:
   "The uploaded documents do not contain enough information to answer this question."
5. Keep answers concise but informative.

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=300
    )

    return response.choices[0].message.content