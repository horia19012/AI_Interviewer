import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise ValueError("GROQ_API_KEY not set in environment")
client = Groq(api_key=API_KEY)


def call_groq(prompt):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    return response.choices[0].message.content.strip()


def generate_next_question(topic, previous_answers, previous_questions=None):
    if previous_questions is None:
        previous_questions = []
    if not previous_answers:
        prompt = (
            f"You are a friendly interviewer talking to a user about '{topic}'. "
            "Ask a personal, engaging first question that encourages them to share their experience. "
            "Use 'you' or 'your' in the question. Do not include introductions or commentary."
        )
    else:
        last_answer = previous_answers[-1]
        conversation_history = "\n".join(
            [f"Q: {q}\nA: {a}" for q, a in zip(previous_questions, previous_answers)]
        )
        prompt = (
            f"You are a professional interviewer conducting a thoughtful conversation about '{topic}'.\n"
            f"Conversation so far:\n{conversation_history}\n"
            f"The user just said:\n\"{last_answer}\"\n\n"

            "Your job:\n"
            " 1. Start with a brief, natural reaction to what the user said (1 sentence max)\n"
            "2. Then ask ONE follow-up question\n\n"

            "The follow-up question should:\n"
            "- Go deeper into the 'why' or 'how'\n"

            "- OR connect their experience to real-world situations\n"
            "- OR gently challenge or expand their perspective\n\n"

            "Rules:\n"
            "- Do NOT repeat previous questions\n"
            "- Do NOT copy the user's words exactly\n"
            "- Use 'you' or 'your'\n"
            "- Keep it natural and conversational\n"
            "- Keep total response short (2 sentences max)\n"
            "- Output ONLY the reaction + question (no explanations)\n\n"

            "Example format:\n"
            "That’s really interesting, especially how you approached that. How did that experience influence your perspective on this topic?"
        )
    return call_groq(prompt)


def generate_summary(answers):
    combined = "\n".join(answers)
    prompt = (
            "Summarize these interview answers, highlighting key points and sentiment:\n"
            + combined
    )
    return call_groq(prompt)


def analyze_sentiment(answers):
    combined = "\n".join(answers)

    prompt = (
            "You are an AI that analyzes interview responses.\n\n"
            "Analyze sentiment of the following answers and extract keywords.\n\n"
            "Return ONLY valid JSON. STRICTLY follow this format:\n"
            "{\n"
            '  "positive": { "percentage": 0, "keywords": [] },\n'
            '  "neutral": { "percentage": 0, "keywords": [] },\n'
            '  "negative": { "percentage": 0, "keywords": [] }\n'
            "}\n\n"
            "Rules:\n"
            "- Percentages must be integers\n"
            "- Percentages must sum to 100\n"
            "- Keywords must be short words (1-2 words) from the answers\n"
            "- No explanations, no extra text, output only JSON\n\n"
            "Answers:\n"
            + combined
    )

    result = call_groq(prompt)

    result = result.strip()
    if result.startswith("```"):
        result = result.split("```")[1].strip()
    return result
