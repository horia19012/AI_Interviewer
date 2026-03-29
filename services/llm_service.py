import os

import nltk
from dotenv import load_dotenv
from groq import Groq
from keybert import KeyBERT
from nltk.sentiment import SentimentIntensityAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise ValueError("GROQ_API_KEY not set in environment")
client = Groq(api_key=API_KEY)


classifier = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    return_all_scores=True
)

keyword_extraction_model = KeyBERT('all-MiniLM-L6-v2')


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
            "2. Then ask one follow-up question\n"

            "The follow-up question should:"
            "-GO DEEPER into the 'why' or 'how'"
            "- OR connect their experience to real-world situations"
            "-OR gently challenge or expand their perspective"

            "Rules:\n"
            "- Do NOT repeat previous questions\n"
            "- Do NOT copy the user's words exactly\n"
            "- Use 'you' or 'your'\n"
            "- Keep it natural and conversational\n"
            "- Keep total response short (2 sentences max)\n"
            "- Output ONLY the reaction + question (no explanations)\n\n"
            "Example format:"
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
    full_text = " ".join(answers)
    truncated_text = " ".join(full_text.split()[:400])

    full_result = classifier(truncated_text)[0]
    scores = {item['label']: item['score'] for item in full_result}

    neg_pct = round(scores.get('LABEL_0', 0) * 100, 2)
    neu_pct = round(scores.get('LABEL_1', 0) * 100, 2)
    pos_pct = round(scores.get('LABEL_2', 0) * 100, 2)

    keywords = keyword_extraction_model.extract_keywords(
        full_text,
        stop_words='english',
        keyphrase_ngram_range=(1, 1),
        top_n=50
    )

    pos_words = []
    neg_words = []
    neu_words = []

    for word, relevance_score in keywords:
        word = word.lower().strip()

        result = classifier(word)[0]
        word_scores = {item['label']: item['score'] for item in result}

        best_label = max(word_scores, key=word_scores.get)
        best_score = word_scores[best_label]

        if best_label == 'LABEL_2' and best_score >= 0.6:
            pos_words.append(word)
        elif best_label == 'LABEL_0' and best_score >= 0.6:
            neg_words.append(word)
        else:
            neu_words.append(word)

    return {
        "positive": {"percentage": pos_pct, "keywords": pos_words[:8]},
        "neutral":  {"percentage": neu_pct, "keywords": neu_words[:8]},
        "negative": {"percentage": neg_pct, "keywords": neg_words[:8]},
    }