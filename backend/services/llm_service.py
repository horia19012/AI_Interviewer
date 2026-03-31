import os

import nltk
from dotenv import load_dotenv
from groq import Groq
from keybert import KeyBERT
from nltk.sentiment import SentimentIntensityAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import re

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise ValueError("GROQ_API_KEY not set in environment")
client = Groq(api_key=API_KEY)

classifier = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    top_k=None
)

keyword_extraction_model = KeyBERT('all-MiniLM-L6-v2')

nltk.download('vader_lexicon')
nltk.download('punkt')

vader = SentimentIntensityAnalyzer()


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


def score_keywords(keywords, dominant, vader):
    pos, neg, neu = [], [], []
    for kw, _ in keywords:
        score = vader.polarity_scores(kw)['compound']
        if dominant == 'positive' and score >= 0.05:
            pos.append(kw)
        elif dominant == 'negative' and score <= -0.05:
            neg.append(kw)
        elif dominant == 'neutral' and -0.05 < score < 0.05:
            neu.append(kw)
    return pos, neg, neu


def analyze_sentiment(answers):
    full_text = " ".join(answers)

    ideas = [idea.strip() for idea in re.split(r'[.,]', full_text) if idea.strip()]

    if not ideas:
        return {"positive": {"percentage": 0, "keywords": []},
                "negative": {"percentage": 0, "keywords": []},
                "neutral":  {"percentage": 0, "keywords": []}}

    total_pos, total_neg, total_neu = 0, 0, 0
    pos_words, neg_words, neu_words = [], [], []

    for idea in ideas:
        scores = {item['label'].lower(): item['score'] for item in classifier(idea, top_k=None)}

        total_pos += scores.get('positive', 0)
        total_neg += scores.get('negative', 0)
        total_neu += scores.get('neutral', 0)

        dominant = max(scores, key=scores.get)

        keywords = keyword_extraction_model.extract_keywords(
            idea,
            stop_words='english',
            keyphrase_ngram_range=(1, 1),
            top_n=10
        )

        p, n, ne = score_keywords(keywords, dominant, vader)
        pos_words.extend(p)
        neg_words.extend(n)
        neu_words.extend(ne)

    num_ideas = len(ideas)

    pos_pct = round((total_pos / num_ideas) * 100, 2)
    neg_pct = round((total_neg / num_ideas) * 100, 2)
    neu_pct = round((total_neu / num_ideas) * 100, 2)

    pos_words = list(dict.fromkeys(pos_words))
    neg_words = list(dict.fromkeys(neg_words))
    neu_words = list(dict.fromkeys(neu_words))

    return {
        "positive": {"percentage": pos_pct, "keywords": pos_words[:20]},
        "negative": {"percentage": neg_pct, "keywords": neg_words[:20]},
        "neutral":  {"percentage": neu_pct, "keywords": neu_words[:20]},
    }
