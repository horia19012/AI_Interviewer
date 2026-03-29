from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os, json
from services.llm_service import generate_next_question, generate_summary, analyze_sentiment
import uuid


load_dotenv()

app = Flask(__name__)
CORS(app)
DATA_FILE = "data/interviews.json"

ongoing_interviews = {}

def save_interview(topic, questions, answers, summary):
    if not os.path.exists("data"):
        os.makedirs("data")
    data = []
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
    data.append({
        "topic": topic,
        "questions": questions,
        "answers": answers,
        "summary": summary
    })
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)


@app.route("/start-interview", methods=["POST"])
def start_interview():
    interview_id = str(uuid.uuid4())
    ongoing_interviews[interview_id] = {
        "topic": None,
        "answers": [],
        "questions": [],
        "count": 0,
        "max_questions": 4
    }
    first_question = "Please provide a topic you would like to discuss in this interview."
    return jsonify({"interview_id": interview_id, "question": first_question})


@app.route("/next-question", methods=["POST"])
def next_question():
    try:
        interview_id = request.json.get("interview_id")
        user_input = request.json.get("answer")

        if not interview_id or interview_id not in ongoing_interviews:
            return jsonify({"error": "Invalid or missing interview_id"}), 400

        state = ongoing_interviews[interview_id]

        if state["topic"] is None:
            state["topic"] = user_input
            state["count"] = 0
            question = generate_next_question(state["topic"], [], [])
            state["questions"].append(question)
            return jsonify({"question": question, "finished": False})
        else:
            state["answers"].append(user_input)
            state["count"] += 1

        if state["count"] >= state["max_questions"]:
            summary = generate_summary(state["answers"])
            sentiment = analyze_sentiment(state["answers"])
            save_interview(
                state["topic"],
                state["questions"],
                state["answers"],
                {"summary": summary, "sentiment": sentiment}
            )
            del ongoing_interviews[interview_id]
            return jsonify({"finished": True, "summary": summary, "sentiment": sentiment})

        question = generate_next_question(state["topic"], state["answers"], state["questions"])
        state["questions"].append(question)
        return jsonify({"question": question, "finished": False})

    except Exception as e:
        print(f"Route error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))