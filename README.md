# AI Interviewer Assistant

- Built an AI-powered interview system using a chat interface connected to an API-based language model.  
- Dynamically generates interview questions and provides feedback based on user responses.  
- Maintains conversation history to ensure context-aware interactions.  
- After 4 questions, generates a structured summary with key points extracted from answers.  
- Uses a **sentiment classifier** (`cardiffnlp/twitter-roberta-base-sentiment-latest`) to determine the sentiment of user responses.  
- Uses **VADER** to extract and classify **keywords** that contribute to the sentiment.  
- Combines LLM responses with sentiment insights to deliver personalized feedback.  

<img width="625" height="948" alt="demo1" src="https://github.com/user-attachments/assets/2b5948b8-c2dd-4a40-a54e-d2d376fb8231" />
<img width="628" height="947" alt="demo2" src="https://github.com/user-attachments/assets/6ea83edf-6cbf-4511-8318-04e63c18dcdd" />
<img width="620" height="947" alt="demo3" src="https://github.com/user-attachments/assets/fa941723-0dfb-4d03-9820-63173f1b5a54" />
