import { useState, useEffect, useRef } from 'react'
import './ChatInterface.css'

function SentimentAnalysis({ sentiment }) {
  const sentiments = [
    { type: 'positive', color: '#27ae60', label: 'Positive' },
    { type: 'neutral', color: '#95a5a6', label: 'Neutral' },
    { type: 'negative', color: '#e74c3c', label: 'Negative' }
  ]

  return (
    <div className="sentiment-container">
      <h3>Sentiment Analysis</h3>
      <div className="sentiment-bars">
        {sentiments.map(({ type, color, label }) => (
          <div key={type} className="sentiment-item">
            <div className="sentiment-header">
              <span className="sentiment-label">{label}</span>
              <span className="sentiment-percentage">{sentiment[type]?.percentage || 0}%</span>
            </div>
            <div className="bar-background">
              <div 
                className="bar-fill" 
                style={{
                  width: `${sentiment[type]?.percentage || 0}%`,
                  backgroundColor: color
                }}
              ></div>
            </div>
            {sentiment[type]?.keywords && sentiment[type].keywords.length > 0 && (
              <div className="keywords">
                <strong>Keywords:</strong> {sentiment[type].keywords.join(', ')}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

export default function ChatInterface() {
  const [messages, setMessages] = useState([])
  const [interviewId, setInterviewId] = useState(null)
  const [userInput, setUserInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isFinished, setIsFinished] = useState(false)
  const [sentiment, setSentiment] = useState(null)
  const messagesEndRef = useRef(null)

  // Initialize interview on component mount
  useEffect(() => {
    const startInterview = async () => {
      try {
        const apiUrl = 'http://localhost:5000'
        const response = await fetch(`${apiUrl}/start-interview`, {
          method: 'POST',
        })
        const data = await response.json()
        setInterviewId(data.interview_id)
        setMessages([{ 
          type: 'bot', 
          content: data.question 
        }])
      } catch (error) {
        console.error('Error starting interview:', error)
        setMessages([{ 
          type: 'bot', 
          content: 'Error starting interview. Please refresh the page.' 
        }])
      }
    }

    startInterview()
  }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSendMessage = async (e) => {
    e.preventDefault()
    
    if (!userInput.trim() || !interviewId) return

    const userMessage = userInput
    setMessages(prev => [...prev, { type: 'user', content: userMessage }])
    setUserInput('')
    setIsLoading(true)

    try {
      const apiUrl = 'http://localhost:5000'
      const response = await fetch(`${apiUrl}/next-question`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          interview_id: interviewId,
          answer: userMessage
        })
      })

      const data = await response.json()

      if (data.finished) {
        console.log('Sentiment data:', data.sentiment)
        setMessages(prev => [...prev, { 
          type: 'bot', 
          content: `Interview completed!\n\nSummary:\n${data.summary}` 
        }])
        setSentiment(data.sentiment)
        setIsFinished(true)
      } else {
        setMessages(prev => [...prev, { 
          type: 'bot', 
          content: data.question 
        }])
      }
    } catch (error) {
      console.error('Error sending message:', error)
      setMessages(prev => [...prev, { 
        type: 'bot', 
        content: 'Error sending request. Please try again.' 
      }])
    } finally {
      setIsLoading(false)
    }
  }

  const restartInterview = () => {
    window.location.reload()
  }

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h1>AI Interview Assistant</h1>
        <p className="interview-id">Interview ID: {interviewId?.substring(0, 8)}...</p>
      </div>

      <div className="messages-container">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message message-${msg.type}`}>
            <div className="message-content">
              {msg.type === 'bot' && <span className="bot-label">AI Interviewer: </span>}
              {msg.type === 'user' && <span className="user-label"> You: </span>}
              <p>{msg.content}</p>
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="message message-bot">
            <div className="message-content">
              <span className="bot-label">AI Interviewer: </span>
              <p className="typing-indicator">
                <span></span><span></span><span></span>
              </p>
            </div>
          </div>
        )}
        {sentiment && (
          <SentimentAnalysis sentiment={sentiment} />
        )}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSendMessage} className="input-form">
        <input
          type="text"
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          placeholder={isFinished ? "Interview completed" : "Type your response..."}
          disabled={isLoading || isFinished}
          className="message-input"
        />
        <button 
          type="submit" 
          disabled={isLoading || isFinished || !userInput.trim()}
          className="send-button"
        >
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </form>

      {isFinished && (
        <button onClick={restartInterview} className="restart-button">
          Start New Interview
        </button>
      )}
    </div>
  )
}
