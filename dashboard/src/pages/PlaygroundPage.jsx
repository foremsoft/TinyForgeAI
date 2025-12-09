import React, { useState } from 'react'

const cardStyle = {
  background: 'white',
  borderRadius: '8px',
  padding: '24px',
  boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
  marginBottom: '24px'
}

const inputStyle = {
  width: '100%',
  padding: '10px 12px',
  border: '1px solid #ddd',
  borderRadius: '4px',
  fontSize: '14px',
  marginBottom: '12px'
}

const buttonStyle = {
  background: '#4a4a6a',
  color: 'white',
  border: 'none',
  padding: '12px 24px',
  borderRadius: '4px',
  cursor: 'pointer',
  fontSize: '14px',
  fontWeight: '500'
}

export default function PlaygroundPage() {
  const [apiUrl, setApiUrl] = useState('http://localhost:8000/predict')
  const [input, setInput] = useState('What is your refund policy?')
  const [response, setResponse] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)

  async function handleSubmit() {
    setIsLoading(true)
    setError(null)
    setResponse(null)

    try {
      const res = await fetch(apiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input })
      })

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`)
      }

      const data = await res.json()
      setResponse(data)
    } catch (err) {
      // For demo purposes, show a mock response if API is not available
      if (err.message.includes('Failed to fetch')) {
        setResponse({
          output: input.split('').reverse().join(''),
          confidence: 0.75,
          _note: 'Mock response (API not available)'
        })
      } else {
        setError(err.message)
      }
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div>
      <h1 style={{ marginBottom: '24px' }}>Playground</h1>

      <div style={cardStyle}>
        <h3 style={{ marginBottom: '16px' }}>API Configuration</h3>
        <label style={{ display: 'block', marginBottom: '6px', fontWeight: '500' }}>
          Service URL
        </label>
        <input
          type="text"
          value={apiUrl}
          onChange={(e) => setApiUrl(e.target.value)}
          style={inputStyle}
          placeholder="http://localhost:8000/predict"
        />
      </div>

      <div style={cardStyle}>
        <h3 style={{ marginBottom: '16px' }}>Input</h3>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          style={{
            ...inputStyle,
            minHeight: '100px',
            resize: 'vertical',
            fontFamily: 'inherit'
          }}
          placeholder="Enter your input text..."
        />
        <button
          onClick={handleSubmit}
          disabled={isLoading || !input.trim()}
          style={{
            ...buttonStyle,
            opacity: (isLoading || !input.trim()) ? 0.7 : 1
          }}
        >
          {isLoading ? 'Sending...' : 'Send Request'}
        </button>
      </div>

      {error && (
        <div style={{
          ...cardStyle,
          background: '#ffebee',
          borderLeft: '4px solid #c62828'
        }}>
          <h4 style={{ color: '#c62828', marginBottom: '8px' }}>Error</h4>
          <p>{error}</p>
        </div>
      )}

      {response && (
        <div style={{
          ...cardStyle,
          background: '#e8f5e9',
          borderLeft: '4px solid #4caf50'
        }}>
          <h4 style={{ marginBottom: '12px' }}>Response</h4>
          <pre style={{
            background: '#1a1a2e',
            color: '#e0e0e0',
            padding: '16px',
            borderRadius: '4px',
            overflow: 'auto',
            fontSize: '13px'
          }}>
            {JSON.stringify(response, null, 2)}
          </pre>
        </div>
      )}

      <div style={cardStyle}>
        <h3 style={{ marginBottom: '16px' }}>cURL Equivalent</h3>
        <pre style={{
          background: '#1a1a2e',
          color: '#e0e0e0',
          padding: '16px',
          borderRadius: '4px',
          overflow: 'auto',
          fontSize: '13px'
        }}>
{`curl -X POST ${apiUrl} \\
  -H "Content-Type: application/json" \\
  -d '{"input": "${input.replace(/'/g, "\\'")}"}'`}
        </pre>
      </div>
    </div>
  )
}
