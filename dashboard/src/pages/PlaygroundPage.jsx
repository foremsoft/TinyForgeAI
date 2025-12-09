import React, { useState, useEffect } from 'react'
import { predict, getServices, getModels } from '../api/client'

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
  const [services, setServices] = useState([])
  const [models, setModels] = useState([])
  const [selectedService, setSelectedService] = useState('')
  const [input, setInput] = useState('What is your refund policy?')
  const [response, setResponse] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const [history, setHistory] = useState([])

  useEffect(() => {
    const loadData = async () => {
      try {
        const [servicesData, modelsData] = await Promise.all([
          getServices(),
          getModels()
        ])
        const runningServices = servicesData.filter(s => s.status === 'running')
        setServices(runningServices)
        setModels(modelsData)
        if (runningServices.length > 0) {
          setSelectedService(runningServices[0].id)
        }
      } catch (err) {
        setError(err.message)
      }
    }
    loadData()
  }, [])

  async function handleSubmit() {
    if (!input.trim()) return

    setIsLoading(true)
    setError(null)
    setResponse(null)

    const startTime = Date.now()

    try {
      const data = await predict(input, selectedService || null)
      const latency = Date.now() - startTime

      const result = {
        ...data,
        latency_ms: latency
      }
      setResponse(result)

      // Add to history
      setHistory(prev => [{
        timestamp: new Date().toISOString(),
        input,
        output: data.output || data.prediction,
        latency_ms: latency,
        service: selectedService
      }, ...prev.slice(0, 9)]) // Keep last 10

    } catch (err) {
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  const getServiceInfo = (serviceId) => {
    const service = services.find(s => s.id === serviceId)
    if (!service) return null
    const model = models.find(m => m.id === service.model_id)
    return { service, model }
  }

  const selectedInfo = getServiceInfo(selectedService)

  return (
    <div>
      <h1 style={{ marginBottom: '24px' }}>Playground</h1>

      {error && (
        <div style={{
          ...cardStyle,
          background: '#fff3e0',
          borderLeft: '4px solid #ff9800',
          marginBottom: '24px'
        }}>
          <strong>Error:</strong> {error}
          <br />
          <small>Make sure the Dashboard API is running and a service is deployed</small>
        </div>
      )}

      <div style={cardStyle}>
        <h3 style={{ marginBottom: '16px' }}>Select Service</h3>

        {services.length === 0 ? (
          <div style={{
            padding: '16px',
            background: '#f5f5f5',
            borderRadius: '4px',
            color: '#666'
          }}>
            No running services available. Deploy a service first to use the playground.
          </div>
        ) : (
          <>
            <select
              value={selectedService}
              onChange={(e) => setSelectedService(e.target.value)}
              style={{ ...inputStyle, cursor: 'pointer' }}
            >
              {services.map(service => (
                <option key={service.id} value={service.id}>
                  {service.name} (Port {service.port})
                </option>
              ))}
            </select>

            {selectedInfo && (
              <div style={{
                padding: '12px',
                background: '#f5f5f5',
                borderRadius: '4px',
                fontSize: '13px'
              }}>
                <div><strong>Endpoint:</strong> <code>http://localhost:{selectedInfo.service.port}/predict</code></div>
                {selectedInfo.model && (
                  <div style={{ marginTop: '4px' }}>
                    <strong>Model:</strong> {selectedInfo.model.name} ({selectedInfo.model.model_type})
                  </div>
                )}
              </div>
            )}
          </>
        )}
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
          disabled={isLoading || !input.trim() || services.length === 0}
          style={{
            ...buttonStyle,
            opacity: (isLoading || !input.trim() || services.length === 0) ? 0.7 : 1
          }}
        >
          {isLoading ? 'Processing...' : 'Send Request'}
        </button>
      </div>

      {response && (
        <div style={{
          ...cardStyle,
          background: '#e8f5e9',
          borderLeft: '4px solid #4caf50'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
            <h4>Response</h4>
            <span style={{ fontSize: '12px', color: '#666' }}>
              Latency: {response.latency_ms}ms
            </span>
          </div>

          {response.output && (
            <div style={{
              padding: '16px',
              background: 'white',
              borderRadius: '4px',
              marginBottom: '12px'
            }}>
              <strong>Output:</strong>
              <p style={{ marginTop: '8px', whiteSpace: 'pre-wrap' }}>{response.output}</p>
            </div>
          )}

          <details style={{ cursor: 'pointer' }}>
            <summary style={{ fontSize: '13px', color: '#666' }}>Raw Response</summary>
            <pre style={{
              background: '#1a1a2e',
              color: '#e0e0e0',
              padding: '16px',
              borderRadius: '4px',
              overflow: 'auto',
              fontSize: '13px',
              marginTop: '8px'
            }}>
              {JSON.stringify(response, null, 2)}
            </pre>
          </details>
        </div>
      )}

      {history.length > 0 && (
        <div style={cardStyle}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <h3>Recent Requests</h3>
            <button
              onClick={() => setHistory([])}
              style={{
                padding: '4px 8px',
                background: '#f5f5f5',
                border: '1px solid #ddd',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '12px'
              }}
            >
              Clear History
            </button>
          </div>

          <div style={{ maxHeight: '300px', overflow: 'auto' }}>
            {history.map((item, idx) => (
              <div
                key={idx}
                style={{
                  padding: '12px',
                  background: idx % 2 === 0 ? '#f9f9f9' : 'white',
                  borderRadius: '4px',
                  marginBottom: '8px',
                  fontSize: '13px'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <span style={{ color: '#666' }}>
                    {new Date(item.timestamp).toLocaleTimeString()}
                  </span>
                  <span style={{ color: '#666' }}>
                    {item.latency_ms}ms
                  </span>
                </div>
                <div style={{ marginBottom: '4px' }}>
                  <strong>Input:</strong> {item.input.substring(0, 100)}{item.input.length > 100 ? '...' : ''}
                </div>
                <div style={{ color: '#2e7d32' }}>
                  <strong>Output:</strong> {item.output?.substring(0, 100)}{item.output?.length > 100 ? '...' : ''}
                </div>
              </div>
            ))}
          </div>
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
{`# Using Dashboard API
curl -X POST http://localhost:8001/api/predict \\
  -H "Content-Type: application/json" \\
  -d '{"input": "${input.replace(/'/g, "\\'").replace(/\n/g, '\\n')}"${selectedService ? `, "service_id": "${selectedService}"` : ''}}'

# Direct service call (if running)
${selectedInfo ? `curl -X POST http://localhost:${selectedInfo.service.port}/predict \\
  -H "Content-Type: application/json" \\
  -d '{"input": "${input.replace(/'/g, "\\'").replace(/\n/g, '\\n')}"}'` : '# No service selected'}`}
        </pre>
      </div>
    </div>
  )
}
