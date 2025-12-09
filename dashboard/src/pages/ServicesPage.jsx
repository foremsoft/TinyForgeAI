import React, { useState } from 'react'

const cardStyle = {
  background: 'white',
  borderRadius: '8px',
  padding: '24px',
  boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
  marginBottom: '16px'
}

const buttonStyle = {
  background: '#4a4a6a',
  color: 'white',
  border: 'none',
  padding: '8px 16px',
  borderRadius: '4px',
  cursor: 'pointer',
  fontSize: '13px',
  marginRight: '8px'
}

const statusBadge = (status) => ({
  display: 'inline-block',
  padding: '4px 12px',
  borderRadius: '12px',
  fontSize: '12px',
  fontWeight: '500',
  background: status === 'running' ? '#e8f5e9' : status === 'stopped' ? '#ffebee' : '#fff3e0',
  color: status === 'running' ? '#2e7d32' : status === 'stopped' ? '#c62828' : '#ef6c00'
})

const mockServices = [
  {
    id: 1,
    name: 'demo-service',
    model: 'model_stub.json',
    port: 8000,
    status: 'running',
    requests: 1247,
    uptime: '2h 34m'
  },
  {
    id: 2,
    name: 'qa-model-svc',
    model: 'qa_model_v2.json',
    port: 8001,
    status: 'stopped',
    requests: 0,
    uptime: '-'
  }
]

export default function ServicesPage() {
  const [services, setServices] = useState(mockServices)

  const toggleService = (id) => {
    setServices(services.map(svc =>
      svc.id === id
        ? { ...svc, status: svc.status === 'running' ? 'stopped' : 'running' }
        : svc
    ))
  }

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <h1>Services</h1>
        <button style={buttonStyle}>+ Deploy New Service</button>
      </div>

      {services.map(service => (
        <div key={service.id} style={cardStyle}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div>
              <h3 style={{ marginBottom: '8px' }}>{service.name}</h3>
              <p style={{ color: '#666', fontSize: '14px', marginBottom: '12px' }}>
                Model: <code>{service.model}</code> | Port: {service.port}
              </p>
              <div style={{ display: 'flex', gap: '24px', fontSize: '14px', color: '#555' }}>
                <span>Requests: <strong>{service.requests.toLocaleString()}</strong></span>
                <span>Uptime: <strong>{service.uptime}</strong></span>
              </div>
            </div>
            <div style={{ textAlign: 'right' }}>
              <span style={statusBadge(service.status)}>{service.status}</span>
              <div style={{ marginTop: '12px' }}>
                <button
                  style={{
                    ...buttonStyle,
                    background: service.status === 'running' ? '#c62828' : '#2e7d32'
                  }}
                  onClick={() => toggleService(service.id)}
                >
                  {service.status === 'running' ? 'Stop' : 'Start'}
                </button>
                <button style={{ ...buttonStyle, background: '#1565c0' }}>Logs</button>
              </div>
            </div>
          </div>

          {service.status === 'running' && (
            <div style={{
              marginTop: '16px',
              padding: '12px',
              background: '#f5f5f5',
              borderRadius: '4px',
              fontSize: '13px'
            }}>
              <strong>Endpoint:</strong>{' '}
              <code>http://localhost:{service.port}/predict</code>
              <button
                style={{
                  marginLeft: '12px',
                  background: 'none',
                  border: '1px solid #ccc',
                  padding: '2px 8px',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '12px'
                }}
                onClick={() => navigator.clipboard.writeText(`http://localhost:${service.port}/predict`)}
              >
                Copy
              </button>
            </div>
          )}
        </div>
      ))}

      <div style={cardStyle}>
        <h3 style={{ marginBottom: '16px' }}>CLI Equivalent</h3>
        <pre style={{
          background: '#1a1a2e',
          color: '#e0e0e0',
          padding: '16px',
          borderRadius: '4px',
          overflow: 'auto',
          fontSize: '13px'
        }}>
{`# Export model to service
foremforge export --model ./model/model_stub.json --out ./service

# Start service
foremforge serve --dir ./service --port 8000`}
        </pre>
      </div>
    </div>
  )
}
