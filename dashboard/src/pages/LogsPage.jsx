import React, { useState, useEffect } from 'react'

const cardStyle = {
  background: 'white',
  borderRadius: '8px',
  padding: '24px',
  boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
  marginBottom: '24px'
}

const mockLogs = [
  { timestamp: '2024-01-15 10:30:15', level: 'INFO', service: 'demo-service', message: 'Server started on port 8000' },
  { timestamp: '2024-01-15 10:30:16', level: 'INFO', service: 'demo-service', message: 'Model loaded: model_stub.json' },
  { timestamp: '2024-01-15 10:31:02', level: 'INFO', service: 'demo-service', message: 'POST /predict - 200 OK - 12ms' },
  { timestamp: '2024-01-15 10:31:45', level: 'INFO', service: 'demo-service', message: 'POST /predict - 200 OK - 8ms' },
  { timestamp: '2024-01-15 10:32:10', level: 'WARN', service: 'demo-service', message: 'Request timeout increased to 30s' },
  { timestamp: '2024-01-15 10:33:22', level: 'INFO', service: 'demo-service', message: 'POST /predict - 200 OK - 15ms' },
  { timestamp: '2024-01-15 10:34:01', level: 'ERROR', service: 'qa-model-svc', message: 'Failed to start: Port 8001 already in use' },
  { timestamp: '2024-01-15 10:35:00', level: 'INFO', service: 'demo-service', message: 'Health check passed' },
  { timestamp: '2024-01-15 10:36:12', level: 'INFO', service: 'demo-service', message: 'POST /predict - 200 OK - 10ms' },
]

const levelColors = {
  INFO: '#1565c0',
  WARN: '#ef6c00',
  ERROR: '#c62828',
  DEBUG: '#666'
}

export default function LogsPage() {
  const [logs, setLogs] = useState(mockLogs)
  const [filter, setFilter] = useState('all')
  const [serviceFilter, setServiceFilter] = useState('all')

  const filteredLogs = logs.filter(log => {
    if (filter !== 'all' && log.level !== filter) return false
    if (serviceFilter !== 'all' && log.service !== serviceFilter) return false
    return true
  })

  const services = [...new Set(logs.map(l => l.service))]

  return (
    <div>
      <h1 style={{ marginBottom: '24px' }}>Logs</h1>

      <div style={cardStyle}>
        <div style={{ display: 'flex', gap: '16px', marginBottom: '16px' }}>
          <div>
            <label style={{ display: 'block', marginBottom: '4px', fontSize: '13px', color: '#666' }}>
              Level
            </label>
            <select
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              style={{
                padding: '8px 12px',
                borderRadius: '4px',
                border: '1px solid #ddd',
                fontSize: '14px'
              }}
            >
              <option value="all">All Levels</option>
              <option value="INFO">INFO</option>
              <option value="WARN">WARN</option>
              <option value="ERROR">ERROR</option>
            </select>
          </div>
          <div>
            <label style={{ display: 'block', marginBottom: '4px', fontSize: '13px', color: '#666' }}>
              Service
            </label>
            <select
              value={serviceFilter}
              onChange={(e) => setServiceFilter(e.target.value)}
              style={{
                padding: '8px 12px',
                borderRadius: '4px',
                border: '1px solid #ddd',
                fontSize: '14px'
              }}
            >
              <option value="all">All Services</option>
              {services.map(svc => (
                <option key={svc} value={svc}>{svc}</option>
              ))}
            </select>
          </div>
          <div style={{ marginLeft: 'auto', alignSelf: 'flex-end' }}>
            <button
              onClick={() => setLogs([...mockLogs])}
              style={{
                padding: '8px 16px',
                background: '#f5f5f5',
                border: '1px solid #ddd',
                borderRadius: '4px',
                cursor: 'pointer'
              }}
            >
              Refresh
            </button>
          </div>
        </div>

        <div style={{
          background: '#1a1a2e',
          borderRadius: '4px',
          padding: '16px',
          maxHeight: '500px',
          overflow: 'auto',
          fontFamily: 'Monaco, Consolas, monospace',
          fontSize: '13px',
          lineHeight: '1.6'
        }}>
          {filteredLogs.length === 0 ? (
            <div style={{ color: '#666' }}>No logs matching filters</div>
          ) : (
            filteredLogs.map((log, i) => (
              <div key={i} style={{ marginBottom: '4px' }}>
                <span style={{ color: '#666' }}>{log.timestamp}</span>
                {' '}
                <span style={{
                  color: levelColors[log.level],
                  fontWeight: '500'
                }}>
                  [{log.level}]
                </span>
                {' '}
                <span style={{ color: '#888' }}>[{log.service}]</span>
                {' '}
                <span style={{ color: '#e0e0e0' }}>{log.message}</span>
              </div>
            ))
          )}
        </div>
      </div>

      <div style={cardStyle}>
        <h3 style={{ marginBottom: '12px' }}>CLI Equivalent</h3>
        <pre style={{
          background: '#1a1a2e',
          color: '#e0e0e0',
          padding: '16px',
          borderRadius: '4px',
          overflow: 'auto',
          fontSize: '13px'
        }}>
{`# View logs from running service
docker logs -f tinyforge-inference

# Or with kubectl
kubectl logs -f -l app=tinyforge -n tinyforge`}
        </pre>
      </div>
    </div>
  )
}
