import React, { useState, useEffect, useRef } from 'react'
import { getLogs, createWebSocket } from '../api/client'

const cardStyle = {
  background: 'white',
  borderRadius: '8px',
  padding: '24px',
  boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
  marginBottom: '24px'
}

const levelColors = {
  INFO: '#1565c0',
  WARN: '#ef6c00',
  WARNING: '#ef6c00',
  ERROR: '#c62828',
  DEBUG: '#666',
  CRITICAL: '#b71c1c'
}

export default function LogsPage() {
  const [logs, setLogs] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [levelFilter, setLevelFilter] = useState('all')
  const [sourceFilter, setSourceFilter] = useState('all')
  const [autoScroll, setAutoScroll] = useState(true)
  const [limit, setLimit] = useState(100)
  const logContainerRef = useRef(null)

  const loadLogs = async () => {
    try {
      setLoading(true)
      const data = await getLogs(
        levelFilter !== 'all' ? levelFilter : null,
        sourceFilter !== 'all' ? sourceFilter : null,
        limit
      )
      setLogs(data)
      setError(null)
    } catch (err) {
      setError(err.message)
      setLogs([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadLogs()

    // Set up WebSocket for real-time logs
    const ws = createWebSocket('logs')

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      if (data.type === 'log') {
        setLogs(prev => [...prev, data.log].slice(-limit))
      }
    }

    ws.onerror = () => {
      console.log('WebSocket connection failed, using polling')
    }

    // Fallback polling every 5 seconds
    const interval = setInterval(loadLogs, 5000)

    return () => {
      ws.close()
      clearInterval(interval)
    }
  }, [levelFilter, sourceFilter, limit])

  useEffect(() => {
    if (autoScroll && logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight
    }
  }, [logs, autoScroll])

  const filteredLogs = logs.filter(log => {
    if (levelFilter !== 'all' && log.level !== levelFilter) return false
    if (sourceFilter !== 'all' && log.source !== sourceFilter) return false
    return true
  })

  const sources = [...new Set(logs.map(l => l.source).filter(Boolean))]

  const formatTimestamp = (ts) => {
    if (!ts) return ''
    const date = new Date(ts)
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false
    })
  }

  const downloadLogs = () => {
    const content = filteredLogs.map(log =>
      `${log.timestamp || ''} [${log.level}] [${log.source || 'unknown'}] ${log.message}`
    ).join('\n')

    const blob = new Blob([content], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `tinyforge-logs-${new Date().toISOString().split('T')[0]}.txt`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div>
      <h1 style={{ marginBottom: '24px' }}>Logs</h1>

      {error && (
        <div style={{
          ...cardStyle,
          background: '#fff3e0',
          borderLeft: '4px solid #ff9800',
          marginBottom: '24px'
        }}>
          <strong>API Connection Issue:</strong> {error}
          <br />
          <small>Make sure the Dashboard API is running on localhost:8001</small>
        </div>
      )}

      <div style={cardStyle}>
        <div style={{ display: 'flex', gap: '16px', marginBottom: '16px', flexWrap: 'wrap', alignItems: 'flex-end' }}>
          <div>
            <label style={{ display: 'block', marginBottom: '4px', fontSize: '13px', color: '#666' }}>
              Level
            </label>
            <select
              value={levelFilter}
              onChange={(e) => setLevelFilter(e.target.value)}
              style={{
                padding: '8px 12px',
                borderRadius: '4px',
                border: '1px solid #ddd',
                fontSize: '14px'
              }}
            >
              <option value="all">All Levels</option>
              <option value="DEBUG">DEBUG</option>
              <option value="INFO">INFO</option>
              <option value="WARNING">WARNING</option>
              <option value="ERROR">ERROR</option>
              <option value="CRITICAL">CRITICAL</option>
            </select>
          </div>

          <div>
            <label style={{ display: 'block', marginBottom: '4px', fontSize: '13px', color: '#666' }}>
              Source
            </label>
            <select
              value={sourceFilter}
              onChange={(e) => setSourceFilter(e.target.value)}
              style={{
                padding: '8px 12px',
                borderRadius: '4px',
                border: '1px solid #ddd',
                fontSize: '14px'
              }}
            >
              <option value="all">All Sources</option>
              {sources.map(src => (
                <option key={src} value={src}>{src}</option>
              ))}
            </select>
          </div>

          <div>
            <label style={{ display: 'block', marginBottom: '4px', fontSize: '13px', color: '#666' }}>
              Limit
            </label>
            <select
              value={limit}
              onChange={(e) => setLimit(parseInt(e.target.value))}
              style={{
                padding: '8px 12px',
                borderRadius: '4px',
                border: '1px solid #ddd',
                fontSize: '14px'
              }}
            >
              <option value="50">50 logs</option>
              <option value="100">100 logs</option>
              <option value="200">200 logs</option>
              <option value="500">500 logs</option>
            </select>
          </div>

          <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={autoScroll}
              onChange={(e) => setAutoScroll(e.target.checked)}
            />
            Auto-scroll
          </label>

          <div style={{ marginLeft: 'auto', display: 'flex', gap: '8px' }}>
            <button
              onClick={downloadLogs}
              style={{
                padding: '8px 16px',
                background: '#f5f5f5',
                border: '1px solid #ddd',
                borderRadius: '4px',
                cursor: 'pointer'
              }}
            >
              Download
            </button>
            <button
              onClick={loadLogs}
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

        <div
          ref={logContainerRef}
          style={{
            background: '#1a1a2e',
            borderRadius: '4px',
            padding: '16px',
            maxHeight: '500px',
            overflow: 'auto',
            fontFamily: 'Monaco, Consolas, monospace',
            fontSize: '13px',
            lineHeight: '1.6'
          }}
        >
          {loading && logs.length === 0 ? (
            <div style={{ color: '#666' }}>Loading logs...</div>
          ) : filteredLogs.length === 0 ? (
            <div style={{ color: '#666' }}>No logs matching filters</div>
          ) : (
            filteredLogs.map((log, i) => (
              <div key={log.id || i} style={{ marginBottom: '4px' }}>
                <span style={{ color: '#666' }}>{formatTimestamp(log.timestamp)}</span>
                {' '}
                <span style={{
                  color: levelColors[log.level] || '#888',
                  fontWeight: '500'
                }}>
                  [{log.level}]
                </span>
                {' '}
                {log.source && (
                  <>
                    <span style={{ color: '#888' }}>[{log.source}]</span>
                    {' '}
                  </>
                )}
                <span style={{ color: '#e0e0e0' }}>{log.message}</span>
                {log.details && (
                  <details style={{ marginLeft: '20px', marginTop: '4px' }}>
                    <summary style={{ color: '#666', cursor: 'pointer' }}>Details</summary>
                    <pre style={{ color: '#aaa', marginTop: '4px' }}>
                      {JSON.stringify(log.details, null, 2)}
                    </pre>
                  </details>
                )}
              </div>
            ))
          )}
        </div>

        <div style={{ marginTop: '12px', fontSize: '12px', color: '#666', display: 'flex', justifyContent: 'space-between' }}>
          <span>Showing {filteredLogs.length} of {logs.length} logs</span>
          {logs.length > 0 && logs[0].timestamp && (
            <span>Latest: {new Date(logs[logs.length - 1]?.timestamp).toLocaleString()}</span>
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
{`# Fetch logs via API
curl "http://localhost:8001/api/logs?limit=${limit}${levelFilter !== 'all' ? `&level=${levelFilter}` : ''}${sourceFilter !== 'all' ? `&source=${sourceFilter}` : ''}"

# View logs from running containers
docker logs -f tinyforge-api

# View logs with kubectl
kubectl logs -f -l app=tinyforge -n tinyforge

# Stream logs via WebSocket
wscat -c ws://localhost:8001/ws/logs`}
        </pre>
      </div>
    </div>
  )
}
