import React, { useState, useEffect } from 'react'
import { getServices, createService, startService, stopService, deleteService, getModels } from '../api/client'

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

const inputStyle = {
  width: '100%',
  padding: '10px 12px',
  border: '1px solid #ddd',
  borderRadius: '4px',
  fontSize: '14px',
  marginBottom: '12px'
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

export default function ServicesPage() {
  const [services, setServices] = useState([])
  const [models, setModels] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [showCreate, setShowCreate] = useState(false)
  const [createForm, setCreateForm] = useState({
    name: '',
    model_id: '',
    port: 8000,
    replicas: 1
  })

  const loadData = async () => {
    try {
      setLoading(true)
      const [servicesData, modelsData] = await Promise.all([
        getServices(),
        getModels()
      ])
      setServices(servicesData)
      setModels(modelsData)
      setError(null)
    } catch (err) {
      setError(err.message)
      setServices([])
      setModels([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
    const interval = setInterval(loadData, 10000) // Poll every 10 seconds
    return () => clearInterval(interval)
  }, [])

  const handleCreate = async (e) => {
    e.preventDefault()
    try {
      await createService(createForm)
      setShowCreate(false)
      setCreateForm({ name: '', model_id: '', port: 8000, replicas: 1 })
      loadData()
    } catch (err) {
      alert(`Failed to create service: ${err.message}`)
    }
  }

  const handleStart = async (serviceId) => {
    try {
      await startService(serviceId)
      loadData()
    } catch (err) {
      alert(`Failed to start: ${err.message}`)
    }
  }

  const handleStop = async (serviceId) => {
    try {
      await stopService(serviceId)
      loadData()
    } catch (err) {
      alert(`Failed to stop: ${err.message}`)
    }
  }

  const handleDelete = async (serviceId) => {
    if (!confirm('Are you sure you want to delete this service?')) return
    try {
      await deleteService(serviceId)
      loadData()
    } catch (err) {
      alert(`Failed to delete: ${err.message}`)
    }
  }

  const formatUptime = (startedAt) => {
    if (!startedAt) return '-'
    const start = new Date(startedAt)
    const now = new Date()
    const diff = Math.floor((now - start) / 1000)

    if (diff < 60) return `${diff}s`
    if (diff < 3600) return `${Math.floor(diff / 60)}m ${diff % 60}s`
    const hours = Math.floor(diff / 3600)
    const mins = Math.floor((diff % 3600) / 60)
    return `${hours}h ${mins}m`
  }

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <h1>Services</h1>
        <div>
          <button style={buttonStyle} onClick={() => setShowCreate(!showCreate)}>
            {showCreate ? 'Cancel' : '+ Deploy New Service'}
          </button>
          <button style={{ ...buttonStyle, background: '#666' }} onClick={loadData}>
            Refresh
          </button>
        </div>
      </div>

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

      {showCreate && (
        <div style={{ ...cardStyle, marginBottom: '24px' }}>
          <h3 style={{ marginBottom: '16px' }}>Deploy New Service</h3>
          <form onSubmit={handleCreate}>
            <label style={{ display: 'block', marginBottom: '6px', fontWeight: '500' }}>Service Name</label>
            <input
              type="text"
              value={createForm.name}
              onChange={(e) => setCreateForm({ ...createForm, name: e.target.value })}
              style={inputStyle}
              placeholder="my-inference-service"
              required
            />

            <label style={{ display: 'block', marginBottom: '6px', fontWeight: '500' }}>Model</label>
            <select
              value={createForm.model_id}
              onChange={(e) => setCreateForm({ ...createForm, model_id: e.target.value })}
              style={{ ...inputStyle, cursor: 'pointer' }}
              required
            >
              <option value="">Select a model...</option>
              {models.map(model => (
                <option key={model.id} value={model.id}>
                  {model.name} ({model.model_type})
                </option>
              ))}
            </select>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
              <div>
                <label style={{ display: 'block', marginBottom: '6px', fontWeight: '500' }}>Port</label>
                <input
                  type="number"
                  value={createForm.port}
                  onChange={(e) => setCreateForm({ ...createForm, port: parseInt(e.target.value) })}
                  style={inputStyle}
                  min="1024"
                  max="65535"
                />
              </div>
              <div>
                <label style={{ display: 'block', marginBottom: '6px', fontWeight: '500' }}>Replicas</label>
                <input
                  type="number"
                  value={createForm.replicas}
                  onChange={(e) => setCreateForm({ ...createForm, replicas: parseInt(e.target.value) })}
                  style={inputStyle}
                  min="1"
                  max="10"
                />
              </div>
            </div>

            <button type="submit" style={{ ...buttonStyle, background: '#2e7d32' }}>
              Deploy Service
            </button>
          </form>
        </div>
      )}

      {loading ? (
        <div style={cardStyle}>Loading services...</div>
      ) : services.length === 0 ? (
        <div style={cardStyle}>
          <h3>No Services Deployed</h3>
          <p style={{ color: '#666', marginTop: '8px' }}>
            Deploy a service to start serving your models. Services provide HTTP endpoints for inference.
          </p>
        </div>
      ) : (
        services.map(service => (
          <div key={service.id} style={cardStyle}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <div>
                <h3 style={{ marginBottom: '8px' }}>{service.name}</h3>
                <p style={{ color: '#666', fontSize: '14px', marginBottom: '12px' }}>
                  Model: <code>{service.model_id || service.model}</code> | Port: {service.port}
                </p>
                <div style={{ display: 'flex', gap: '24px', fontSize: '14px', color: '#555' }}>
                  <span>Requests: <strong>{(service.request_count || 0).toLocaleString()}</strong></span>
                  <span>Uptime: <strong>{formatUptime(service.started_at)}</strong></span>
                  {service.replicas > 1 && (
                    <span>Replicas: <strong>{service.replicas}</strong></span>
                  )}
                </div>
              </div>
              <div style={{ textAlign: 'right' }}>
                <span style={statusBadge(service.status)}>{service.status}</span>
                <div style={{ marginTop: '12px' }}>
                  {service.status === 'running' ? (
                    <button
                      style={{ ...buttonStyle, background: '#c62828' }}
                      onClick={() => handleStop(service.id)}
                    >
                      Stop
                    </button>
                  ) : (
                    <button
                      style={{ ...buttonStyle, background: '#2e7d32' }}
                      onClick={() => handleStart(service.id)}
                    >
                      Start
                    </button>
                  )}
                  <button
                    style={{ ...buttonStyle, background: '#c62828' }}
                    onClick={() => handleDelete(service.id)}
                    disabled={service.status === 'running'}
                    title={service.status === 'running' ? 'Stop the service before deleting' : ''}
                  >
                    Delete
                  </button>
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

            {service.error && (
              <div style={{
                marginTop: '16px',
                padding: '12px',
                background: '#ffebee',
                borderRadius: '4px',
                fontSize: '13px',
                color: '#c62828'
              }}>
                <strong>Error:</strong> {service.error}
              </div>
            )}
          </div>
        ))
      )}

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
{`# List services
curl http://localhost:8001/api/services

# Create a new service
curl -X POST http://localhost:8001/api/services \\
  -H "Content-Type: application/json" \\
  -d '{"name": "my-service", "model_id": "model-uuid", "port": 8000}'

# Start a service
curl -X POST http://localhost:8001/api/services/{service_id}/start

# Stop a service
curl -X POST http://localhost:8001/api/services/{service_id}/stop`}
        </pre>
      </div>
    </div>
  )
}
