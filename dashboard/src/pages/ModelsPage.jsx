import React, { useState, useEffect } from 'react'
import { getModels, deployModel, undeployModel, deleteModel, registerModel } from '../api/client'

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

const statusBadge = (isDeployed) => ({
  display: 'inline-block',
  padding: '4px 12px',
  borderRadius: '12px',
  fontSize: '12px',
  fontWeight: '500',
  background: isDeployed ? '#e8f5e9' : '#f5f5f5',
  color: isDeployed ? '#2e7d32' : '#666'
})

const gridStyle = {
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fill, minmax(350px, 1fr))',
  gap: '16px'
}

export default function ModelsPage() {
  const [models, setModels] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [showRegister, setShowRegister] = useState(false)
  const [registerForm, setRegisterForm] = useState({
    name: '',
    description: '',
    model_type: 'seq2seq',
    path: '',
    size_bytes: 0
  })

  const loadModels = async () => {
    try {
      setLoading(true)
      const data = await getModels()
      setModels(data)
      setError(null)
    } catch (err) {
      setError(err.message)
      // Show empty state with mock data if API unavailable
      setModels([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadModels()
  }, [])

  const handleDeploy = async (modelId) => {
    try {
      await deployModel(modelId)
      loadModels()
    } catch (err) {
      alert(`Failed to deploy: ${err.message}`)
    }
  }

  const handleUndeploy = async (modelId) => {
    try {
      await undeployModel(modelId)
      loadModels()
    } catch (err) {
      alert(`Failed to undeploy: ${err.message}`)
    }
  }

  const handleDelete = async (modelId) => {
    if (!confirm('Are you sure you want to delete this model?')) return
    try {
      await deleteModel(modelId)
      loadModels()
    } catch (err) {
      alert(`Failed to delete: ${err.message}`)
    }
  }

  const handleRegister = async (e) => {
    e.preventDefault()
    try {
      await registerModel(registerForm)
      setShowRegister(false)
      setRegisterForm({ name: '', description: '', model_type: 'seq2seq', path: '', size_bytes: 0 })
      loadModels()
    } catch (err) {
      alert(`Failed to register: ${err.message}`)
    }
  }

  const formatDate = (dateStr) => {
    if (!dateStr) return 'N/A'
    return new Date(dateStr).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <h1>Model Registry</h1>
        <div>
          <button style={buttonStyle} onClick={() => setShowRegister(!showRegister)}>
            {showRegister ? 'Cancel' : '+ Register Model'}
          </button>
          <button style={{ ...buttonStyle, background: '#666' }} onClick={loadModels}>
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

      {showRegister && (
        <div style={{ ...cardStyle, marginBottom: '24px' }}>
          <h3 style={{ marginBottom: '16px' }}>Register New Model</h3>
          <form onSubmit={handleRegister}>
            <label style={{ display: 'block', marginBottom: '6px', fontWeight: '500' }}>Name</label>
            <input
              type="text"
              value={registerForm.name}
              onChange={(e) => setRegisterForm({ ...registerForm, name: e.target.value })}
              style={inputStyle}
              placeholder="my-fine-tuned-model"
              required
            />

            <label style={{ display: 'block', marginBottom: '6px', fontWeight: '500' }}>Description</label>
            <input
              type="text"
              value={registerForm.description}
              onChange={(e) => setRegisterForm({ ...registerForm, description: e.target.value })}
              style={inputStyle}
              placeholder="A fine-tuned model for Q&A"
            />

            <label style={{ display: 'block', marginBottom: '6px', fontWeight: '500' }}>Model Type</label>
            <select
              value={registerForm.model_type}
              onChange={(e) => setRegisterForm({ ...registerForm, model_type: e.target.value })}
              style={{ ...inputStyle, cursor: 'pointer' }}
            >
              <option value="seq2seq">Seq2Seq (T5, BART)</option>
              <option value="causal">Causal LM (GPT-2, LLaMA)</option>
              <option value="classification">Classification</option>
              <option value="embedding">Embedding</option>
            </select>

            <label style={{ display: 'block', marginBottom: '6px', fontWeight: '500' }}>Path</label>
            <input
              type="text"
              value={registerForm.path}
              onChange={(e) => setRegisterForm({ ...registerForm, path: e.target.value })}
              style={inputStyle}
              placeholder="./models/my-model"
              required
            />

            <button type="submit" style={{ ...buttonStyle, background: '#2e7d32' }}>
              Register Model
            </button>
          </form>
        </div>
      )}

      {loading ? (
        <div style={cardStyle}>Loading models...</div>
      ) : models.length === 0 ? (
        <div style={cardStyle}>
          <h3>No Models Registered</h3>
          <p style={{ color: '#666', marginTop: '8px' }}>
            Register your first model to get started. Models can be created through training jobs
            or registered manually.
          </p>
        </div>
      ) : (
        <div style={gridStyle}>
          {models.map(model => (
            <div key={model.id} style={cardStyle}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '12px' }}>
                <div>
                  <h3 style={{ marginBottom: '4px' }}>{model.name}</h3>
                  <code style={{ fontSize: '12px', color: '#666' }}>{model.id}</code>
                </div>
                <span style={statusBadge(model.is_deployed)}>
                  {model.is_deployed ? 'Deployed' : 'Not Deployed'}
                </span>
              </div>

              {model.description && (
                <p style={{ color: '#555', fontSize: '14px', marginBottom: '12px' }}>
                  {model.description}
                </p>
              )}

              <div style={{ fontSize: '13px', color: '#666', marginBottom: '12px' }}>
                <div style={{ marginBottom: '4px' }}>
                  <strong>Type:</strong> {model.model_type || 'Unknown'}
                </div>
                <div style={{ marginBottom: '4px' }}>
                  <strong>Path:</strong> <code>{model.path}</code>
                </div>
                <div style={{ marginBottom: '4px' }}>
                  <strong>Size:</strong> {model.size || 'N/A'}
                </div>
                <div>
                  <strong>Created:</strong> {formatDate(model.created_at)}
                </div>
              </div>

              <div style={{ display: 'flex', gap: '8px' }}>
                {model.is_deployed ? (
                  <button
                    style={{ ...buttonStyle, background: '#ff9800' }}
                    onClick={() => handleUndeploy(model.id)}
                  >
                    Undeploy
                  </button>
                ) : (
                  <button
                    style={{ ...buttonStyle, background: '#2e7d32' }}
                    onClick={() => handleDeploy(model.id)}
                  >
                    Deploy
                  </button>
                )}
                <button
                  style={{ ...buttonStyle, background: '#c62828' }}
                  onClick={() => handleDelete(model.id)}
                  disabled={model.is_deployed}
                  title={model.is_deployed ? 'Undeploy before deleting' : ''}
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      <div style={{ ...cardStyle, marginTop: '24px' }}>
        <h3 style={{ marginBottom: '12px' }}>CLI Equivalent</h3>
        <pre style={{
          background: '#1a1a2e',
          color: '#e0e0e0',
          padding: '16px',
          borderRadius: '4px',
          overflow: 'auto',
          fontSize: '13px'
        }}>
{`# List registered models
curl http://localhost:8001/api/models

# Register a new model
curl -X POST http://localhost:8001/api/models \\
  -H "Content-Type: application/json" \\
  -d '{"name": "my-model", "path": "./models/my-model"}'

# Deploy a model
curl -X POST http://localhost:8001/api/models/{model_id}/deploy`}
        </pre>
      </div>
    </div>
  )
}
