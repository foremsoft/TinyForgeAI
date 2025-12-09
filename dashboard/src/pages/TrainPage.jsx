import React, { useState, useEffect } from 'react'
import { createJob, getJobs, cancelJob, deleteJob, createWebSocket } from '../api/client'

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

const labelStyle = {
  display: 'block',
  marginBottom: '6px',
  fontWeight: '500',
  color: '#555'
}

const statusBadge = (status) => ({
  display: 'inline-block',
  padding: '4px 12px',
  borderRadius: '12px',
  fontSize: '12px',
  fontWeight: '500',
  background: status === 'completed' ? '#e8f5e9' :
              status === 'running' ? '#e3f2fd' :
              status === 'failed' ? '#ffebee' :
              status === 'cancelled' ? '#fafafa' : '#fff3e0',
  color: status === 'completed' ? '#2e7d32' :
         status === 'running' ? '#1565c0' :
         status === 'failed' ? '#c62828' :
         status === 'cancelled' ? '#666' : '#ef6c00'
})

export default function TrainPage() {
  const [dataPath, setDataPath] = useState('examples/data/demo_dataset.jsonl')
  const [outputDir, setOutputDir] = useState('./tmp/model')
  const [baseModel, setBaseModel] = useState('google/flan-t5-small')
  const [epochs, setEpochs] = useState(3)
  const [batchSize, setBatchSize] = useState(4)
  const [learningRate, setLearningRate] = useState(0.0001)
  const [useLora, setUseLora] = useState(false)
  const [loraRank, setLoraRank] = useState(8)

  const [status, setStatus] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [jobs, setJobs] = useState([])
  const [error, setError] = useState(null)

  const loadJobs = async () => {
    try {
      const data = await getJobs(null, 10)
      setJobs(data)
      setError(null)
    } catch (err) {
      setError(err.message)
      setJobs([])
    }
  }

  useEffect(() => {
    loadJobs()

    // Set up WebSocket for real-time updates
    const ws = createWebSocket('jobs')

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      if (data.type === 'job_update') {
        loadJobs() // Reload jobs when update received
      }
    }

    ws.onerror = () => {
      console.log('WebSocket connection failed, using polling')
    }

    // Fallback polling every 5 seconds
    const interval = setInterval(loadJobs, 5000)

    return () => {
      ws.close()
      clearInterval(interval)
    }
  }, [])

  async function handleTrain() {
    setIsSubmitting(true)
    setStatus('Submitting training job...')
    setError(null)

    try {
      const jobData = {
        name: `train-${Date.now()}`,
        config: {
          dataset_path: dataPath,
          output_dir: outputDir,
          base_model: baseModel,
          epochs: epochs,
          batch_size: batchSize,
          learning_rate: learningRate,
          use_lora: useLora,
          lora_rank: useLora ? loraRank : null
        }
      }

      const result = await createJob(jobData)
      setStatus(`Job submitted! ID: ${result.id}`)
      loadJobs()
    } catch (err) {
      setError(err.message)
      setStatus('')
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleCancel = async (jobId) => {
    try {
      await cancelJob(jobId)
      loadJobs()
    } catch (err) {
      alert(`Failed to cancel: ${err.message}`)
    }
  }

  const handleDelete = async (jobId) => {
    if (!confirm('Delete this job?')) return
    try {
      await deleteJob(jobId)
      loadJobs()
    } catch (err) {
      alert(`Failed to delete: ${err.message}`)
    }
  }

  const formatDate = (dateStr) => {
    if (!dateStr) return 'N/A'
    return new Date(dateStr).toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  return (
    <div>
      <h1 style={{ marginBottom: '24px' }}>Train Model</h1>

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
        <h3 style={{ marginBottom: '16px' }}>Training Configuration</h3>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
          <div>
            <label style={labelStyle}>Dataset Path</label>
            <input
              type="text"
              value={dataPath}
              onChange={(e) => setDataPath(e.target.value)}
              style={inputStyle}
              placeholder="Path to JSONL dataset"
            />
          </div>

          <div>
            <label style={labelStyle}>Output Directory</label>
            <input
              type="text"
              value={outputDir}
              onChange={(e) => setOutputDir(e.target.value)}
              style={inputStyle}
              placeholder="Output directory for model"
            />
          </div>

          <div>
            <label style={labelStyle}>Base Model</label>
            <select
              value={baseModel}
              onChange={(e) => setBaseModel(e.target.value)}
              style={{ ...inputStyle, cursor: 'pointer' }}
            >
              <option value="google/flan-t5-small">Flan-T5 Small (77M)</option>
              <option value="google/flan-t5-base">Flan-T5 Base (250M)</option>
              <option value="distilgpt2">DistilGPT-2 (82M)</option>
              <option value="gpt2">GPT-2 (124M)</option>
            </select>
          </div>

          <div>
            <label style={labelStyle}>Epochs</label>
            <input
              type="number"
              value={epochs}
              onChange={(e) => setEpochs(parseInt(e.target.value) || 1)}
              style={inputStyle}
              min="1"
              max="100"
            />
          </div>

          <div>
            <label style={labelStyle}>Batch Size</label>
            <input
              type="number"
              value={batchSize}
              onChange={(e) => setBatchSize(parseInt(e.target.value) || 1)}
              style={inputStyle}
              min="1"
              max="64"
            />
          </div>

          <div>
            <label style={labelStyle}>Learning Rate</label>
            <input
              type="number"
              value={learningRate}
              onChange={(e) => setLearningRate(parseFloat(e.target.value) || 0.0001)}
              style={inputStyle}
              step="0.00001"
              min="0.000001"
              max="0.1"
            />
          </div>
        </div>

        <div style={{ marginTop: '16px', marginBottom: '16px' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={useLora}
              onChange={(e) => setUseLora(e.target.checked)}
            />
            Use LoRA Adapter (Parameter-Efficient Fine-Tuning)
          </label>
        </div>

        {useLora && (
          <div style={{ marginBottom: '16px' }}>
            <label style={labelStyle}>LoRA Rank</label>
            <input
              type="number"
              value={loraRank}
              onChange={(e) => setLoraRank(parseInt(e.target.value) || 8)}
              style={{ ...inputStyle, maxWidth: '120px' }}
              min="4"
              max="64"
            />
          </div>
        )}

        <button
          onClick={handleTrain}
          disabled={isSubmitting}
          style={{
            ...buttonStyle,
            opacity: isSubmitting ? 0.7 : 1
          }}
        >
          {isSubmitting ? 'Submitting...' : 'Start Training'}
        </button>
      </div>

      {status && (
        <div style={{
          ...cardStyle,
          background: status.includes('submitted') ? '#e8f5e9' : '#fff3e0',
          borderLeft: `4px solid ${status.includes('submitted') ? '#4caf50' : '#ff9800'}`
        }}>
          <h4 style={{ marginBottom: '8px' }}>Status</h4>
          <p style={{ fontFamily: 'monospace' }}>{status}</p>
        </div>
      )}

      {/* Recent Jobs */}
      <div style={cardStyle}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <h3>Recent Training Jobs</h3>
          <button
            onClick={loadJobs}
            style={{
              padding: '6px 12px',
              background: '#f5f5f5',
              border: '1px solid #ddd',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '13px'
            }}
          >
            Refresh
          </button>
        </div>

        {jobs.length === 0 ? (
          <p style={{ color: '#666' }}>No training jobs yet. Start a new training job above.</p>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #eee' }}>
                <th style={{ textAlign: 'left', padding: '8px', fontSize: '13px', color: '#666' }}>Job ID</th>
                <th style={{ textAlign: 'left', padding: '8px', fontSize: '13px', color: '#666' }}>Name</th>
                <th style={{ textAlign: 'left', padding: '8px', fontSize: '13px', color: '#666' }}>Status</th>
                <th style={{ textAlign: 'left', padding: '8px', fontSize: '13px', color: '#666' }}>Progress</th>
                <th style={{ textAlign: 'left', padding: '8px', fontSize: '13px', color: '#666' }}>Created</th>
                <th style={{ textAlign: 'right', padding: '8px', fontSize: '13px', color: '#666' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {jobs.map(job => (
                <tr key={job.id} style={{ borderBottom: '1px solid #eee' }}>
                  <td style={{ padding: '12px 8px' }}>
                    <code style={{ fontSize: '12px' }}>{job.id.slice(0, 8)}...</code>
                  </td>
                  <td style={{ padding: '12px 8px' }}>{job.name || 'Unnamed'}</td>
                  <td style={{ padding: '12px 8px' }}>
                    <span style={statusBadge(job.status)}>{job.status}</span>
                  </td>
                  <td style={{ padding: '12px 8px' }}>
                    {job.status === 'running' && (
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <div style={{
                          width: '100px',
                          height: '6px',
                          background: '#e0e0e0',
                          borderRadius: '3px',
                          overflow: 'hidden'
                        }}>
                          <div style={{
                            width: `${(job.progress || 0) * 100}%`,
                            height: '100%',
                            background: '#1565c0',
                            transition: 'width 0.3s'
                          }} />
                        </div>
                        <span style={{ fontSize: '12px', color: '#666' }}>
                          {Math.round((job.progress || 0) * 100)}%
                        </span>
                      </div>
                    )}
                    {job.status !== 'running' && '-'}
                  </td>
                  <td style={{ padding: '12px 8px', fontSize: '13px', color: '#666' }}>
                    {formatDate(job.created_at)}
                  </td>
                  <td style={{ padding: '12px 8px', textAlign: 'right' }}>
                    {job.status === 'running' && (
                      <button
                        onClick={() => handleCancel(job.id)}
                        style={{
                          padding: '4px 8px',
                          background: '#ff9800',
                          color: 'white',
                          border: 'none',
                          borderRadius: '4px',
                          cursor: 'pointer',
                          fontSize: '12px',
                          marginRight: '4px'
                        }}
                      >
                        Cancel
                      </button>
                    )}
                    {['completed', 'failed', 'cancelled'].includes(job.status) && (
                      <button
                        onClick={() => handleDelete(job.id)}
                        style={{
                          padding: '4px 8px',
                          background: '#c62828',
                          color: 'white',
                          border: 'none',
                          borderRadius: '4px',
                          cursor: 'pointer',
                          fontSize: '12px'
                        }}
                      >
                        Delete
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div style={cardStyle}>
        <h3 style={{ marginBottom: '16px' }}>CLI Equivalent</h3>
        <pre style={{
          background: '#1a1a2e',
          color: '#e0e0e0',
          padding: '16px',
          borderRadius: '4px',
          overflow: 'auto'
        }}>
{`tinyforge train \\
  --data ${dataPath} \\
  --out ${outputDir} \\
  --base-model ${baseModel} \\
  --epochs ${epochs} \\
  --batch-size ${batchSize} \\
  --lr ${learningRate}${useLora ? ` \\
  --use-lora --lora-rank ${loraRank}` : ''}`}
        </pre>
      </div>
    </div>
  )
}
