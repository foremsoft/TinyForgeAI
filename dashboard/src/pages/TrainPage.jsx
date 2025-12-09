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

const labelStyle = {
  display: 'block',
  marginBottom: '6px',
  fontWeight: '500',
  color: '#555'
}

export default function TrainPage() {
  const [dataPath, setDataPath] = useState('examples/data/demo_dataset.jsonl')
  const [outputDir, setOutputDir] = useState('./tmp/model')
  const [dryRun, setDryRun] = useState(true)
  const [useLora, setUseLora] = useState(false)
  const [status, setStatus] = useState('')
  const [isTraining, setIsTraining] = useState(false)

  async function handleTrain() {
    setIsTraining(true)
    setStatus('Starting training...')

    // Simulate training progress
    setTimeout(() => setStatus('Loading dataset...'), 500)
    setTimeout(() => setStatus('Validating records...'), 1000)
    setTimeout(() => setStatus('Running dry-run training...'), 1500)
    setTimeout(() => {
      setStatus('Training complete! Model saved to: ' + outputDir + '/model_stub.json')
      setIsTraining(false)
    }, 2500)
  }

  return (
    <div>
      <h1 style={{ marginBottom: '24px' }}>Train Model</h1>

      <div style={cardStyle}>
        <h3 style={{ marginBottom: '16px' }}>Training Configuration</h3>

        <label style={labelStyle}>Dataset Path</label>
        <input
          type="text"
          value={dataPath}
          onChange={(e) => setDataPath(e.target.value)}
          style={inputStyle}
          placeholder="Path to JSONL dataset"
        />

        <label style={labelStyle}>Output Directory</label>
        <input
          type="text"
          value={outputDir}
          onChange={(e) => setOutputDir(e.target.value)}
          style={inputStyle}
          placeholder="Output directory for model"
        />

        <div style={{ marginBottom: '16px' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={dryRun}
              onChange={(e) => setDryRun(e.target.checked)}
            />
            Dry Run (stub training)
          </label>
        </div>

        <div style={{ marginBottom: '24px' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={useLora}
              onChange={(e) => setUseLora(e.target.checked)}
            />
            Use LoRA Adapter
          </label>
        </div>

        <button
          onClick={handleTrain}
          disabled={isTraining}
          style={{
            ...buttonStyle,
            opacity: isTraining ? 0.7 : 1
          }}
        >
          {isTraining ? 'Training...' : 'Start Training'}
        </button>
      </div>

      {status && (
        <div style={{
          ...cardStyle,
          background: status.includes('complete') ? '#e8f5e9' : '#fff3e0',
          borderLeft: `4px solid ${status.includes('complete') ? '#4caf50' : '#ff9800'}`
        }}>
          <h4 style={{ marginBottom: '8px' }}>Status</h4>
          <p style={{ fontFamily: 'monospace' }}>{status}</p>
        </div>
      )}

      <div style={cardStyle}>
        <h3 style={{ marginBottom: '16px' }}>CLI Equivalent</h3>
        <pre style={{
          background: '#1a1a2e',
          color: '#e0e0e0',
          padding: '16px',
          borderRadius: '4px',
          overflow: 'auto'
        }}>
{`foremforge train \\
  --data ${dataPath} \\
  --out ${outputDir}${dryRun ? ' \\\n  --dry-run' : ''}${useLora ? ' \\\n  --use-lora' : ''}`}
        </pre>
      </div>
    </div>
  )
}
