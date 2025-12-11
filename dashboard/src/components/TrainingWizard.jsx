/**
 * TrainingWizard Component
 *
 * A beginner-friendly step-by-step wizard for training AI models.
 * Designed for non-technical users with basic computer knowledge.
 */
import React, { useState, useCallback } from 'react'
import { createJob, uploadTrainingData } from '../api/client'

// Styles
const wizardContainer = {
  background: 'white',
  borderRadius: '12px',
  boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
  overflow: 'hidden'
}

const stepIndicator = {
  display: 'flex',
  background: '#f8f9fa',
  borderBottom: '1px solid #eee',
  padding: '0'
}

const stepTab = (active, completed) => ({
  flex: 1,
  padding: '16px 12px',
  textAlign: 'center',
  cursor: 'pointer',
  background: active ? 'white' : 'transparent',
  borderBottom: active ? '3px solid #4a4a6a' : '3px solid transparent',
  color: active ? '#4a4a6a' : completed ? '#4caf50' : '#999',
  fontWeight: active ? '600' : '400',
  transition: 'all 0.2s',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  gap: '8px'
})

const stepNumber = (active, completed) => ({
  width: '28px',
  height: '28px',
  borderRadius: '50%',
  background: completed ? '#4caf50' : active ? '#4a4a6a' : '#ddd',
  color: 'white',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  fontSize: '14px',
  fontWeight: 'bold'
})

const contentArea = {
  padding: '32px',
  minHeight: '400px'
}

const dropZone = (isDragging) => ({
  border: `2px dashed ${isDragging ? '#4a4a6a' : '#ddd'}`,
  borderRadius: '12px',
  padding: '48px',
  textAlign: 'center',
  background: isDragging ? '#f0f0ff' : '#fafafa',
  cursor: 'pointer',
  transition: 'all 0.2s'
})

const buttonStyle = {
  background: '#4a4a6a',
  color: 'white',
  border: 'none',
  padding: '14px 32px',
  borderRadius: '8px',
  cursor: 'pointer',
  fontSize: '16px',
  fontWeight: '500',
  transition: 'all 0.2s'
}

const secondaryButton = {
  ...buttonStyle,
  background: '#f5f5f5',
  color: '#555',
  border: '1px solid #ddd'
}

const modelCard = (selected) => ({
  border: `2px solid ${selected ? '#4a4a6a' : '#eee'}`,
  borderRadius: '12px',
  padding: '20px',
  cursor: 'pointer',
  background: selected ? '#f8f8ff' : 'white',
  transition: 'all 0.2s'
})

const presetCard = (selected) => ({
  border: `2px solid ${selected ? '#4caf50' : '#eee'}`,
  borderRadius: '12px',
  padding: '16px',
  cursor: 'pointer',
  background: selected ? '#f0fff0' : 'white',
  transition: 'all 0.2s',
  textAlign: 'center'
})

const helpText = {
  background: '#e3f2fd',
  border: '1px solid #90caf9',
  borderRadius: '8px',
  padding: '16px',
  marginBottom: '24px',
  fontSize: '14px',
  color: '#1565c0',
  display: 'flex',
  alignItems: 'flex-start',
  gap: '12px'
}

const progressBar = {
  width: '100%',
  height: '8px',
  background: '#eee',
  borderRadius: '4px',
  overflow: 'hidden',
  marginTop: '16px'
}

const progressFill = (progress) => ({
  width: `${progress}%`,
  height: '100%',
  background: 'linear-gradient(90deg, #4a4a6a, #6a6a8a)',
  transition: 'width 0.3s'
})

// Model options with descriptions
const MODEL_OPTIONS = [
  {
    id: 'distilbert-base-uncased',
    name: 'DistilBERT',
    description: 'Fast and efficient. Best for beginners.',
    size: '66M parameters',
    memory: '~2GB',
    speed: 'Fast',
    recommended: true
  },
  {
    id: 'bert-base-uncased',
    name: 'BERT Base',
    description: 'Industry standard. Good balance of quality and speed.',
    size: '110M parameters',
    memory: '~4GB',
    speed: 'Medium',
    recommended: false
  },
  {
    id: 'roberta-base',
    name: 'RoBERTa',
    description: 'Better accuracy. Use if quality matters most.',
    size: '125M parameters',
    memory: '~4GB',
    speed: 'Medium',
    recommended: false
  },
  {
    id: 'google/flan-t5-small',
    name: 'Flan-T5 Small',
    description: 'Text generation and Q&A. Very versatile.',
    size: '77M parameters',
    memory: '~2GB',
    speed: 'Fast',
    recommended: false
  }
]

// Training presets
const TRAINING_PRESETS = [
  {
    id: 'quick',
    name: 'Quick Test',
    icon: '⚡',
    description: 'Test in ~5 minutes',
    epochs: 1,
    batchSize: 8,
    learningRate: 0.0001
  },
  {
    id: 'balanced',
    name: 'Balanced',
    icon: '⚖️',
    description: 'Good results in ~15 min',
    epochs: 3,
    batchSize: 4,
    learningRate: 0.00005
  },
  {
    id: 'thorough',
    name: 'Thorough',
    icon: '🎯',
    description: 'Best results, ~30 min',
    epochs: 5,
    batchSize: 4,
    learningRate: 0.00003
  }
]

export default function TrainingWizard({ onComplete, onCancel }) {
  // Wizard state
  const [currentStep, setCurrentStep] = useState(0)
  const [completedSteps, setCompletedSteps] = useState(new Set())

  // Step 1: Data
  const [uploadedFile, setUploadedFile] = useState(null)
  const [dataPreview, setDataPreview] = useState(null)
  const [isDragging, setIsDragging] = useState(false)
  const [uploadError, setUploadError] = useState(null)
  const [uploadProgress, setUploadProgress] = useState(0)

  // Step 2: Model
  const [selectedModel, setSelectedModel] = useState('distilbert-base-uncased')

  // Step 3: Settings
  const [selectedPreset, setSelectedPreset] = useState('balanced')
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [epochs, setEpochs] = useState(3)
  const [batchSize, setBatchSize] = useState(4)
  const [learningRate, setLearningRate] = useState(0.00005)
  const [useLora, setUseLora] = useState(true)
  const [modelName, setModelName] = useState('')

  // Step 4: Training
  const [isTraining, setIsTraining] = useState(false)
  const [trainingStatus, setTrainingStatus] = useState('')
  const [trainingProgress, setTrainingProgress] = useState(0)
  const [trainingError, setTrainingError] = useState(null)
  const [jobId, setJobId] = useState(null)

  const steps = [
    { name: 'Upload Data', icon: '📁' },
    { name: 'Choose Model', icon: '🤖' },
    { name: 'Settings', icon: '⚙️' },
    { name: 'Train', icon: '🚀' }
  ]

  // Handle file drop
  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setIsDragging(false)

    const file = e.dataTransfer?.files?.[0] || e.target?.files?.[0]
    if (file) {
      processFile(file)
    }
  }, [])

  const processFile = async (file) => {
    setUploadError(null)
    setUploadProgress(0)

    // Validate file type
    const validTypes = ['.csv', '.jsonl', '.json']
    const fileExt = '.' + file.name.split('.').pop().toLowerCase()

    if (!validTypes.includes(fileExt)) {
      setUploadError('Please upload a CSV or JSONL file')
      return
    }

    setUploadedFile(file)
    setUploadProgress(30)

    // Read file preview
    try {
      const text = await file.text()
      let preview = { rows: 0, sample: [] }

      if (fileExt === '.csv') {
        const lines = text.trim().split('\n')
        preview.rows = lines.length - 1 // Exclude header
        preview.headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''))
        preview.sample = lines.slice(1, 4).map(line => {
          const values = line.split(',').map(v => v.trim().replace(/"/g, ''))
          return preview.headers.reduce((obj, header, i) => {
            obj[header] = values[i]
            return obj
          }, {})
        })
      } else {
        const lines = text.trim().split('\n')
        preview.rows = lines.length
        preview.sample = lines.slice(0, 3).map(line => JSON.parse(line))
      }

      setDataPreview(preview)
      setUploadProgress(100)

      // Auto-generate model name from file
      if (!modelName) {
        const baseName = file.name.replace(/\.[^/.]+$/, '').replace(/[^a-zA-Z0-9]/g, '_')
        setModelName(`${baseName}_model`)
      }
    } catch (err) {
      setUploadError('Could not read file. Please check the format.')
      setUploadedFile(null)
    }
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handlePresetChange = (preset) => {
    setSelectedPreset(preset.id)
    setEpochs(preset.epochs)
    setBatchSize(preset.batchSize)
    setLearningRate(preset.learningRate)
  }

  const goToStep = (step) => {
    if (step < currentStep || canProceed()) {
      setCurrentStep(step)
    }
  }

  const canProceed = () => {
    switch (currentStep) {
      case 0: return uploadedFile && dataPreview
      case 1: return selectedModel
      case 2: return modelName.trim().length > 0
      case 3: return true
      default: return false
    }
  }

  const nextStep = () => {
    if (canProceed()) {
      setCompletedSteps(prev => new Set([...prev, currentStep]))
      setCurrentStep(prev => Math.min(prev + 1, steps.length - 1))
    }
  }

  const prevStep = () => {
    setCurrentStep(prev => Math.max(prev - 1, 0))
  }

  const startTraining = async () => {
    setIsTraining(true)
    setTrainingError(null)
    setTrainingStatus('Preparing training job...')
    setTrainingProgress(5)

    try {
      // Create job
      const jobData = {
        name: modelName,
        config: {
          dataset_path: uploadedFile.name, // Will be uploaded separately
          output_dir: `./models/${modelName}`,
          base_model: selectedModel,
          epochs: epochs,
          batch_size: batchSize,
          learning_rate: learningRate,
          use_lora: useLora,
          lora_rank: useLora ? 8 : null
        }
      }

      setTrainingStatus('Submitting training job...')
      setTrainingProgress(15)

      const result = await createJob(jobData)
      setJobId(result.id)
      setTrainingStatus(`Training started! Job ID: ${result.id}`)
      setTrainingProgress(25)

      // Simulate progress updates (in production, this would come from WebSocket)
      simulateProgress()
    } catch (err) {
      setTrainingError(err.message)
      setIsTraining(false)
    }
  }

  const simulateProgress = () => {
    // This is a demo - in production, progress would come from WebSocket
    const stages = [
      { progress: 30, status: 'Loading model...' },
      { progress: 40, status: 'Processing training data...' },
      { progress: 50, status: 'Training epoch 1...' },
      { progress: 65, status: 'Training epoch 2...' },
      { progress: 80, status: 'Training epoch 3...' },
      { progress: 90, status: 'Saving model...' },
      { progress: 100, status: 'Training complete!' }
    ]

    let i = 0
    const interval = setInterval(() => {
      if (i < stages.length) {
        setTrainingProgress(stages[i].progress)
        setTrainingStatus(stages[i].status)
        i++
      } else {
        clearInterval(interval)
        setIsTraining(false)
      }
    }, 2000)
  }

  // Render steps
  const renderStep = () => {
    switch (currentStep) {
      case 0:
        return renderDataStep()
      case 1:
        return renderModelStep()
      case 2:
        return renderSettingsStep()
      case 3:
        return renderTrainStep()
      default:
        return null
    }
  }

  const renderDataStep = () => (
    <div>
      <h2 style={{ marginBottom: '8px' }}>Upload Your Training Data</h2>
      <p style={{ color: '#666', marginBottom: '24px' }}>
        Upload a file with questions and answers (or inputs and outputs) to train your AI model.
      </p>

      <div style={helpText}>
        <span style={{ fontSize: '20px' }}>💡</span>
        <div>
          <strong>Supported formats:</strong>
          <ul style={{ margin: '8px 0 0 0', paddingLeft: '20px' }}>
            <li><strong>CSV:</strong> With columns like "question" and "answer" or "input" and "output"</li>
            <li><strong>JSONL:</strong> One JSON object per line with "input" and "output" fields</li>
          </ul>
        </div>
      </div>

      <div
        style={dropZone(isDragging)}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={() => document.getElementById('file-input').click()}
      >
        <input
          id="file-input"
          type="file"
          accept=".csv,.jsonl,.json"
          onChange={(e) => processFile(e.target.files[0])}
          style={{ display: 'none' }}
        />

        {!uploadedFile ? (
          <>
            <div style={{ fontSize: '48px', marginBottom: '16px' }}>📄</div>
            <p style={{ fontSize: '18px', fontWeight: '500', marginBottom: '8px' }}>
              Drag and drop your file here
            </p>
            <p style={{ color: '#666' }}>or click to browse</p>
            <p style={{ color: '#999', fontSize: '13px', marginTop: '16px' }}>
              Accepted: .csv, .jsonl
            </p>
          </>
        ) : (
          <>
            <div style={{ fontSize: '48px', marginBottom: '16px' }}>✅</div>
            <p style={{ fontSize: '18px', fontWeight: '500', marginBottom: '8px' }}>
              {uploadedFile.name}
            </p>
            <p style={{ color: '#666' }}>
              {dataPreview?.rows} training examples found
            </p>
            <button
              onClick={(e) => {
                e.stopPropagation()
                setUploadedFile(null)
                setDataPreview(null)
              }}
              style={{ ...secondaryButton, marginTop: '16px', padding: '8px 16px' }}
            >
              Choose Different File
            </button>
          </>
        )}
      </div>

      {uploadError && (
        <div style={{ color: '#c62828', marginTop: '16px', padding: '12px', background: '#ffebee', borderRadius: '8px' }}>
          ❌ {uploadError}
        </div>
      )}

      {dataPreview && dataPreview.sample.length > 0 && (
        <div style={{ marginTop: '24px' }}>
          <h4 style={{ marginBottom: '12px' }}>Data Preview (first 3 rows):</h4>
          <div style={{ background: '#f5f5f5', borderRadius: '8px', padding: '16px', overflow: 'auto' }}>
            {dataPreview.sample.map((row, i) => (
              <div key={i} style={{ marginBottom: i < dataPreview.sample.length - 1 ? '12px' : 0, paddingBottom: '12px', borderBottom: i < dataPreview.sample.length - 1 ? '1px solid #ddd' : 'none' }}>
                {Object.entries(row).map(([key, value]) => (
                  <div key={key} style={{ marginBottom: '4px' }}>
                    <strong style={{ color: '#4a4a6a' }}>{key}:</strong>{' '}
                    <span style={{ color: '#555' }}>{String(value).slice(0, 100)}{String(value).length > 100 ? '...' : ''}</span>
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )

  const renderModelStep = () => (
    <div>
      <h2 style={{ marginBottom: '8px' }}>Choose Your AI Model</h2>
      <p style={{ color: '#666', marginBottom: '24px' }}>
        Select a base model to train. We'll fine-tune it on your data.
      </p>

      <div style={helpText}>
        <span style={{ fontSize: '20px' }}>💡</span>
        <div>
          <strong>Not sure which to pick?</strong> Start with <strong>DistilBERT</strong> - it's fast, efficient, and great for most use cases. You can always train again with a different model.
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '16px' }}>
        {MODEL_OPTIONS.map(model => (
          <div
            key={model.id}
            style={modelCard(selectedModel === model.id)}
            onClick={() => setSelectedModel(model.id)}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '8px' }}>
              <h4 style={{ margin: 0 }}>
                {model.name}
                {model.recommended && (
                  <span style={{ marginLeft: '8px', background: '#4caf50', color: 'white', padding: '2px 8px', borderRadius: '4px', fontSize: '11px' }}>
                    Recommended
                  </span>
                )}
              </h4>
              {selectedModel === model.id && <span style={{ color: '#4a4a6a', fontSize: '20px' }}>✓</span>}
            </div>
            <p style={{ color: '#666', marginBottom: '12px', fontSize: '14px' }}>{model.description}</p>
            <div style={{ display: 'flex', gap: '16px', fontSize: '12px', color: '#888' }}>
              <span>📊 {model.size}</span>
              <span>💾 {model.memory}</span>
              <span>⚡ {model.speed}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )

  const renderSettingsStep = () => (
    <div>
      <h2 style={{ marginBottom: '8px' }}>Training Settings</h2>
      <p style={{ color: '#666', marginBottom: '24px' }}>
        Choose how thoroughly to train your model.
      </p>

      <div style={{ marginBottom: '32px' }}>
        <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500' }}>Model Name</label>
        <input
          type="text"
          value={modelName}
          onChange={(e) => setModelName(e.target.value)}
          placeholder="my_faq_bot"
          style={{
            width: '100%',
            maxWidth: '400px',
            padding: '12px 16px',
            border: '1px solid #ddd',
            borderRadius: '8px',
            fontSize: '16px'
          }}
        />
        <p style={{ color: '#888', fontSize: '13px', marginTop: '4px' }}>
          A name to identify your trained model
        </p>
      </div>

      <h3 style={{ marginBottom: '16px' }}>Training Intensity</h3>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px', marginBottom: '24px' }}>
        {TRAINING_PRESETS.map(preset => (
          <div
            key={preset.id}
            style={presetCard(selectedPreset === preset.id)}
            onClick={() => handlePresetChange(preset)}
          >
            <div style={{ fontSize: '32px', marginBottom: '8px' }}>{preset.icon}</div>
            <h4 style={{ margin: '0 0 4px 0' }}>{preset.name}</h4>
            <p style={{ color: '#666', fontSize: '13px', margin: 0 }}>{preset.description}</p>
            {selectedPreset === preset.id && (
              <div style={{ marginTop: '8px', color: '#4caf50' }}>✓ Selected</div>
            )}
          </div>
        ))}
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
        <input
          type="checkbox"
          id="use-lora"
          checked={useLora}
          onChange={(e) => setUseLora(e.target.checked)}
        />
        <label htmlFor="use-lora" style={{ cursor: 'pointer' }}>
          Use LoRA (faster training, less memory)
        </label>
        <span style={{ color: '#4caf50', fontSize: '12px', background: '#e8f5e9', padding: '2px 8px', borderRadius: '4px' }}>
          Recommended
        </span>
      </div>

      <button
        onClick={() => setShowAdvanced(!showAdvanced)}
        style={{ ...secondaryButton, padding: '8px 16px', fontSize: '14px' }}
      >
        {showAdvanced ? '▼ Hide' : '▶ Show'} Advanced Settings
      </button>

      {showAdvanced && (
        <div style={{ marginTop: '24px', padding: '20px', background: '#f8f9fa', borderRadius: '8px' }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
            <div>
              <label style={{ display: 'block', marginBottom: '4px', fontSize: '14px', fontWeight: '500' }}>Epochs</label>
              <input
                type="number"
                value={epochs}
                onChange={(e) => setEpochs(parseInt(e.target.value) || 1)}
                min="1"
                max="20"
                style={{ width: '100%', padding: '8px', border: '1px solid #ddd', borderRadius: '4px' }}
              />
              <p style={{ color: '#888', fontSize: '12px', marginTop: '4px' }}>How many times to go through data</p>
            </div>
            <div>
              <label style={{ display: 'block', marginBottom: '4px', fontSize: '14px', fontWeight: '500' }}>Batch Size</label>
              <input
                type="number"
                value={batchSize}
                onChange={(e) => setBatchSize(parseInt(e.target.value) || 1)}
                min="1"
                max="32"
                style={{ width: '100%', padding: '8px', border: '1px solid #ddd', borderRadius: '4px' }}
              />
              <p style={{ color: '#888', fontSize: '12px', marginTop: '4px' }}>Examples per training step</p>
            </div>
            <div>
              <label style={{ display: 'block', marginBottom: '4px', fontSize: '14px', fontWeight: '500' }}>Learning Rate</label>
              <input
                type="number"
                value={learningRate}
                onChange={(e) => setLearningRate(parseFloat(e.target.value) || 0.00001)}
                step="0.00001"
                style={{ width: '100%', padding: '8px', border: '1px solid #ddd', borderRadius: '4px' }}
              />
              <p style={{ color: '#888', fontSize: '12px', marginTop: '4px' }}>How fast the model learns</p>
            </div>
          </div>
        </div>
      )}

      <div style={{ marginTop: '32px', padding: '20px', background: '#f0f7ff', borderRadius: '8px', border: '1px solid #90caf9' }}>
        <h4 style={{ marginBottom: '12px', color: '#1565c0' }}>📋 Training Summary</h4>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '8px', fontSize: '14px' }}>
          <div><strong>Data:</strong> {uploadedFile?.name}</div>
          <div><strong>Examples:</strong> {dataPreview?.rows}</div>
          <div><strong>Model:</strong> {MODEL_OPTIONS.find(m => m.id === selectedModel)?.name}</div>
          <div><strong>Epochs:</strong> {epochs}</div>
          <div><strong>LoRA:</strong> {useLora ? 'Yes (efficient)' : 'No'}</div>
          <div><strong>Output:</strong> ./models/{modelName}</div>
        </div>
      </div>
    </div>
  )

  const renderTrainStep = () => (
    <div>
      <h2 style={{ marginBottom: '8px' }}>Train Your Model</h2>
      <p style={{ color: '#666', marginBottom: '24px' }}>
        {!isTraining && trainingProgress < 100
          ? "Everything is set up! Click the button below to start training."
          : trainingProgress === 100
          ? "Training complete! Your model is ready to use."
          : "Training in progress..."}
      </p>

      {!isTraining && trainingProgress === 0 && (
        <div style={{ textAlign: 'center', padding: '40px' }}>
          <div style={{ fontSize: '64px', marginBottom: '24px' }}>🚀</div>
          <h3 style={{ marginBottom: '16px' }}>Ready to Train!</h3>
          <p style={{ color: '#666', marginBottom: '32px', maxWidth: '400px', margin: '0 auto 32px' }}>
            Your AI model will learn from {dataPreview?.rows} examples using {MODEL_OPTIONS.find(m => m.id === selectedModel)?.name}.
          </p>
          <button
            onClick={startTraining}
            style={{ ...buttonStyle, padding: '16px 48px', fontSize: '18px' }}
          >
            🚀 Start Training
          </button>
        </div>
      )}

      {(isTraining || trainingProgress > 0) && (
        <div style={{ padding: '24px' }}>
          <div style={{ marginBottom: '24px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <span style={{ fontWeight: '500' }}>{trainingStatus}</span>
              <span style={{ color: '#4a4a6a', fontWeight: 'bold' }}>{trainingProgress}%</span>
            </div>
            <div style={progressBar}>
              <div style={progressFill(trainingProgress)} />
            </div>
          </div>

          {trainingProgress === 100 && (
            <div style={{ textAlign: 'center', padding: '32px', background: '#e8f5e9', borderRadius: '12px' }}>
              <div style={{ fontSize: '64px', marginBottom: '16px' }}>🎉</div>
              <h3 style={{ color: '#2e7d32', marginBottom: '8px' }}>Training Complete!</h3>
              <p style={{ color: '#555', marginBottom: '24px' }}>
                Your model "{modelName}" has been trained and saved.
              </p>
              <div style={{ display: 'flex', gap: '16px', justifyContent: 'center' }}>
                <button
                  onClick={() => onComplete && onComplete({ jobId, modelName })}
                  style={buttonStyle}
                >
                  View in Models
                </button>
                <button
                  onClick={() => {
                    // Reset wizard for another training
                    setCurrentStep(0)
                    setCompletedSteps(new Set())
                    setUploadedFile(null)
                    setDataPreview(null)
                    setTrainingProgress(0)
                    setJobId(null)
                  }}
                  style={secondaryButton}
                >
                  Train Another Model
                </button>
              </div>
            </div>
          )}

          {trainingError && (
            <div style={{ color: '#c62828', padding: '16px', background: '#ffebee', borderRadius: '8px', marginTop: '16px' }}>
              <strong>Error:</strong> {trainingError}
              <button
                onClick={startTraining}
                style={{ ...secondaryButton, marginLeft: '16px', padding: '8px 16px' }}
              >
                Retry
              </button>
            </div>
          )}

          {isTraining && (
            <div style={helpText}>
              <span style={{ fontSize: '20px' }}>⏳</span>
              <div>
                <strong>Training in progress...</strong>
                <p style={{ margin: '8px 0 0 0' }}>
                  This may take a few minutes depending on your data size and settings.
                  You can leave this page - the training will continue in the background.
                </p>
              </div>
            </div>
          )}
        </div>
      )}

      {jobId && (
        <div style={{ marginTop: '24px', padding: '16px', background: '#f5f5f5', borderRadius: '8px', fontFamily: 'monospace', fontSize: '13px' }}>
          <strong>Job ID:</strong> {jobId}
        </div>
      )}
    </div>
  )

  return (
    <div style={wizardContainer}>
      {/* Step Indicator */}
      <div style={stepIndicator}>
        {steps.map((step, index) => (
          <div
            key={index}
            style={stepTab(currentStep === index, completedSteps.has(index))}
            onClick={() => goToStep(index)}
          >
            <span style={stepNumber(currentStep === index, completedSteps.has(index))}>
              {completedSteps.has(index) ? '✓' : index + 1}
            </span>
            <span>{step.icon} {step.name}</span>
          </div>
        ))}
      </div>

      {/* Content */}
      <div style={contentArea}>
        {renderStep()}
      </div>

      {/* Navigation */}
      {currentStep < 3 && (
        <div style={{ padding: '20px 32px', borderTop: '1px solid #eee', display: 'flex', justifyContent: 'space-between' }}>
          <button
            onClick={currentStep === 0 ? onCancel : prevStep}
            style={secondaryButton}
          >
            {currentStep === 0 ? 'Cancel' : '← Back'}
          </button>
          <button
            onClick={nextStep}
            disabled={!canProceed()}
            style={{
              ...buttonStyle,
              opacity: canProceed() ? 1 : 0.5,
              cursor: canProceed() ? 'pointer' : 'not-allowed'
            }}
          >
            {currentStep === 2 ? 'Review & Train →' : 'Continue →'}
          </button>
        </div>
      )}
    </div>
  )
}
