/**
 * TinyForgeAI API Client
 *
 * Centralized API communication layer for the dashboard.
 * Connects to the Dashboard API backend at localhost:8001.
 */

const API_BASE_URL = 'http://localhost:8001';

/**
 * Generic fetch wrapper with error handling
 */
async function apiFetch(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint}`;

  const defaultHeaders = {
    'Content-Type': 'application/json',
  };

  const config = {
    ...options,
    headers: {
      ...defaultHeaders,
      ...options.headers,
    },
  };

  try {
    const response = await fetch(url, config);

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    if (error.message === 'Failed to fetch') {
      throw new Error('Unable to connect to API. Is the server running?');
    }
    throw error;
  }
}

// ============================================
// Health & Stats
// ============================================

export async function getHealth() {
  return apiFetch('/health');
}

export async function getStats() {
  return apiFetch('/api/stats');
}

// ============================================
// Training Jobs
// ============================================

export async function getJobs(status = null, limit = 50) {
  const params = new URLSearchParams();
  if (status) params.set('status', status);
  params.set('limit', limit);
  return apiFetch(`/api/jobs?${params}`);
}

export async function getJob(jobId) {
  return apiFetch(`/api/jobs/${jobId}`);
}

export async function createJob(jobData) {
  return apiFetch('/api/jobs', {
    method: 'POST',
    body: JSON.stringify(jobData),
  });
}

export async function cancelJob(jobId) {
  return apiFetch(`/api/jobs/${jobId}/cancel`, {
    method: 'POST',
  });
}

export async function deleteJob(jobId) {
  return apiFetch(`/api/jobs/${jobId}`, {
    method: 'DELETE',
  });
}

// ============================================
// Services
// ============================================

export async function getServices(status = null) {
  const params = new URLSearchParams();
  if (status) params.set('status', status);
  return apiFetch(`/api/services?${params}`);
}

export async function getService(serviceId) {
  return apiFetch(`/api/services/${serviceId}`);
}

export async function createService(serviceData) {
  return apiFetch('/api/services', {
    method: 'POST',
    body: JSON.stringify(serviceData),
  });
}

export async function startService(serviceId) {
  return apiFetch(`/api/services/${serviceId}/start`, {
    method: 'POST',
  });
}

export async function stopService(serviceId) {
  return apiFetch(`/api/services/${serviceId}/stop`, {
    method: 'POST',
  });
}

export async function deleteService(serviceId) {
  return apiFetch(`/api/services/${serviceId}`, {
    method: 'DELETE',
  });
}

// ============================================
// Models
// ============================================

export async function getModels(modelType = null) {
  const params = new URLSearchParams();
  if (modelType) params.set('model_type', modelType);
  return apiFetch(`/api/models?${params}`);
}

export async function getModel(modelId) {
  return apiFetch(`/api/models/${modelId}`);
}

export async function registerModel(modelData) {
  return apiFetch('/api/models', {
    method: 'POST',
    body: JSON.stringify(modelData),
  });
}

export async function updateModel(modelId, modelData) {
  return apiFetch(`/api/models/${modelId}`, {
    method: 'PUT',
    body: JSON.stringify(modelData),
  });
}

export async function deleteModel(modelId) {
  return apiFetch(`/api/models/${modelId}`, {
    method: 'DELETE',
  });
}

export async function deployModel(modelId) {
  return apiFetch(`/api/models/${modelId}/deploy`, {
    method: 'POST',
  });
}

export async function undeployModel(modelId) {
  return apiFetch(`/api/models/${modelId}/undeploy`, {
    method: 'POST',
  });
}

// ============================================
// Inference
// ============================================

export async function predict(input, serviceId = null) {
  return apiFetch('/api/predict', {
    method: 'POST',
    body: JSON.stringify({ input, service_id: serviceId }),
  });
}

// ============================================
// Logs
// ============================================

export async function getLogs(level = null, source = null, limit = 100) {
  const params = new URLSearchParams();
  if (level) params.set('level', level);
  if (source) params.set('source', source);
  params.set('limit', limit);
  return apiFetch(`/api/logs?${params}`);
}

// ============================================
// WebSocket Connection
// ============================================

export function createWebSocket(channel = 'jobs') {
  const wsUrl = `ws://localhost:8001/ws/${channel}`;
  const ws = new WebSocket(wsUrl);

  // Heartbeat to keep connection alive
  let heartbeatInterval = null;

  ws.onopen = () => {
    console.log(`WebSocket connected to ${channel}`);
    heartbeatInterval = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'ping' }));
      }
    }, 30000);
  };

  ws.onclose = () => {
    console.log(`WebSocket disconnected from ${channel}`);
    if (heartbeatInterval) {
      clearInterval(heartbeatInterval);
    }
  };

  return ws;
}

// ============================================
// API Status Check
// ============================================

export async function checkApiStatus() {
  try {
    const health = await getHealth();
    return {
      connected: true,
      ...health,
    };
  } catch (error) {
    return {
      connected: false,
      error: error.message,
    };
  }
}

export default {
  getHealth,
  getStats,
  getJobs,
  getJob,
  createJob,
  cancelJob,
  deleteJob,
  getServices,
  getService,
  createService,
  startService,
  stopService,
  deleteService,
  getModels,
  getModel,
  registerModel,
  updateModel,
  deleteModel,
  deployModel,
  undeployModel,
  predict,
  getLogs,
  createWebSocket,
  checkApiStatus,
};
