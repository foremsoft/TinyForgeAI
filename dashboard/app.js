/**
 * TinyForgeAI Dashboard Application
 * Vanilla JavaScript frontend for the dashboard API
 * Includes WebSocket support for real-time updates
 */

// Configuration
const API_BASE_URL = 'http://localhost:8001';
const WS_BASE_URL = 'ws://localhost:8001';

// State
let currentPage = 'overview';
let jobs = [];
let models = [];

// WebSocket connections
let jobsWs = null;
let logsWs = null;
let wsReconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 5;

// =============================================================================
// Initialization
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    checkAPIConnection();
    refreshStats();
    initWebSocket();

    // Auto-refresh every 30 seconds (as fallback if WebSocket disconnects)
    setInterval(refreshStats, 30000);
});

// =============================================================================
// WebSocket Connection
// =============================================================================

function initWebSocket() {
    connectJobsWebSocket();
    connectLogsWebSocket();
}

function connectJobsWebSocket() {
    if (jobsWs && jobsWs.readyState === WebSocket.OPEN) {
        return;
    }

    try {
        jobsWs = new WebSocket(`${WS_BASE_URL}/ws/jobs`);

        jobsWs.onopen = () => {
            console.log('Jobs WebSocket connected');
            wsReconnectAttempts = 0;
            addActivity('Real-time updates connected', '&#128994;');
        };

        jobsWs.onmessage = (event) => {
            const message = JSON.parse(event.data);
            handleJobsMessage(message);
        };

        jobsWs.onclose = () => {
            console.log('Jobs WebSocket disconnected');
            scheduleReconnect(connectJobsWebSocket);
        };

        jobsWs.onerror = (error) => {
            console.error('Jobs WebSocket error:', error);
        };

        // Send ping every 30 seconds to keep connection alive
        setInterval(() => {
            if (jobsWs && jobsWs.readyState === WebSocket.OPEN) {
                jobsWs.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000);

    } catch (error) {
        console.error('Failed to connect Jobs WebSocket:', error);
    }
}

function connectLogsWebSocket() {
    if (logsWs && logsWs.readyState === WebSocket.OPEN) {
        return;
    }

    try {
        logsWs = new WebSocket(`${WS_BASE_URL}/ws/logs`);

        logsWs.onopen = () => {
            console.log('Logs WebSocket connected');
        };

        logsWs.onmessage = (event) => {
            const message = JSON.parse(event.data);
            handleLogsMessage(message);
        };

        logsWs.onclose = () => {
            console.log('Logs WebSocket disconnected');
            scheduleReconnect(connectLogsWebSocket);
        };

        logsWs.onerror = (error) => {
            console.error('Logs WebSocket error:', error);
        };

    } catch (error) {
        console.error('Failed to connect Logs WebSocket:', error);
    }
}

function scheduleReconnect(connectFn) {
    if (wsReconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
        wsReconnectAttempts++;
        const delay = Math.min(1000 * Math.pow(2, wsReconnectAttempts), 30000);
        console.log(`Reconnecting in ${delay}ms (attempt ${wsReconnectAttempts})`);
        setTimeout(connectFn, delay);
    } else {
        console.log('Max reconnection attempts reached');
        addActivity('Real-time updates disconnected', '&#128308;');
    }
}

function handleJobsMessage(message) {
    switch (message.type) {
        case 'initial_jobs':
            // Initial jobs list received
            jobs = message.data;
            if (currentPage === 'jobs') {
                renderJobsTable();
            }
            break;

        case 'job_update':
            // Job progress update
            updateJobInList(message.data);
            if (currentPage === 'jobs') {
                renderJobsTable();
            }
            // Update stats
            refreshStats();
            // Add activity for status changes
            if (message.data.status === 'completed') {
                addActivity(`Job completed: ${message.data.name}`, '&#9989;');
                showToast(`Training job "${message.data.name}" completed!`, 'success');
            } else if (message.data.status === 'failed') {
                addActivity(`Job failed: ${message.data.name}`, '&#10060;');
                showToast(`Training job "${message.data.name}" failed`, 'error');
            }
            break;

        case 'job_detail':
            // Detailed job info (for subscribed job)
            console.log('Job detail:', message.data);
            break;

        case 'pong':
            // Keep-alive response
            break;
    }
}

function handleLogsMessage(message) {
    if (message.type === 'log') {
        const log = message.data;
        // Add important logs to activity feed
        if (log.level === 'INFO' || log.level === 'ERROR' || log.level === 'WARN') {
            const icon = log.level === 'ERROR' ? '&#10060;' :
                         log.level === 'WARN' ? '&#9888;' : '&#128994;';
            addActivity(log.message, icon);
        }
    }
}

function updateJobInList(jobData) {
    const index = jobs.findIndex(j => j.id === jobData.id);
    if (index >= 0) {
        jobs[index] = { ...jobs[index], ...jobData };
    } else {
        jobs.unshift(jobData);
    }
}

function subscribeToJob(jobId) {
    if (jobsWs && jobsWs.readyState === WebSocket.OPEN) {
        jobsWs.send(JSON.stringify({
            type: 'subscribe_job',
            job_id: jobId
        }));
    }
}

// =============================================================================
// Navigation
// =============================================================================

function initNavigation() {
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const page = link.dataset.page;
            showPage(page);
        });
    });
}

function showPage(pageName) {
    // Update nav links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.toggle('active', link.dataset.page === pageName);
    });

    // Update pages
    document.querySelectorAll('.page').forEach(page => {
        page.classList.toggle('active', page.id === `page-${pageName}`);
    });

    currentPage = pageName;

    // Refresh data for the page
    if (pageName === 'jobs') {
        loadJobs();
    } else if (pageName === 'models') {
        loadModels();
    }
}

// =============================================================================
// API Communication
// =============================================================================

async function apiRequest(endpoint, options = {}) {
    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
            headers: {
                'Content-Type': 'application/json',
            },
            ...options,
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('API request failed:', error);
        throw error;
    }
}

async function checkAPIConnection() {
    const statusEl = document.getElementById('api-status');

    try {
        await apiRequest('/health');
        statusEl.textContent = 'API Connected';
        statusEl.className = 'status-indicator connected';
    } catch (error) {
        statusEl.textContent = 'API Disconnected';
        statusEl.className = 'status-indicator disconnected';
    }
}

// =============================================================================
// Stats & Overview
// =============================================================================

async function refreshStats() {
    try {
        const stats = await apiRequest('/api/stats');

        document.getElementById('stat-jobs').textContent = stats.total_jobs || 0;
        document.getElementById('stat-models').textContent = stats.total_models || 0;
        document.getElementById('stat-documents').textContent = stats.total_documents || 0;
        document.getElementById('stat-requests').textContent = stats.total_requests || 0;

        showToast('Stats refreshed', 'success');
    } catch (error) {
        // Use mock data if API is not available
        document.getElementById('stat-jobs').textContent = jobs.length;
        document.getElementById('stat-models').textContent = models.length;
    }
}

function addActivity(text, icon = '&#128994;') {
    const list = document.getElementById('activity-list');
    const item = document.createElement('div');
    item.className = 'activity-item';
    item.innerHTML = `
        <span class="activity-icon">${icon}</span>
        <span class="activity-text">${text}</span>
        <span class="activity-time">Just now</span>
    `;
    list.insertBefore(item, list.firstChild);

    // Keep only last 10 activities
    while (list.children.length > 10) {
        list.removeChild(list.lastChild);
    }
}

// =============================================================================
// Training Jobs
// =============================================================================

async function loadJobs() {
    try {
        const response = await apiRequest('/api/jobs');
        jobs = response || [];
        renderJobsTable();
    } catch (error) {
        // If WebSocket already populated jobs, use those
        if (jobs.length > 0) {
            renderJobsTable();
        } else {
            const tbody = document.getElementById('jobs-table-body');
            tbody.innerHTML = `
                <tr>
                    <td colspan="6" class="empty-state">Could not load jobs. API may be disconnected.</td>
                </tr>
            `;
        }
    }
}

function renderJobsTable() {
    const tbody = document.getElementById('jobs-table-body');

    if (jobs.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="6" class="empty-state">No training jobs yet. Create one to get started!</td>
            </tr>
        `;
        return;
    }

    tbody.innerHTML = jobs.map(job => `
        <tr data-job-id="${job.id}">
            <td>${job.id}</td>
            <td>${job.model_name || job.model || 'N/A'}</td>
            <td><span class="status-badge ${job.status}">${job.status}</span></td>
            <td>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${job.progress || 0}%"></div>
                </div>
                <span class="progress-text">${(job.progress || 0).toFixed(1)}%</span>
            </td>
            <td>${formatDate(job.created_at)}</td>
            <td>
                <button class="action-btn" onclick="viewJob('${job.id}')">View</button>
                ${job.status === 'pending' || job.status === 'running' ?
                    `<button class="action-btn" onclick="cancelJob('${job.id}')">Cancel</button>` : ''}
            </td>
        </tr>
    `).join('');
}

async function cancelJob(jobId) {
    try {
        await apiRequest(`/api/jobs/${jobId}/cancel`, { method: 'POST' });
        showToast('Job cancelled', 'success');
        loadJobs();
    } catch (error) {
        showToast('Failed to cancel job', 'error');
    }
}

function showNewJobModal() {
    document.getElementById('new-job-modal').classList.add('active');
}

function closeModal() {
    document.getElementById('new-job-modal').classList.remove('active');
}

async function createJob(event) {
    event.preventDefault();

    const jobData = {
        model: document.getElementById('job-model').value,
        data_path: document.getElementById('job-data').value,
        epochs: parseInt(document.getElementById('job-epochs').value),
        batch_size: parseInt(document.getElementById('job-batch-size').value),
        use_lora: document.getElementById('job-use-lora').checked,
    };

    try {
        const response = await apiRequest('/api/jobs', {
            method: 'POST',
            body: JSON.stringify(jobData),
        });

        showToast('Training job created successfully!', 'success');
        addActivity(`Created training job: ${jobData.model}`, '&#128640;');
        closeModal();
        loadJobs();
        refreshStats();
    } catch (error) {
        // Mock job creation
        const mockJob = {
            id: `job-${Date.now()}`,
            ...jobData,
            status: 'pending',
            progress: 0,
            created_at: new Date().toISOString(),
        };
        jobs.push(mockJob);

        showToast('Job created (mock mode)', 'warning');
        addActivity(`Created training job: ${jobData.model}`, '&#128640;');
        closeModal();
        document.getElementById('stat-jobs').textContent = jobs.length;
    }
}

function viewJob(jobId) {
    showToast(`Viewing job ${jobId}`, 'success');
}

// =============================================================================
// Models
// =============================================================================

async function loadModels() {
    const grid = document.getElementById('models-grid');

    try {
        const response = await apiRequest('/api/models');
        models = response.models || [];

        if (models.length === 0) {
            grid.innerHTML = `
                <div class="empty-state-card">
                    <span class="empty-icon">&#129302;</span>
                    <p>No trained models yet.</p>
                    <p>Complete a training job to see your models here.</p>
                </div>
            `;
            return;
        }

        grid.innerHTML = models.map(model => `
            <div class="model-card">
                <h4>${model.name}</h4>
                <p>${model.description || 'No description'}</p>
                <div class="model-meta">
                    <span>Size: ${formatSize(model.size)}</span>
                    <span>Created: ${formatDate(model.created_at)}</span>
                </div>
                <div style="margin-top: 16px;">
                    <button class="action-btn" onclick="downloadModel('${model.id}')">Download</button>
                    <button class="action-btn primary" onclick="deployModel('${model.id}')">Deploy</button>
                </div>
            </div>
        `).join('');
    } catch (error) {
        // Keep empty state
    }
}

function downloadModel(modelId) {
    showToast(`Downloading model ${modelId}...`, 'success');
}

function deployModel(modelId) {
    showToast(`Deploying model ${modelId}...`, 'success');
}

// =============================================================================
// RAG Search
// =============================================================================

async function searchDocuments() {
    const query = document.getElementById('rag-search-input').value.trim();
    const resultsContainer = document.getElementById('search-results');

    if (!query) {
        showToast('Please enter a search query', 'warning');
        return;
    }

    try {
        const response = await apiRequest(`/api/search?q=${encodeURIComponent(query)}&top_k=5`);
        const results = response.results || [];

        if (results.length === 0) {
            resultsContainer.innerHTML = `
                <div class="empty-state-card">
                    <span class="empty-icon">&#128533;</span>
                    <p>No results found for "${query}"</p>
                </div>
            `;
            return;
        }

        resultsContainer.innerHTML = results.map(result => `
            <div class="search-result-item">
                <span class="search-result-score">Score: ${result.score.toFixed(3)}</span>
                <p class="search-result-content">${result.content}</p>
                <p class="search-result-source">Source: ${result.source || 'Unknown'}</p>
            </div>
        `).join('');

        addActivity(`Searched: "${query}"`, '&#128270;');
    } catch (error) {
        // Mock search results
        resultsContainer.innerHTML = `
            <div class="empty-state-card">
                <span class="empty-icon">&#128533;</span>
                <p>Search is not available. RAG index may not be configured.</p>
                <p>Run: python examples/rag/quick_start_rag.py</p>
            </div>
        `;
    }
}

// Allow Enter key to search
document.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && e.target.id === 'rag-search-input') {
        searchDocuments();
    }
});

// =============================================================================
// Utilities
// =============================================================================

function formatDate(dateString) {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
}

function formatSize(bytes) {
    if (!bytes) return 'N/A';
    const units = ['B', 'KB', 'MB', 'GB'];
    let unitIndex = 0;
    let size = bytes;

    while (size >= 1024 && unitIndex < units.length - 1) {
        size /= 1024;
        unitIndex++;
    }

    return `${size.toFixed(1)} ${units[unitIndex]}`;
}

// =============================================================================
// Toast Notifications
// =============================================================================

function showToast(message, type = 'success') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <span>${getToastIcon(type)}</span>
        <span>${message}</span>
    `;

    container.appendChild(toast);

    // Auto-remove after 3 seconds
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

function getToastIcon(type) {
    switch (type) {
        case 'success': return '&#10004;';
        case 'error': return '&#10006;';
        case 'warning': return '&#9888;';
        default: return '&#8505;';
    }
}

// =============================================================================
// Export for global access
// =============================================================================

window.showPage = showPage;
window.showNewJobModal = showNewJobModal;
window.closeModal = closeModal;
window.createJob = createJob;
window.viewJob = viewJob;
window.cancelJob = cancelJob;
window.refreshStats = refreshStats;
window.searchDocuments = searchDocuments;
window.downloadModel = downloadModel;
window.deployModel = deployModel;
window.subscribeToJob = subscribeToJob;
