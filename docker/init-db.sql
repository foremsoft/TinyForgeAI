-- TinyForgeAI Database Initialization Script
-- This script runs when the PostgreSQL container starts for the first time

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ==========================================================================
-- Models Table
-- ==========================================================================
CREATE TABLE IF NOT EXISTS models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL DEFAULT 'v1',
    description TEXT,
    model_type VARCHAR(100),
    framework VARCHAR(50) DEFAULT 'pytorch',
    file_path TEXT,
    file_size_bytes BIGINT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name, version)
);

CREATE INDEX idx_models_name ON models(name);
CREATE INDEX idx_models_created_at ON models(created_at DESC);

-- ==========================================================================
-- Training Jobs Table
-- ==========================================================================
CREATE TABLE IF NOT EXISTS training_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    config JSONB DEFAULT '{}',
    progress REAL DEFAULT 0,
    metrics JSONB DEFAULT '{}',
    error_message TEXT,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_training_jobs_status ON training_jobs(status);
CREATE INDEX idx_training_jobs_model_name ON training_jobs(model_name);
CREATE INDEX idx_training_jobs_created_at ON training_jobs(created_at DESC);

-- ==========================================================================
-- API Keys Table
-- ==========================================================================
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    key_hash VARCHAR(255) NOT NULL,
    key_prefix VARCHAR(10) NOT NULL,
    scopes TEXT[] DEFAULT ARRAY['read'],
    rate_limit INTEGER DEFAULT 1000,
    is_active BOOLEAN DEFAULT true,
    last_used_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_api_keys_key_prefix ON api_keys(key_prefix);
CREATE INDEX idx_api_keys_is_active ON api_keys(is_active);

-- ==========================================================================
-- Inference Requests Table (for logging/analytics)
-- ==========================================================================
CREATE TABLE IF NOT EXISTS inference_requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(255) NOT NULL,
    model_version VARCHAR(50),
    api_key_id UUID REFERENCES api_keys(id),
    input_size_bytes INTEGER,
    output_size_bytes INTEGER,
    latency_ms REAL,
    status VARCHAR(50) NOT NULL,
    error_message TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_inference_requests_model ON inference_requests(model_name);
CREATE INDEX idx_inference_requests_created_at ON inference_requests(created_at DESC);
CREATE INDEX idx_inference_requests_api_key ON inference_requests(api_key_id);

-- ==========================================================================
-- Webhook Endpoints Table
-- ==========================================================================
CREATE TABLE IF NOT EXISTS webhook_endpoints (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    url TEXT NOT NULL,
    secret VARCHAR(255),
    events TEXT[] DEFAULT ARRAY['*'],
    headers JSONB DEFAULT '{}',
    is_enabled BOOLEAN DEFAULT true,
    max_retries INTEGER DEFAULT 3,
    retry_delay_seconds REAL DEFAULT 1.0,
    timeout_seconds INTEGER DEFAULT 30,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_webhook_endpoints_enabled ON webhook_endpoints(is_enabled);

-- ==========================================================================
-- Webhook Deliveries Table
-- ==========================================================================
CREATE TABLE IF NOT EXISTS webhook_deliveries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    endpoint_id UUID REFERENCES webhook_endpoints(id) ON DELETE CASCADE,
    event_type VARCHAR(100) NOT NULL,
    event_data JSONB NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    attempts INTEGER DEFAULT 0,
    response_code INTEGER,
    response_body TEXT,
    error_message TEXT,
    next_retry_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_webhook_deliveries_endpoint ON webhook_deliveries(endpoint_id);
CREATE INDEX idx_webhook_deliveries_status ON webhook_deliveries(status);
CREATE INDEX idx_webhook_deliveries_created_at ON webhook_deliveries(created_at DESC);

-- ==========================================================================
-- Model Deployments Table
-- ==========================================================================
CREATE TABLE IF NOT EXISTS model_deployments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID REFERENCES models(id) ON DELETE CASCADE,
    endpoint_name VARCHAR(255) NOT NULL UNIQUE,
    is_active BOOLEAN DEFAULT true,
    replicas INTEGER DEFAULT 1,
    min_replicas INTEGER DEFAULT 1,
    max_replicas INTEGER DEFAULT 10,
    config JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_model_deployments_active ON model_deployments(is_active);
CREATE INDEX idx_model_deployments_endpoint ON model_deployments(endpoint_name);

-- ==========================================================================
-- A/B Test Experiments Table
-- ==========================================================================
CREATE TABLE IF NOT EXISTS ab_experiments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    status VARCHAR(50) NOT NULL DEFAULT 'draft',
    model_a_id UUID REFERENCES models(id),
    model_b_id UUID REFERENCES models(id),
    traffic_split_a REAL DEFAULT 0.5,
    metrics_config JSONB DEFAULT '{}',
    started_at TIMESTAMP WITH TIME ZONE,
    ended_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_ab_experiments_status ON ab_experiments(status);

-- ==========================================================================
-- A/B Test Results Table
-- ==========================================================================
CREATE TABLE IF NOT EXISTS ab_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_id UUID REFERENCES ab_experiments(id) ON DELETE CASCADE,
    variant VARCHAR(10) NOT NULL,
    request_count INTEGER DEFAULT 0,
    latency_sum_ms REAL DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    custom_metrics JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_ab_results_experiment ON ab_results(experiment_id);

-- ==========================================================================
-- Update timestamp trigger function
-- ==========================================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply triggers to tables with updated_at
CREATE TRIGGER update_models_updated_at BEFORE UPDATE ON models
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_training_jobs_updated_at BEFORE UPDATE ON training_jobs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_webhook_endpoints_updated_at BEFORE UPDATE ON webhook_endpoints
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_model_deployments_updated_at BEFORE UPDATE ON model_deployments
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_ab_experiments_updated_at BEFORE UPDATE ON ab_experiments
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_ab_results_updated_at BEFORE UPDATE ON ab_results
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ==========================================================================
-- Insert default data
-- ==========================================================================

-- No default data needed - this is a fresh development database

COMMENT ON DATABASE tinyforge IS 'TinyForgeAI development database';
