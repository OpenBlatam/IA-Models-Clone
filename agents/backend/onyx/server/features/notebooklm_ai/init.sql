-- Database initialization script for notebooklm_ai
-- Creates tables and indexes for the application

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create image_generations table
CREATE TABLE IF NOT EXISTS image_generations (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    prompt TEXT NOT NULL,
    negative_prompt TEXT,
    pipeline_type VARCHAR(50) NOT NULL,
    model_type VARCHAR(100) NOT NULL,
    num_inference_steps INTEGER NOT NULL,
    guidance_scale DECIMAL(4,2) NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    seed INTEGER,
    batch_size INTEGER NOT NULL,
    result_url TEXT NOT NULL,
    processing_time DECIMAL(10,3) NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create batch_generations table
CREATE TABLE IF NOT EXISTS batch_generations (
    id SERIAL PRIMARY KEY,
    batch_id VARCHAR(100) UNIQUE NOT NULL,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    total_processing_time DECIMAL(10,3) NOT NULL,
    successful_generations INTEGER NOT NULL,
    failed_generations INTEGER NOT NULL,
    status VARCHAR(20) DEFAULT 'processing',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Create batch_generation_items table
CREATE TABLE IF NOT EXISTS batch_generation_items (
    id SERIAL PRIMARY KEY,
    batch_id VARCHAR(100) REFERENCES batch_generations(batch_id) ON DELETE CASCADE,
    generation_id INTEGER REFERENCES image_generations(id) ON DELETE CASCADE,
    status VARCHAR(20) DEFAULT 'pending',
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_image_generations_user_id ON image_generations(user_id);
CREATE INDEX IF NOT EXISTS idx_image_generations_created_at ON image_generations(created_at);
CREATE INDEX IF NOT EXISTS idx_image_generations_model_type ON image_generations(model_type);
CREATE INDEX IF NOT EXISTS idx_batch_generations_user_id ON batch_generations(user_id);
CREATE INDEX IF NOT EXISTS idx_batch_generations_status ON batch_generations(status);
CREATE INDEX IF NOT EXISTS idx_batch_generation_items_batch_id ON batch_generation_items(batch_id);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for users table
CREATE TRIGGER update_users_updated_at 
    BEFORE UPDATE ON users 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Insert default admin user (password: admin123)
INSERT INTO users (username, email, password_hash) 
VALUES ('admin', 'admin@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj4tbQJhHm2i')
ON CONFLICT (username) DO NOTHING;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO user; 