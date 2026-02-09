-- PostgreSQL initialization script for Music Analyzer AI
-- This script runs automatically when the database container is first created

-- Create extensions if needed
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search

-- Set timezone
SET timezone = 'UTC';

-- Create schema if needed
CREATE SCHEMA IF NOT EXISTS music_analyzer;

-- Example: Create tables (adjust based on your models)
-- Note: This is a template. Update based on your actual database models.

-- Users table (if needed)
CREATE TABLE IF NOT EXISTS music_analyzer.users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Analysis history table
CREATE TABLE IF NOT EXISTS music_analyzer.analysis_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    track_id VARCHAR(255) NOT NULL,
    track_name VARCHAR(500),
    artist_name VARCHAR(500),
    analysis_data JSONB,
    user_id UUID REFERENCES music_analyzer.users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_analysis_history_track_id ON music_analyzer.analysis_history(track_id);
CREATE INDEX IF NOT EXISTS idx_analysis_history_user_id ON music_analyzer.analysis_history(user_id);
CREATE INDEX IF NOT EXISTS idx_analysis_history_created_at ON music_analyzer.analysis_history(created_at DESC);

-- Create GIN index for JSONB search
CREATE INDEX IF NOT EXISTS idx_analysis_history_data ON music_analyzer.analysis_history USING GIN(analysis_data);

-- Cache table (if using database for caching)
CREATE TABLE IF NOT EXISTS music_analyzer.cache (
    key VARCHAR(255) PRIMARY KEY,
    value TEXT,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_cache_expires_at ON music_analyzer.cache(expires_at);

-- Function to clean expired cache entries
CREATE OR REPLACE FUNCTION music_analyzer.clean_expired_cache()
RETURNS void AS $$
BEGIN
    DELETE FROM music_analyzer.cache
    WHERE expires_at < CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT ALL PRIVILEGES ON SCHEMA music_analyzer TO music_analyzer;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA music_analyzer TO music_analyzer;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA music_analyzer TO music_analyzer;

-- Set default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA music_analyzer
    GRANT ALL ON TABLES TO music_analyzer;
ALTER DEFAULT PRIVILEGES IN SCHEMA music_analyzer
    GRANT ALL ON SEQUENCES TO music_analyzer;

-- Log initialization
DO $$
BEGIN
    RAISE NOTICE 'Music Analyzer AI database initialized successfully';
END $$;




