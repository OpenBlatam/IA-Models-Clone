-- Initialize database schema
-- This script runs automatically when PostgreSQL container starts for the first time

-- Create database if it doesn't exist (usually already created by POSTGRES_DB)
-- CREATE DATABASE IF NOT EXISTS manuales_hogar;

-- Connect to the database
\c manuales_hogar;

-- Create extensions if needed
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Note: Alembic migrations will handle table creation
-- This file is just for any initial setup needed




