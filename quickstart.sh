#!/bin/bash

# Quickstart script for Agentic RAG Research Assistant

set -e

echo "🚀 Starting quickstart for Agentic RAG Research Assistant..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ .env file not found. Please create it with your API keys."
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Start Qdrant
echo "🗄️ Starting Qdrant vector database..."
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant

# Wait for Qdrant to be ready
echo "⏳ Waiting for Qdrant to be ready..."
sleep 10

# Run ingestion
echo "📥 Running ingestion on seed data..."
python src/ingestion.py

# Start the application
echo "🌐 Starting the application..."
python src/app.py &
APP_PID=$!

# Wait a bit for app to start
sleep 5

# Run a sample query
echo "🔍 Running a sample query..."
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What are the key principles of AI ethics?"}' \
     | jq .

echo "✅ Quickstart complete! Application is running on http://localhost:8000"
echo "To stop: kill $APP_PID && docker stop qdrant && docker rm qdrant"

# Keep running
wait $APP_PID
