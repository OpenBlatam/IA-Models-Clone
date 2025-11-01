#!/bin/bash
# Script de Benchmarking para Blatam Academy Features

echo "📊 Benchmarking del Sistema"
echo "=========================="

# Configuración
BASE_URL="http://localhost:8000"
NUM_REQUESTS=1000
CONCURRENT=50

echo ""
echo "🔍 Configuración:"
echo "  Base URL: $BASE_URL"
echo "  Requests: $NUM_REQUESTS"
echo "  Concurrent: $CONCURRENT"
echo ""

# Test de Latencia
echo "⏱️  Test de Latencia..."
ab -n $NUM_REQUESTS -c $CONCURRENT "${BASE_URL}/health" > latency_report.txt
echo "✅ Resultados guardados en latency_report.txt"

# Test de Throughput
echo ""
echo "🚀 Test de Throughput..."
siege -c $CONCURRENT -t 60s "${BASE_URL}/health" > throughput_report.txt
echo "✅ Resultados guardados en throughput_report.txt"

# Test de KV Cache
echo ""
echo "⚡ Test de KV Cache..."
python bulk/core/ultra_adaptive_kv_cache_benchmark.py \
    --iterations $NUM_REQUESTS \
    --concurrent $CONCURRENT \
    --output benchmark_results.json

echo ""
echo "📊 Benchmarking completo"
echo "Ver resultados en:"
echo "  - latency_report.txt"
echo "  - throughput_report.txt"
echo "  - benchmark_results.json"

