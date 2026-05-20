#!/bin/bash
# MOEA Quick Commands - Script de comandos rápidos
# Uso: source moea_quick_commands.sh

# Aliases para comandos comunes
alias moea-gen='python quick_moea.py'
alias moea-setup='python moea_setup.py'
alias moea-test='python moea_test_api.py'
alias moea-monitor='python moea_monitor.py'
alias moea-dash='python moea_dashboard.py'
alias moea-health='python moea_health.py'
alias moea-backup='python moea_backup.py'
alias moea-clean='python moea_cleanup.py'
alias moea-analytics='python moea_analytics.py'
alias moea-security='python moea_security.py'
alias moea-docs='python moea_docs.py'
alias moea-perf='python moea_performance.py'
alias moea-ai='python moea_ai_assistant.py'

# Funciones útiles
moea-full-setup() {
    echo "🚀 MOEA Full Setup"
    python quick_moea.py
    python moea_setup.py
    python verify_moea_project.py
    python moea_health.py
}

moea-full-test() {
    echo "🧪 MOEA Full Test Suite"
    python moea_health.py
    python moea_test_api.py
    python moea_security.py generated_projects/moea_optimization_system
    python moea_performance.py --report test_performance.json
}

moea-daily() {
    echo "📊 MOEA Daily Tasks"
    python moea_analytics.py --report daily_analytics.json
    python moea_cleanup.py --all
    python moea_backup.py create generated_projects/moea_optimization_system
}

echo "✅ MOEA Quick Commands cargados!"
echo "   Usa 'moea-full-setup', 'moea-full-test', 'moea-daily'"
echo "   O usa los aliases: moea-gen, moea-setup, moea-test, etc."

