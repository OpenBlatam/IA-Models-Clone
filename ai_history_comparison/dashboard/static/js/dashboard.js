/**
 * AI History Comparison Dashboard JavaScript
 * Funcionalidades del dashboard web para visualización de datos
 */

class DashboardManager {
    constructor() {
        this.apiBaseUrl = window.location.origin.replace('8002', '8001');
        this.websocketUrl = `ws://${window.location.hostname}:8765/ws/dashboard`;
        this.websocket = null;
        this.charts = {};
        this.updateInterval = null;
        this.isConnected = false;
        
        this.init();
    }
    
    async init() {
        console.log('🚀 Inicializando Dashboard de AI History Comparison');
        
        // Cargar datos iniciales
        await this.loadInitialData();
        
        // Configurar WebSocket
        this.setupWebSocket();
        
        // Configurar actualizaciones automáticas
        this.setupAutoRefresh();
        
        // Configurar eventos
        this.setupEventListeners();
        
        console.log('✅ Dashboard inicializado correctamente');
    }
    
    async loadInitialData() {
        try {
            console.log('📊 Cargando datos iniciales...');
            
            // Cargar resumen general
            await this.loadDashboardOverview();
            
            // Cargar datos de rendimiento
            await this.loadPerformanceData();
            
            // Cargar métricas en tiempo real
            await this.loadRealtimeMetrics();
            
            // Cargar alertas
            await this.loadAlerts();
            
            // Cargar insights
            await this.loadInsights();
            
            // Cargar recomendaciones
            await this.loadRecommendations();
            
            // Cargar rendimiento ML
            await this.loadMLPerformance();
            
            // Cargar análisis de queries
            await this.loadQueryAnalysis();
            
            // Cargar tendencias
            await this.loadTrends();
            
            console.log('✅ Datos iniciales cargados');
            
        } catch (error) {
            console.error('❌ Error cargando datos iniciales:', error);
            this.showNotification('Error cargando datos iniciales', 'error');
        }
    }
    
    async loadDashboardOverview() {
        try {
            const response = await fetch('/api/dashboard/overview');
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            // Actualizar tarjetas de resumen
            this.updateOverviewCards(data.statistics);
            
            // Actualizar estado de conexión
            this.updateConnectionStatus(data.realtime_status);
            
        } catch (error) {
            console.error('Error cargando resumen del dashboard:', error);
        }
    }
    
    updateOverviewCards(stats) {
        if (!stats) return;
        
        // Total documentos
        const totalDocs = document.getElementById('total-documents');
        if (totalDocs) {
            totalDocs.textContent = stats.total_documents || 0;
        }
        
        // Calidad promedio
        const avgQuality = document.getElementById('avg-quality');
        if (avgQuality) {
            const quality = stats.average_quality || 0;
            avgQuality.textContent = quality.toFixed(2);
            avgQuality.style.color = this.getQualityColor(quality);
        }
        
        // Alertas activas
        const activeAlerts = document.getElementById('active-alerts');
        if (activeAlerts) {
            activeAlerts.textContent = stats.active_alerts || 0;
        }
        
        // Total insights
        const totalInsights = document.getElementById('total-insights');
        if (totalInsights) {
            totalInsights.textContent = stats.insights_generated || 0;
        }
    }
    
    getQualityColor(quality) {
        if (quality >= 0.8) return '#198754'; // Verde
        if (quality >= 0.6) return '#ffc107'; // Amarillo
        return '#dc3545'; // Rojo
    }
    
    async loadPerformanceData() {
        try {
            const response = await fetch('/api/dashboard/performance?days=7');
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            this.updatePerformanceChart(data);
            
        } catch (error) {
            console.error('Error cargando datos de rendimiento:', error);
        }
    }
    
    updatePerformanceChart(data) {
        const ctx = document.getElementById('performanceChart');
        if (!ctx) return;
        
        // Destruir gráfico existente si existe
        if (this.charts.performance) {
            this.charts.performance.destroy();
        }
        
        if (!data.daily_performance) {
            ctx.getContext('2d').clearRect(0, 0, ctx.width, ctx.height);
            return;
        }
        
        const days = Object.keys(data.daily_performance).sort();
        const qualityData = days.map(day => data.daily_performance[day].avg_quality || 0);
        const readabilityData = days.map(day => data.daily_performance[day].avg_readability || 0);
        const originalityData = days.map(day => data.daily_performance[day].avg_originality || 0);
        
        this.charts.performance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: days.map(day => new Date(day).toLocaleDateString()),
                datasets: [
                    {
                        label: 'Calidad',
                        data: qualityData,
                        borderColor: '#0d6efd',
                        backgroundColor: 'rgba(13, 110, 253, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: 'Legibilidad',
                        data: readabilityData,
                        borderColor: '#198754',
                        backgroundColor: 'rgba(25, 135, 84, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: 'Originalidad',
                        data: originalityData,
                        borderColor: '#ffc107',
                        backgroundColor: 'rgba(255, 193, 7, 0.1)',
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1
                    }
                },
                plugins: {
                    legend: {
                        position: 'top'
                    }
                }
            }
        });
    }
    
    async loadRealtimeMetrics() {
        try {
            const response = await fetch('/api/dashboard/realtime-metrics');
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            this.updateRealtimeMetrics(data.metrics);
            
        } catch (error) {
            console.error('Error cargando métricas en tiempo real:', error);
        }
    }
    
    updateRealtimeMetrics(metrics) {
        const container = document.getElementById('realtime-metrics');
        if (!container) return;
        
        if (!metrics || Object.keys(metrics).length === 0) {
            container.innerHTML = '<div class="text-center text-muted">No hay métricas disponibles</div>';
            return;
        }
        
        let html = '';
        for (const [metricType, metricData] of Object.entries(metrics)) {
            if (metricData && metricData.length > 0) {
                const latestMetric = metricData[metricData.length - 1];
                const value = latestMetric.value;
                const timestamp = new Date(latestMetric.timestamp).toLocaleTimeString();
                
                html += `
                    <div class="metric-card">
                        <div class="d-flex justify-content-between align-items-center">
                            <div>
                                <div class="metric-value">${value.toFixed(3)}</div>
                                <div class="metric-label">${this.getMetricLabel(metricType)}</div>
                            </div>
                            <div class="text-end">
                                <small class="text-muted">${timestamp}</small>
                            </div>
                        </div>
                    </div>
                `;
            }
        }
        
        container.innerHTML = html || '<div class="text-center text-muted">No hay métricas disponibles</div>';
    }
    
    getMetricLabel(metricType) {
        const labels = {
            'quality_score': 'Calidad',
            'readability': 'Legibilidad',
            'originality': 'Originalidad',
            'processing_time': 'Tiempo de Procesamiento',
            'user_satisfaction': 'Satisfacción del Usuario',
            'error_rate': 'Tasa de Error'
        };
        return labels[metricType] || metricType;
    }
    
    async loadAlerts() {
        try {
            const response = await fetch('/api/dashboard/alerts');
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            this.updateAlerts(data.alerts || []);
            
        } catch (error) {
            console.error('Error cargando alertas:', error);
        }
    }
    
    updateAlerts(alerts) {
        const container = document.getElementById('alerts-container');
        if (!container) return;
        
        if (alerts.length === 0) {
            container.innerHTML = '<div class="text-center text-muted">No hay alertas activas</div>';
            return;
        }
        
        let html = '';
        alerts.forEach(alert => {
            const levelClass = alert.level.toLowerCase();
            const levelIcon = this.getAlertIcon(alert.level);
            
            html += `
                <div class="alert-card ${levelClass}">
                    <div class="alert-level">
                        <i class="${levelIcon} me-1"></i>
                        ${alert.level.toUpperCase()}
                    </div>
                    <div class="alert-message">${alert.message}</div>
                    <div class="alert-timestamp">
                        ${new Date(alert.timestamp).toLocaleString()}
                    </div>
                    <div class="mt-2">
                        <button class="btn btn-sm btn-outline-primary" onclick="resolveAlert('${alert.id}')">
                            <i class="fas fa-check"></i> Resolver
                        </button>
                    </div>
                </div>
            `;
        });
        
        container.innerHTML = html;
    }
    
    getAlertIcon(level) {
        const icons = {
            'critical': 'fas fa-exclamation-triangle',
            'warning': 'fas fa-exclamation-circle',
            'info': 'fas fa-info-circle'
        };
        return icons[level.toLowerCase()] || 'fas fa-bell';
    }
    
    async loadInsights() {
        try {
            const response = await fetch('/api/dashboard/overview');
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            this.updateInsights(data.insights || []);
            
        } catch (error) {
            console.error('Error cargando insights:', error);
        }
    }
    
    updateInsights(insights) {
        const container = document.getElementById('insights-container');
        if (!container) return;
        
        if (insights.length === 0) {
            container.innerHTML = '<div class="text-center text-muted">No hay insights disponibles</div>';
            return;
        }
        
        let html = '';
        insights.forEach(insight => {
            html += `
                <div class="insight-card">
                    <div class="insight-title">${insight.title}</div>
                    <div class="insight-description">${insight.description}</div>
                    <div class="d-flex justify-content-between align-items-center">
                        <span class="insight-confidence">
                            Confianza: ${(insight.confidence * 100).toFixed(0)}%
                        </span>
                        <small class="text-muted">
                            ${new Date(insight.created_at).toLocaleDateString()}
                        </small>
                    </div>
                </div>
            `;
        });
        
        container.innerHTML = html;
    }
    
    async loadRecommendations() {
        try {
            const response = await fetch('/api/dashboard/overview');
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            this.updateRecommendations(data.recommendations || []);
            
        } catch (error) {
            console.error('Error cargando recomendaciones:', error);
        }
    }
    
    updateRecommendations(recommendations) {
        const container = document.getElementById('recommendations-container');
        if (!container) return;
        
        if (recommendations.length === 0) {
            container.innerHTML = '<div class="text-center text-muted">No hay recomendaciones disponibles</div>';
            return;
        }
        
        let html = '';
        recommendations.forEach(rec => {
            html += `
                <div class="recommendation-card">
                    <div class="recommendation-priority ${rec.priority.toLowerCase()}">
                        ${rec.priority}
                    </div>
                    <div class="recommendation-title">${rec.title}</div>
                    <div class="recommendation-description">${rec.description}</div>
                    <div class="text-muted">
                        <small>Mejora esperada: ${(rec.expected_improvement * 100).toFixed(0)}%</small>
                    </div>
                </div>
            `;
        });
        
        container.innerHTML = html;
    }
    
    async loadMLPerformance() {
        try {
            const response = await fetch('/api/dashboard/ml-performance');
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            this.updateMLPerformance(data);
            
        } catch (error) {
            console.error('Error cargando rendimiento ML:', error);
        }
    }
    
    updateMLPerformance(data) {
        const container = document.getElementById('ml-performance');
        if (!container) return;
        
        if (!data.models || Object.keys(data.models).length === 0) {
            container.innerHTML = '<div class="text-center text-muted">No hay modelos ML entrenados</div>';
            return;
        }
        
        let html = '';
        for (const [modelName, modelData] of Object.entries(data.models)) {
            html += `
                <div class="ml-model-card">
                    <div class="model-name">${modelName.replace('_', ' ').toUpperCase()}</div>
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            <div class="model-score">R²: ${modelData.r2_score.toFixed(3)}</div>
                            <small class="text-muted">MSE: ${modelData.mse.toFixed(4)}</small>
                        </div>
                        <div class="text-end">
                            <small class="text-muted">
                                Tiempo: ${modelData.training_time.toFixed(1)}s
                            </small>
                        </div>
                    </div>
                </div>
            `;
        }
        
        container.innerHTML = html;
    }
    
    async loadQueryAnalysis() {
        try {
            const response = await fetch('/api/dashboard/query-analysis');
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            this.updateQueryAnalysis(data);
            
        } catch (error) {
            console.error('Error cargando análisis de queries:', error);
        }
    }
    
    updateQueryAnalysis(data) {
        const container = document.getElementById('query-analysis');
        if (!container) return;
        
        if (!data.best_performing_queries || data.best_performing_queries.length === 0) {
            container.innerHTML = '<div class="text-center text-muted">No hay análisis de queries disponible</div>';
            return;
        }
        
        let html = '<h6>Mejores Queries:</h6>';
        data.best_performing_queries.forEach(query => {
            html += `
                <div class="query-item">
                    <div class="query-text">${query.query}</div>
                    <div class="query-stats">
                        <div class="query-stat">
                            <div class="query-stat-value">${query.avg_quality.toFixed(2)}</div>
                            <div class="query-stat-label">Calidad</div>
                        </div>
                        <div class="query-stat">
                            <div class="query-stat-value">${query.count}</div>
                            <div class="query-stat-label">Usos</div>
                        </div>
                    </div>
                </div>
            `;
        });
        
        container.innerHTML = html;
    }
    
    async loadTrends() {
        try {
            const response = await fetch('/api/dashboard/trends');
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            this.updateTrends(data.trends || {});
            
        } catch (error) {
            console.error('Error cargando tendencias:', error);
        }
    }
    
    updateTrends(trends) {
        const container = document.getElementById('trends-container');
        if (!container) return;
        
        if (Object.keys(trends).length === 0) {
            container.innerHTML = '<div class="text-center text-muted">No hay tendencias disponibles</div>';
            return;
        }
        
        let html = '';
        for (const [metricType, trend] of Object.entries(trends)) {
            html += `
                <div class="trend-item">
                    <div class="trend-metric">${this.getMetricLabel(metricType)}</div>
                    <div class="trend-direction ${trend.direction}">
                        ${trend.direction === 'increasing' ? '↗️' : 
                          trend.direction === 'decreasing' ? '↘️' : '→'} 
                        ${trend.direction}
                    </div>
                    <div class="trend-confidence">
                        Confianza: ${(trend.confidence * 100).toFixed(0)}% | 
                        Fuerza: ${(trend.strength * 100).toFixed(0)}% | 
                        Cambio: ${trend.change_rate.toFixed(3)}
                    </div>
                </div>
            `;
        }
        
        container.innerHTML = html;
    }
    
    setupWebSocket() {
        try {
            this.websocket = new WebSocket(this.websocketUrl);
            
            this.websocket.onopen = () => {
                console.log('🔗 WebSocket conectado');
                this.isConnected = true;
                this.updateConnectionStatus({ is_running: true });
            };
            
            this.websocket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                } catch (error) {
                    console.error('Error procesando mensaje WebSocket:', error);
                }
            };
            
            this.websocket.onclose = () => {
                console.log('🔌 WebSocket desconectado');
                this.isConnected = false;
                this.updateConnectionStatus({ is_running: false });
                
                // Reconectar después de 5 segundos
                setTimeout(() => {
                    this.setupWebSocket();
                }, 5000);
            };
            
            this.websocket.onerror = (error) => {
                console.error('❌ Error WebSocket:', error);
                this.isConnected = false;
                this.updateConnectionStatus({ is_running: false });
            };
            
        } catch (error) {
            console.error('Error configurando WebSocket:', error);
        }
    }
    
    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'dashboard_update':
                this.updateOverviewCards(data.data.statistics);
                this.updateConnectionStatus(data.data.realtime_status);
                break;
            case 'new_alert':
                this.showNotification(`Nueva alerta: ${data.alert.title}`, 'warning');
                this.loadAlerts(); // Recargar alertas
                break;
            case 'alert_escalation':
                this.showNotification(`Alerta escalada: ${data.alert.title}`, 'danger');
                this.loadAlerts(); // Recargar alertas
                break;
            default:
                console.log('Mensaje WebSocket no reconocido:', data);
        }
    }
    
    updateConnectionStatus(status) {
        const statusElement = document.getElementById('connection-status');
        const textElement = document.getElementById('connection-text');
        
        if (statusElement && textElement) {
            if (status && status.is_running) {
                statusElement.className = 'fas fa-circle text-success me-1';
                textElement.textContent = 'Conectado';
            } else {
                statusElement.className = 'fas fa-circle text-danger me-1';
                textElement.textContent = 'Desconectado';
            }
        }
    }
    
    setupAutoRefresh() {
        // Actualizar datos cada 30 segundos
        this.updateInterval = setInterval(() => {
            this.loadDashboardOverview();
            this.loadRealtimeMetrics();
            this.loadAlerts();
        }, 30000);
    }
    
    setupEventListeners() {
        // Configurar eventos de botones
        window.refreshMetrics = () => this.loadRealtimeMetrics();
        window.refreshAlerts = () => this.loadAlerts();
        window.optimizeML = () => this.optimizeML();
        window.resolveAlert = (alertId) => this.resolveAlert(alertId);
    }
    
    async optimizeML() {
        try {
            this.showNotification('Iniciando optimización de ML...', 'info');
            
            const response = await fetch('/api/dashboard/optimize-ml', {
                method: 'POST'
            });
            
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            this.showNotification('Optimización de ML iniciada en segundo plano', 'success');
            
            // Recargar rendimiento ML después de un tiempo
            setTimeout(() => {
                this.loadMLPerformance();
            }, 10000);
            
        } catch (error) {
            console.error('Error optimizando ML:', error);
            this.showNotification('Error iniciando optimización de ML', 'error');
        }
    }
    
    async resolveAlert(alertId) {
        try {
            const response = await fetch(`/api/dashboard/resolve-alert/${alertId}`, {
                method: 'POST'
            });
            
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            this.showNotification('Alerta resuelta exitosamente', 'success');
            this.loadAlerts(); // Recargar alertas
            
        } catch (error) {
            console.error('Error resolviendo alerta:', error);
            this.showNotification('Error resolviendo alerta', 'error');
        }
    }
    
    showNotification(message, type = 'info') {
        const toast = document.getElementById('notification-toast');
        const toastMessage = document.getElementById('toast-message');
        
        if (toast && toastMessage) {
            toastMessage.textContent = message;
            
            // Cambiar color según el tipo
            const toastHeader = toast.querySelector('.toast-header');
            const icon = toastHeader.querySelector('i');
            
            icon.className = `fas me-2 ${
                type === 'success' ? 'fa-check-circle text-success' :
                type === 'error' ? 'fa-exclamation-circle text-danger' :
                type === 'warning' ? 'fa-exclamation-triangle text-warning' :
                'fa-info-circle text-primary'
            }`;
            
            // Mostrar toast
            const bsToast = new bootstrap.Toast(toast);
            bsToast.show();
        }
    }
    
    destroy() {
        // Limpiar intervalos
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
        
        // Cerrar WebSocket
        if (this.websocket) {
            this.websocket.close();
        }
        
        // Destruir gráficos
        Object.values(this.charts).forEach(chart => {
            if (chart && typeof chart.destroy === 'function') {
                chart.destroy();
            }
        });
    }
}

// Inicializar dashboard cuando se carga la página
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new DashboardManager();
});

// Limpiar recursos cuando se cierra la página
window.addEventListener('beforeunload', () => {
    if (window.dashboard) {
        window.dashboard.destroy();
    }
});



























