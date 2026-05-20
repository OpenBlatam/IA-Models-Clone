// Dashboard JavaScript - Mejoras y gráficos
// ===========================================

class BulkChatDashboard {
    constructor(apiBase) {
        this.apiBase = apiBase;
        this.charts = {};
        this.updateInterval = 5000; // 5 segundos
        this.init();
    }
    
    init() {
        this.setupCharts();
        this.startAutoUpdate();
        this.setupEventListeners();
    }
    
    setupCharts() {
        // Gráfico de sesiones activas
        const ctxSessions = document.getElementById('sessionsChart');
        if (ctxSessions) {
            this.charts.sessions = new Chart(ctxSessions, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Sesiones Activas',
                        data: [],
                        borderColor: 'rgb(102, 126, 234)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: { beginAtZero: true }
                    }
                }
            });
        }
        
        // Gráfico de mensajes
        const ctxMessages = document.getElementById('messagesChart');
        if (ctxMessages) {
            this.charts.messages = new Chart(ctxMessages, {
                type: 'bar',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Mensajes por Hora',
                        data: [],
                        backgroundColor: 'rgba(102, 126, 234, 0.5)'
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: { beginAtZero: true }
                    }
                }
            });
        }
    }
    
    async fetchData(endpoint) {
        try {
            const response = await fetch(`${this.apiBase}${endpoint}`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        } catch (error) {
            console.error(`Error fetching ${endpoint}:`, error);
            throw error;
        }
    }
    
    async updateStats() {
        try {
            const [metrics, sessions] = await Promise.all([
                this.fetchData('/api/v1/chat/metrics'),
                this.fetchData('/api/v1/chat/sessions')
            ]);
            
            // Actualizar estadísticas
            this.updateStatCard('activeSessions', metrics.active_sessions || 0);
            this.updateStatCard('totalSessions', metrics.total_sessions || 0);
            this.updateStatCard('totalMessages', metrics.total_messages || 0);
            
            // Cache stats
            try {
                const cacheStats = await this.fetchData('/api/v1/chat/cache/stats');
                const hitRate = (cacheStats.hit_rate * 100).toFixed(1);
                this.updateStatCard('cacheHitRate', `${hitRate}%`);
            } catch (e) {
                this.updateStatCard('cacheHitRate', 'N/A');
            }
            
            // Actualizar gráfico de sesiones
            if (this.charts.sessions) {
                const now = new Date().toLocaleTimeString();
                this.charts.sessions.data.labels.push(now);
                this.charts.sessions.data.datasets[0].data.push(metrics.active_sessions || 0);
                
                // Mantener solo últimos 20 puntos
                if (this.charts.sessions.data.labels.length > 20) {
                    this.charts.sessions.data.labels.shift();
                    this.charts.sessions.data.datasets[0].data.shift();
                }
                
                this.charts.sessions.update();
            }
            
            // Actualizar lista de sesiones
            this.updateSessionsList(sessions.sessions || []);
            
        } catch (error) {
            console.error('Error updating stats:', error);
            const listDiv = document.getElementById('sessionsList');
            if (listDiv) {
                listDiv.innerHTML = `<div class="error">Error cargando datos: ${error.message}</div>`;
            }
        }
    }
    
    updateStatCard(id, value) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
        }
    }
    
    updateSessionsList(sessions) {
        const listDiv = document.getElementById('sessionsList');
        if (!listDiv) return;
        
        if (sessions.length === 0) {
            listDiv.innerHTML = '<div class="loading">No hay sesiones activas</div>';
            return;
        }
        
        listDiv.innerHTML = sessions.map(session => `
            <div class="session-item">
                <div class="session-info">
                    <div class="session-id">${session.session_id.substring(0, 8)}...</div>
                    <div class="session-meta">
                        ${session.message_count} mensajes • 
                        ${session.user_id || 'Anónimo'} • 
                        <span class="status-badge status-${session.state}">${session.state}</span>
                    </div>
                </div>
                <div class="session-actions">
                    <button class="btn-secondary" onclick="dashboard.viewSession('${session.session_id}')">Ver</button>
                    ${session.state === 'paused' ? 
                        `<button class="btn-primary" onclick="dashboard.resumeSession('${session.session_id}')">Reanudar</button>` :
                        `<button class="btn-secondary" onclick="dashboard.pauseSession('${session.session_id}')">Pausar</button>`
                    }
                </div>
            </div>
        `).join('');
    }
    
    async pauseSession(sessionId) {
        try {
            await fetch(`${this.apiBase}/api/v1/chat/sessions/${sessionId}/pause`, { method: 'POST' });
            this.updateStats();
        } catch (error) {
            alert(`Error: ${error.message}`);
        }
    }
    
    async resumeSession(sessionId) {
        try {
            await fetch(`${this.apiBase}/api/v1/chat/sessions/${sessionId}/resume`, { method: 'POST' });
            this.updateStats();
        } catch (error) {
            alert(`Error: ${error.message}`);
        }
    }
    
    viewSession(sessionId) {
        window.open(`${this.apiBase}/api/v1/chat/sessions/${sessionId}`, '_blank');
    }
    
    startAutoUpdate() {
        this.updateStats();
        setInterval(() => this.updateStats(), this.updateInterval);
    }
    
    setupEventListeners() {
        // Agregar listeners adicionales si es necesario
    }
}

// Inicializar dashboard cuando el DOM esté listo
document.addEventListener('DOMContentLoaded', () => {
    const apiBase = window.location.origin;
    window.dashboard = new BulkChatDashboard(apiBase);
});
































