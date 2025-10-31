/**
 * Business Agents System - Web Interface JavaScript
 * ================================================
 * 
 * Advanced JavaScript functionality for the Business Agents web interface.
 */

class BusinessAgentsApp {
    constructor() {
        this.apiBaseUrl = 'http://localhost:8000';
        this.authToken = localStorage.getItem('auth_token');
        this.currentUser = null;
        this.charts = {};
        this.refreshInterval = null;
        
        this.init();
    }

    async init() {
        await this.checkAuthentication();
        this.setupEventListeners();
        this.loadInitialData();
        this.startAutoRefresh();
    }

    // Authentication
    async checkAuthentication() {
        if (!this.authToken) {
            this.showLoginModal();
            return;
        }

        try {
            const user = await this.apiCall('/auth/me');
            this.currentUser = user;
            this.updateUserInterface();
        } catch (error) {
            console.error('Authentication failed:', error);
            this.logout();
        }
    }

    showLoginModal() {
        const modal = document.createElement('div');
        modal.className = 'fixed inset-0 bg-gray-600 bg-opacity-50 z-50';
        modal.innerHTML = `
            <div class="flex items-center justify-center min-h-screen p-4">
                <div class="bg-white rounded-lg shadow-xl max-w-md w-full">
                    <div class="px-6 py-4 border-b border-gray-200">
                        <h3 class="text-lg font-medium text-gray-900">Login to Business Agents</h3>
                    </div>
                    <div class="p-6">
                        <form id="login-form">
                            <div class="mb-4">
                                <label class="block text-sm font-medium text-gray-700 mb-2">Username</label>
                                <input type="text" id="login-username" class="w-full border-gray-300 rounded-md shadow-sm" required>
                            </div>
                            <div class="mb-4">
                                <label class="block text-sm font-medium text-gray-700 mb-2">Password</label>
                                <input type="password" id="login-password" class="w-full border-gray-300 rounded-md shadow-sm" required>
                            </div>
                        </form>
                    </div>
                    <div class="px-6 py-4 border-t border-gray-200 flex justify-end space-x-3">
                        <button class="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700" onclick="app.login()">Login</button>
                    </div>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
    }

    async login() {
        const username = document.getElementById('login-username').value;
        const password = document.getElementById('login-password').value;

        try {
            const response = await this.apiCall('/auth/login', 'POST', {
                username,
                password
            });

            this.authToken = response.access_token;
            localStorage.setItem('auth_token', this.authToken);
            this.currentUser = response.user;
            
            // Remove login modal
            document.querySelector('.fixed.inset-0').remove();
            this.updateUserInterface();
            this.loadInitialData();
        } catch (error) {
            console.error('Login failed:', error);
            alert('Login failed. Please check your credentials.');
        }
    }

    logout() {
        this.authToken = null;
        this.currentUser = null;
        localStorage.removeItem('auth_token');
        window.location.reload();
    }

    updateUserInterface() {
        if (this.currentUser) {
            document.getElementById('username').textContent = this.currentUser.username;
        }
    }

    // Event Listeners
    setupEventListeners() {
        // Sidebar toggle
        document.getElementById('sidebar-toggle')?.addEventListener('click', () => this.toggleSidebar());
        
        // Navigation
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const section = item.dataset.section;
                this.showSection(section);
            });
        });

        // User menu
        document.getElementById('user-menu')?.addEventListener('click', () => this.toggleUserMenu());
        
        // Logout
        document.getElementById('logout-btn')?.addEventListener('click', () => this.logout());

        // Modal close on outside click
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('fixed') && e.target.classList.contains('inset-0')) {
                e.target.classList.add('hidden');
            }
        });
    }

    toggleSidebar() {
        const sidebar = document.getElementById('sidebar');
        sidebar.classList.toggle('collapsed');
    }

    showSection(sectionName) {
        // Hide all sections
        document.querySelectorAll('.section').forEach(section => {
            section.classList.add('hidden');
        });

        // Show selected section
        const targetSection = document.getElementById(sectionName + '-section');
        if (targetSection) {
            targetSection.classList.remove('hidden');
        }

        // Update navigation
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('bg-gray-100', 'text-gray-900');
            item.classList.add('text-gray-600');
        });

        const activeItem = document.querySelector(`[data-section="${sectionName}"]`);
        if (activeItem) {
            activeItem.classList.remove('text-gray-600');
            activeItem.classList.add('bg-gray-100', 'text-gray-900');
        }

        // Load section-specific data
        this.loadSectionData(sectionName);
    }

    toggleUserMenu() {
        const dropdown = document.getElementById('user-dropdown');
        if (dropdown) {
            dropdown.classList.toggle('hidden');
        }
    }

    // Data Loading
    async loadInitialData() {
        await this.loadDashboardData();
    }

    async loadSectionData(sectionName) {
        switch(sectionName) {
            case 'dashboard':
                await this.loadDashboardData();
                break;
            case 'agents':
                await this.loadAgentsData();
                break;
            case 'workflows':
                await this.loadWorkflowsData();
                break;
            case 'documents':
                await this.loadDocumentsData();
                break;
            case 'analytics':
                await this.loadAnalyticsData();
                break;
            case 'templates':
                await this.loadTemplatesData();
                break;
        }
    }

    async loadDashboardData() {
        try {
            // Load system overview
            const overview = await this.apiCall('/business-agents/');
            
            // Load performance metrics
            const performance = await this.apiCall('/analytics/performance');
            
            // Update dashboard stats
            this.updateDashboardStats(overview, performance);
            
            // Load charts
            this.loadDashboardCharts(performance);
            
            // Load recent activity
            await this.loadRecentActivity();
            
        } catch (error) {
            console.error('Failed to load dashboard data:', error);
        }
    }

    updateDashboardStats(overview, performance) {
        const stats = {
            activeAgents: overview.active_agents || 0,
            runningWorkflows: performance.running_workflows || 0,
            documentsGenerated: performance.total_documents_generated || 0,
            successRate: performance.success_rate || 0
        };

        document.getElementById('active-agents').textContent = stats.activeAgents;
        document.getElementById('running-workflows').textContent = stats.runningWorkflows;
        document.getElementById('documents-generated').textContent = stats.documentsGenerated;
        document.getElementById('success-rate').textContent = Math.round(stats.successRate) + '%';
    }

    loadDashboardCharts(performance) {
        // Workflow Performance Chart
        this.createWorkflowChart(performance);
        
        // Business Area Distribution Chart
        this.createBusinessAreaChart(performance.business_area_distribution || {});
    }

    createWorkflowChart(performance) {
        const ctx = document.getElementById('workflow-chart');
        if (!ctx) return;

        if (this.charts.workflow) {
            this.charts.workflow.destroy();
        }

        this.charts.workflow = new Chart(ctx, {
            type: 'line',
            data: {
                labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                datasets: [{
                    label: 'Completed',
                    data: [performance.completed_workflows || 0, 19, 3, 5, 2, 3],
                    borderColor: 'rgb(34, 197, 94)',
                    backgroundColor: 'rgba(34, 197, 94, 0.1)',
                    tension: 0.1
                }, {
                    label: 'Failed',
                    data: [performance.failed_workflows || 0, 3, 1, 1, 0, 1],
                    borderColor: 'rgb(239, 68, 68)',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }

    createBusinessAreaChart(businessAreas) {
        const ctx = document.getElementById('business-area-chart');
        if (!ctx) return;

        if (this.charts.businessArea) {
            this.charts.businessArea.destroy();
        }

        const labels = Object.keys(businessAreas);
        const data = Object.values(businessAreas);

        this.charts.businessArea = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels.length > 0 ? labels : ['Marketing', 'Sales', 'Operations', 'HR', 'Finance', 'Legal', 'Technical', 'Content'],
                datasets: [{
                    data: data.length > 0 ? data : [25, 20, 15, 10, 12, 8, 15, 5],
                    backgroundColor: [
                        '#3B82F6', '#10B981', '#F59E0B', '#EF4444',
                        '#8B5CF6', '#06B6D4', '#84CC16', '#F97316'
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }

    async loadRecentActivity() {
        try {
            // This would be a real API call in production
            const activities = [
                { type: 'workflow', message: 'Marketing Campaign workflow completed', time: '2 minutes ago', status: 'success' },
                { type: 'document', message: 'Business Plan document generated', time: '5 minutes ago', status: 'success' },
                { type: 'agent', message: 'Sales Agent executed lead generation', time: '10 minutes ago', status: 'success' },
                { type: 'workflow', message: 'HR Onboarding workflow started', time: '15 minutes ago', status: 'running' },
                { type: 'document', message: 'Financial Report document generated', time: '20 minutes ago', status: 'success' }
            ];

            const container = document.getElementById('recent-activity');
            if (container) {
                container.innerHTML = activities.map(activity => `
                    <div class="flex items-center space-x-3">
                        <div class="flex-shrink-0">
                            <div class="w-8 h-8 rounded-full flex items-center justify-center ${this.getActivityIconClass(activity.type, activity.status)}">
                                <i class="fas ${this.getActivityIcon(activity.type)} text-sm"></i>
                            </div>
                        </div>
                        <div class="flex-1 min-w-0">
                            <p class="text-sm text-gray-900">${activity.message}</p>
                            <p class="text-sm text-gray-500">${activity.time}</p>
                        </div>
                    </div>
                `).join('');
            }
        } catch (error) {
            console.error('Failed to load recent activity:', error);
        }
    }

    getActivityIcon(type) {
        const icons = {
            workflow: 'fa-project-diagram',
            document: 'fa-file-alt',
            agent: 'fa-robot'
        };
        return icons[type] || 'fa-info-circle';
    }

    getActivityIconClass(type, status) {
        if (status === 'success') return 'bg-green-100 text-green-600';
        if (status === 'running') return 'bg-blue-100 text-blue-600';
        if (status === 'error') return 'bg-red-100 text-red-600';
        return 'bg-gray-100 text-gray-600';
    }

    async loadAgentsData() {
        try {
            const agents = await this.apiCall('/business-agents/agents');
            this.renderAgents(agents);
        } catch (error) {
            console.error('Failed to load agents:', error);
        }
    }

    renderAgents(agents) {
        const container = document.getElementById('agents-grid');
        if (!container) return;

        container.innerHTML = agents.map(agent => `
            <div class="bg-white rounded-lg shadow p-6 card-hover">
                <div class="flex items-center mb-4">
                    <div class="p-3 rounded-full bg-blue-100 text-blue-600">
                        <i class="fas fa-robot text-xl"></i>
                    </div>
                    <div class="ml-4">
                        <h3 class="text-lg font-medium text-gray-900">${agent.name}</h3>
                        <p class="text-sm text-gray-600">${agent.business_area}</p>
                    </div>
                </div>
                <p class="text-gray-600 mb-4">${agent.description}</p>
                <div class="flex items-center justify-between">
                    <span class="text-sm text-gray-500">${agent.capabilities.length} capabilities</span>
                    <button class="px-3 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700" onclick="app.executeAgent('${agent.id}')">
                        Execute
                    </button>
                </div>
            </div>
        `).join('');
    }

    async loadWorkflowsData() {
        try {
            const workflows = await this.apiCall('/business-agents/workflows');
            this.renderWorkflows(workflows);
        } catch (error) {
            console.error('Failed to load workflows:', error);
        }
    }

    renderWorkflows(workflows) {
        const container = document.getElementById('workflows-table');
        if (!container) return;

        container.innerHTML = workflows.map(workflow => `
            <tr>
                <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">${workflow.name}</td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${workflow.business_area}</td>
                <td class="px-6 py-4 whitespace-nowrap">
                    <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${this.getStatusClass(workflow.status)}">
                        ${workflow.status}
                    </span>
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    <div class="w-full bg-gray-200 rounded-full h-2">
                        <div class="bg-blue-600 h-2 rounded-full" style="width: ${this.getProgressPercentage(workflow.status)}%"></div>
                    </div>
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-sm font-medium">
                    <button class="text-blue-600 hover:text-blue-900 mr-3" onclick="app.viewWorkflow('${workflow.id}')">View</button>
                    <button class="text-green-600 hover:text-green-900 mr-3" onclick="app.executeWorkflow('${workflow.id}')">Execute</button>
                    <button class="text-red-600 hover:text-red-900" onclick="app.deleteWorkflow('${workflow.id}')">Delete</button>
                </td>
            </tr>
        `).join('');
    }

    getStatusClass(status) {
        const classes = {
            'active': 'bg-green-100 text-green-800',
            'completed': 'bg-blue-100 text-blue-800',
            'failed': 'bg-red-100 text-red-800',
            'draft': 'bg-gray-100 text-gray-800',
            'paused': 'bg-yellow-100 text-yellow-800'
        };
        return classes[status] || 'bg-gray-100 text-gray-800';
    }

    getProgressPercentage(status) {
        const percentages = {
            'active': 75,
            'completed': 100,
            'failed': 0,
            'draft': 0,
            'paused': 50
        };
        return percentages[status] || 0;
    }

    async loadDocumentsData() {
        try {
            const documents = await this.apiCall('/business-agents/documents');
            this.renderDocuments(documents);
        } catch (error) {
            console.error('Failed to load documents:', error);
        }
    }

    renderDocuments(documents) {
        const container = document.getElementById('documents-grid');
        if (!container) return;

        container.innerHTML = documents.map(doc => `
            <div class="bg-white rounded-lg shadow p-6 card-hover">
                <div class="flex items-center mb-4">
                    <div class="p-3 rounded-full bg-green-100 text-green-600">
                        <i class="fas fa-file-alt text-xl"></i>
                    </div>
                    <div class="ml-4">
                        <h3 class="text-lg font-medium text-gray-900">${doc.title}</h3>
                        <p class="text-sm text-gray-600">${doc.format}</p>
                    </div>
                </div>
                <p class="text-gray-600 mb-4">Size: ${this.formatFileSize(doc.size_bytes)}</p>
                <div class="flex items-center justify-between">
                    <span class="text-sm text-gray-500">${new Date(doc.created_at).toLocaleDateString()}</span>
                    <button class="px-3 py-1 text-sm bg-green-600 text-white rounded hover:bg-green-700" onclick="app.downloadDocument('${doc.id}')">
                        Download
                    </button>
                </div>
            </div>
        `).join('');
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    async loadAnalyticsData() {
        try {
            const dashboard = await this.apiCall('/analytics/dashboard');
            this.renderAnalytics(dashboard);
        } catch (error) {
            console.error('Failed to load analytics:', error);
        }
    }

    renderAnalytics(dashboard) {
        // Performance metrics
        const performanceContainer = document.getElementById('performance-metrics');
        if (performanceContainer) {
            performanceContainer.innerHTML = `
                <div class="space-y-4">
                    <div class="flex justify-between">
                        <span class="text-sm text-gray-600">Total Workflows</span>
                        <span class="text-sm font-medium">${dashboard.overview.total_workflows}</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-sm text-gray-600">Success Rate</span>
                        <span class="text-sm font-medium">${Math.round(dashboard.overview.success_rate)}%</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-sm text-gray-600">Total Documents</span>
                        <span class="text-sm font-medium">${dashboard.overview.total_documents}</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-sm text-gray-600">Health Score</span>
                        <span class="text-sm font-medium">${Math.round(dashboard.overview.health_score)}/100</span>
                    </div>
                </div>
            `;
        }

        // Usage statistics
        const usageContainer = document.getElementById('usage-stats');
        if (usageContainer) {
            usageContainer.innerHTML = `
                <div class="space-y-4">
                    <div class="flex justify-between">
                        <span class="text-sm text-gray-600">Total Metrics</span>
                        <span class="text-sm font-medium">${dashboard.activity.total_metrics}</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-sm text-gray-600">Recent Activity (24h)</span>
                        <span class="text-sm font-medium">${dashboard.activity.recent_metrics_24h}</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-sm text-gray-600">Trend</span>
                        <span class="text-sm font-medium capitalize">${dashboard.activity.metrics_trend}</span>
                    </div>
                </div>
            `;
        }
    }

    async loadTemplatesData() {
        try {
            const templates = await this.apiCall('/business-agents/workflow-templates');
            this.renderTemplates(templates);
        } catch (error) {
            console.error('Failed to load templates:', error);
        }
    }

    renderTemplates(templates) {
        const container = document.getElementById('templates-grid');
        if (!container) return;

        // Flatten templates object
        const allTemplates = [];
        Object.keys(templates).forEach(area => {
            templates[area].forEach(template => {
                allTemplates.push({...template, business_area: area});
            });
        });

        container.innerHTML = allTemplates.map(template => `
            <div class="bg-white rounded-lg shadow p-6 card-hover">
                <div class="flex items-center mb-4">
                    <div class="p-3 rounded-full bg-purple-100 text-purple-600">
                        <i class="fas fa-layer-group text-xl"></i>
                    </div>
                    <div class="ml-4">
                        <h3 class="text-lg font-medium text-gray-900">${template.name}</h3>
                        <p class="text-sm text-gray-600">${template.business_area}</p>
                    </div>
                </div>
                <p class="text-gray-600 mb-4">${template.description}</p>
                <div class="flex items-center justify-between">
                    <span class="text-sm text-gray-500">${template.steps.length} steps</span>
                    <button class="px-3 py-1 text-sm bg-purple-600 text-white rounded hover:bg-purple-700" onclick="app.useTemplate('${template.name}')">
                        Use Template
                    </button>
                </div>
            </div>
        `).join('');
    }

    // Action Methods
    async executeAgent(agentId) {
        try {
            const result = await this.apiCall(`/business-agents/agents/${agentId}/execute`, 'POST', {
                capability_name: 'default',
                inputs: {},
                parameters: {}
            });
            
            this.showNotification('Agent executed successfully!', 'success');
        } catch (error) {
            console.error('Failed to execute agent:', error);
            this.showNotification('Failed to execute agent', 'error');
        }
    }

    async executeWorkflow(workflowId) {
        try {
            const result = await this.apiCall(`/business-agents/workflows/${workflowId}/execute`, 'POST');
            this.showNotification('Workflow execution started!', 'success');
            this.loadWorkflowsData(); // Refresh the list
        } catch (error) {
            console.error('Failed to execute workflow:', error);
            this.showNotification('Failed to execute workflow', 'error');
        }
    }

    async deleteWorkflow(workflowId) {
        if (!confirm('Are you sure you want to delete this workflow?')) {
            return;
        }

        try {
            await this.apiCall(`/business-agents/workflows/${workflowId}`, 'DELETE');
            this.showNotification('Workflow deleted successfully!', 'success');
            this.loadWorkflowsData(); // Refresh the list
        } catch (error) {
            console.error('Failed to delete workflow:', error);
            this.showNotification('Failed to delete workflow', 'error');
        }
    }

    async downloadDocument(documentId) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/business-agents/documents/${documentId}/download`, {
                headers: {
                    'Authorization': `Bearer ${this.authToken}`
                }
            });
            
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `document-${documentId}`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
            } else {
                throw new Error('Download failed');
            }
        } catch (error) {
            console.error('Failed to download document:', error);
            this.showNotification('Failed to download document', 'error');
        }
    }

    useTemplate(templateName) {
        this.showNotification(`Template "${templateName}" selected!`, 'info');
        // Implementation for using template would go here
    }

    viewWorkflow(workflowId) {
        this.showNotification(`Viewing workflow ${workflowId}`, 'info');
        // Implementation for viewing workflow would go here
    }

    // Utility Methods
    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `fixed top-4 right-4 p-4 rounded-lg shadow-lg z-50 ${this.getNotificationClass(type)}`;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 3000);
    }

    getNotificationClass(type) {
        const classes = {
            'success': 'bg-green-100 text-green-800 border border-green-200',
            'error': 'bg-red-100 text-red-800 border border-red-200',
            'warning': 'bg-yellow-100 text-yellow-800 border border-yellow-200',
            'info': 'bg-blue-100 text-blue-800 border border-blue-200'
        };
        return classes[type] || classes['info'];
    }

    startAutoRefresh() {
        // Refresh dashboard data every 30 seconds
        this.refreshInterval = setInterval(() => {
            if (this.currentSection === 'dashboard') {
                this.loadDashboardData();
            }
        }, 30000);
    }

    stopAutoRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }
    }

    // API Helper
    async apiCall(endpoint, method = 'GET', data = null) {
        const config = {
            method,
            headers: {
                'Content-Type': 'application/json',
            }
        };

        if (this.authToken) {
            config.headers['Authorization'] = `Bearer ${this.authToken}`;
        }

        if (data) {
            config.body = JSON.stringify(data);
        }

        try {
            const response = await fetch(`${this.apiBaseUrl}${endpoint}`, config);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('API call failed:', error);
            throw error;
        }
    }
}

// Global functions for HTML onclick handlers
function showCreateWorkflowModal() {
    document.getElementById('create-workflow-modal').classList.remove('hidden');
}

function hideCreateWorkflowModal() {
    document.getElementById('create-workflow-modal').classList.add('hidden');
}

function showCreateDocumentModal() {
    document.getElementById('create-document-modal').classList.remove('hidden');
}

function hideCreateDocumentModal() {
    document.getElementById('create-document-modal').classList.add('hidden');
}

function createWorkflow() {
    const form = document.getElementById('workflow-form');
    const formData = new FormData(form);
    
    // Implementation for creating workflow
    hideCreateWorkflowModal();
    app.showNotification('Workflow created successfully!', 'success');
}

function generateDocument() {
    const form = document.getElementById('document-form');
    const formData = new FormData(form);
    
    // Implementation for generating document
    hideCreateDocumentModal();
    app.showNotification('Document generation started!', 'success');
}

// Initialize the application
let app;
document.addEventListener('DOMContentLoaded', function() {
    app = new BusinessAgentsApp();
});





























