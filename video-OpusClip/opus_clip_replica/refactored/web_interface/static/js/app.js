/**
 * Modern Web Interface JavaScript
 * Real-time video processing and job management
 */

class OpusClipApp {
    constructor() {
        this.ws = null;
        this.currentVideoId = null;
        this.jobs = new Map();
        this.videos = new Map();
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.connectWebSocket();
        this.loadInitialData();
    }

    setupEventListeners() {
        // Upload button
        document.getElementById('uploadBtn').addEventListener('click', () => {
            this.showUploadSection();
        });

        // File input
        document.getElementById('videoFile').addEventListener('change', (e) => {
            this.handleFileSelect(e);
        });

        // Analysis form
        document.getElementById('analysisForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.startAnalysis();
        });

        // Demo button
        document.getElementById('demoBtn').addEventListener('click', () => {
            this.showDemo();
        });
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.showNotification('Connected to server', 'success');
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(data);
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.showNotification('Disconnected from server', 'warning');
            // Reconnect after 3 seconds
            setTimeout(() => this.connectWebSocket(), 3000);
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.showNotification('Connection error', 'error');
        };
    }

    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'video_uploaded':
                this.handleVideoUploaded(data);
                break;
            case 'job_created':
                this.handleJobCreated(data);
                break;
            case 'job_update':
                this.handleJobUpdate(data);
                break;
            case 'job_completed':
                this.handleJobCompleted(data);
                break;
            case 'job_failed':
                this.handleJobFailed(data);
                break;
            case 'job_cancelled':
                this.handleJobCancelled(data);
                break;
            case 'video_deleted':
                this.handleVideoDeleted(data);
                break;
        }
    }

    async loadInitialData() {
        try {
            // Load jobs
            const jobsResponse = await fetch('/api/jobs');
            const jobsData = await jobsResponse.json();
            if (jobsData.success) {
                jobsData.jobs.forEach(job => {
                    this.jobs.set(job.id, job);
                });
                this.updateJobsDisplay();
            }

            // Load videos
            const videosResponse = await fetch('/api/videos');
            const videosData = await videosResponse.json();
            if (videosData.success) {
                videosData.videos.forEach(video => {
                    this.videos.set(video.id, video);
                });
            }
        } catch (error) {
            console.error('Failed to load initial data:', error);
        }
    }

    showUploadSection() {
        document.getElementById('uploadSection').style.display = 'block';
        document.getElementById('uploadSection').scrollIntoView({ behavior: 'smooth' });
    }

    handleFileSelect(event) {
        const file = event.target.files[0];
        if (!file) return;

        // Validate file type
        if (!file.type.startsWith('video/')) {
            this.showNotification('Please select a video file', 'error');
            return;
        }

        // Validate file size (max 500MB)
        if (file.size > 500 * 1024 * 1024) {
            this.showNotification('File size too large. Maximum 500MB allowed.', 'error');
            return;
        }

        this.uploadVideo(file);
    }

    async uploadVideo(file) {
        const formData = new FormData();
        formData.append('file', file);

        // Show progress
        document.getElementById('uploadProgress').style.display = 'block';
        document.getElementById('uploadStatus').textContent = 'Uploading...';

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                this.currentVideoId = result.video_id;
                this.videos.set(result.video_id, {
                    id: result.video_id,
                    filename: result.filename,
                    size: result.size,
                    uploaded_at: new Date().toISOString()
                });

                this.showNotification('Video uploaded successfully!', 'success');
                this.showAnalysisConfig();
                this.updateProgress(100);
            } else {
                throw new Error(result.error || 'Upload failed');
            }
        } catch (error) {
            console.error('Upload error:', error);
            this.showNotification(`Upload failed: ${error.message}`, 'error');
        } finally {
            setTimeout(() => {
                document.getElementById('uploadProgress').style.display = 'none';
            }, 2000);
        }
    }

    showAnalysisConfig() {
        document.getElementById('analysisConfig').style.display = 'block';
        document.getElementById('analysisConfig').scrollIntoView({ behavior: 'smooth' });
    }

    async startAnalysis() {
        if (!this.currentVideoId) {
            this.showNotification('Please upload a video first', 'error');
            return;
        }

        const formData = {
            video_id: this.currentVideoId,
            max_clips: parseInt(document.getElementById('maxClips').value),
            min_duration: parseFloat(document.getElementById('minDuration').value),
            max_duration: parseFloat(document.getElementById('maxDuration').value),
            priority: document.getElementById('priority').value
        };

        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });

            const result = await response.json();

            if (result.success) {
                this.showNotification('Analysis started successfully!', 'success');
                this.hideAnalysisConfig();
            } else {
                throw new Error(result.error || 'Analysis failed');
            }
        } catch (error) {
            console.error('Analysis error:', error);
            this.showNotification(`Analysis failed: ${error.message}`, 'error');
        }
    }

    hideAnalysisConfig() {
        document.getElementById('analysisConfig').style.display = 'none';
    }

    handleVideoUploaded(data) {
        this.videos.set(data.video_id, {
            id: data.video_id,
            filename: data.filename,
            size: data.size,
            uploaded_at: new Date().toISOString()
        });
    }

    handleJobCreated(data) {
        const job = {
            id: data.job_id,
            video_id: data.video_id,
            status: data.status,
            progress: 0,
            created_at: new Date().toISOString()
        };
        this.jobs.set(data.job_id, job);
        this.updateJobsDisplay();
        this.showNotification('New analysis job created', 'info');
    }

    handleJobUpdate(data) {
        const job = this.jobs.get(data.job_id);
        if (job) {
            job.status = data.status;
            job.progress = data.progress;
            if (data.current_step) {
                job.current_step = data.current_step;
            }
            this.updateJobsDisplay();
        }
    }

    handleJobCompleted(data) {
        const job = this.jobs.get(data.job_id);
        if (job) {
            job.status = 'completed';
            job.progress = 100;
            job.completed_at = new Date().toISOString();
            job.result = data.result;
            this.updateJobsDisplay();
            this.showNotification('Analysis completed successfully!', 'success');
        }
    }

    handleJobFailed(data) {
        const job = this.jobs.get(data.job_id);
        if (job) {
            job.status = 'failed';
            job.error = data.error;
            job.failed_at = new Date().toISOString();
            this.updateJobsDisplay();
            this.showNotification(`Analysis failed: ${data.error}`, 'error');
        }
    }

    handleJobCancelled(data) {
        const job = this.jobs.get(data.job_id);
        if (job) {
            job.status = 'cancelled';
            job.cancelled_at = new Date().toISOString();
            this.updateJobsDisplay();
            this.showNotification('Analysis cancelled', 'warning');
        }
    }

    handleVideoDeleted(data) {
        this.videos.delete(data.video_id);
    }

    updateJobsDisplay() {
        const jobsList = document.getElementById('jobsList');
        const jobs = Array.from(this.jobs.values()).reverse(); // Most recent first

        if (jobs.length === 0) {
            jobsList.innerHTML = '<p class="text-gray-500 text-center py-8">No jobs yet. Upload a video to get started!</p>';
            return;
        }

        jobsList.innerHTML = jobs.map(job => this.createJobCard(job)).join('');
    }

    createJobCard(job) {
        const statusColors = {
            pending: 'bg-yellow-100 text-yellow-800',
            running: 'bg-blue-100 text-blue-800',
            completed: 'bg-green-100 text-green-800',
            failed: 'bg-red-100 text-red-800',
            cancelled: 'bg-gray-100 text-gray-800'
        };

        const statusIcon = {
            pending: 'fas fa-clock',
            running: 'fas fa-spinner fa-spin',
            completed: 'fas fa-check-circle',
            failed: 'fas fa-exclamation-circle',
            cancelled: 'fas fa-times-circle'
        };

        const video = this.videos.get(job.video_id);
        const videoName = video ? video.filename : 'Unknown Video';

        return `
            <div class="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
                <div class="flex items-center justify-between">
                    <div class="flex-1">
                        <h3 class="text-lg font-medium text-gray-900">${videoName}</h3>
                        <p class="text-sm text-gray-500">Job ID: ${job.id}</p>
                        <div class="mt-2">
                            <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${statusColors[job.status]}">
                                <i class="${statusIcon[job.status]} mr-1"></i>
                                ${job.status.charAt(0).toUpperCase() + job.status.slice(1)}
                            </span>
                        </div>
                        ${job.current_step ? `<p class="text-sm text-gray-600 mt-1">${job.current_step}</p>` : ''}
                    </div>
                    <div class="flex items-center space-x-2">
                        ${job.status === 'running' || job.status === 'pending' ? `
                            <div class="w-32">
                                <div class="bg-gray-200 rounded-full h-2">
                                    <div class="bg-blue-600 h-2 rounded-full transition-all duration-300" style="width: ${job.progress}%"></div>
                                </div>
                                <p class="text-xs text-gray-500 mt-1">${job.progress}%</p>
                            </div>
                        ` : ''}
                        ${job.status === 'running' || job.status === 'pending' ? `
                            <button onclick="app.cancelJob('${job.id}')" class="text-red-600 hover:text-red-800">
                                <i class="fas fa-times"></i>
                            </button>
                        ` : ''}
                    </div>
                </div>
                ${job.result ? `
                    <div class="mt-4 p-3 bg-gray-50 rounded-lg">
                        <h4 class="text-sm font-medium text-gray-900 mb-2">Analysis Results</h4>
                        <p class="text-sm text-gray-600">Found ${job.result.total_segments} engaging segments</p>
                        <div class="mt-2 space-y-1">
                            ${job.result.segments.slice(0, 3).map(segment => `
                                <div class="text-xs text-gray-500">
                                    ${segment.title}: ${segment.start_time}s - ${segment.end_time}s (Score: ${(segment.engagement_score * 100).toFixed(1)}%)
                                </div>
                            `).join('')}
                            ${job.result.segments.length > 3 ? `<div class="text-xs text-gray-500">... and ${job.result.segments.length - 3} more</div>` : ''}
                        </div>
                    </div>
                ` : ''}
            </div>
        `;
    }

    async cancelJob(jobId) {
        try {
            const response = await fetch(`/api/jobs/${jobId}`, {
                method: 'DELETE'
            });

            const result = await response.json();

            if (result.success) {
                this.showNotification('Job cancelled successfully', 'info');
            } else {
                throw new Error(result.error || 'Failed to cancel job');
            }
        } catch (error) {
            console.error('Cancel job error:', error);
            this.showNotification(`Failed to cancel job: ${error.message}`, 'error');
        }
    }

    showDemo() {
        this.showNotification('Demo feature coming soon!', 'info');
    }

    updateProgress(percentage) {
        const progressBar = document.getElementById('progressBar');
        const uploadStatus = document.getElementById('uploadStatus');
        
        if (progressBar) {
            progressBar.style.width = `${percentage}%`;
        }
        
        if (uploadStatus) {
            uploadStatus.textContent = percentage === 100 ? 'Upload complete!' : `Uploading... ${percentage}%`;
        }
    }

    showNotification(message, type = 'info') {
        const notifications = document.getElementById('notifications');
        const notification = document.createElement('div');
        
        const typeColors = {
            success: 'bg-green-500',
            error: 'bg-red-500',
            warning: 'bg-yellow-500',
            info: 'bg-blue-500'
        };

        const typeIcons = {
            success: 'fas fa-check-circle',
            error: 'fas fa-exclamation-circle',
            warning: 'fas fa-exclamation-triangle',
            info: 'fas fa-info-circle'
        };

        notification.className = `${typeColors[type]} text-white px-4 py-2 rounded-lg shadow-lg flex items-center space-x-2 max-w-sm`;
        notification.innerHTML = `
            <i class="${typeIcons[type]}"></i>
            <span>${message}</span>
            <button onclick="this.parentElement.remove()" class="ml-2 text-white hover:text-gray-200">
                <i class="fas fa-times"></i>
            </button>
        `;

        notifications.appendChild(notification);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 5000);
    }
}

// Initialize app when DOM is loaded
function initializeApp() {
    window.app = new OpusClipApp();
}

// Handle WebSocket messages
function handleWebSocketMessage(data) {
    if (window.app) {
        window.app.handleWebSocketMessage(data);
    }
}


