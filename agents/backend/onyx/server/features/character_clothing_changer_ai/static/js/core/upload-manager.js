/**
 * Upload Manager Module
 * =====================
 * Advanced file upload with progress tracking and chunking
 */

const UploadManager = {
    /**
     * Active uploads
     */
    uploads: new Map(),
    
    /**
     * Default options
     */
    defaultOptions: {
        chunkSize: 1024 * 1024, // 1MB
        maxConcurrent: 3,
        retryAttempts: 3,
        retryDelay: 1000
    },
    
    /**
     * Initialize upload manager
     */
    init() {
        if (typeof Logger !== 'undefined') {
            Logger.info('Upload manager initialized');
        }
    },
    
    /**
     * Upload file
     */
    async upload(file, url, options = {}) {
        const {
            chunkSize = this.defaultOptions.chunkSize,
            onProgress = null,
            onComplete = null,
            onError = null,
            headers = {},
            method = 'POST',
            chunked = false
        } = options;
        
        const uploadId = `upload_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        
        const uploadInfo = {
            id: uploadId,
            file,
            url,
            status: 'uploading',
            progress: 0,
            loaded: 0,
            total: file.size,
            error: null
        };
        
        this.uploads.set(uploadId, uploadInfo);
        
        try {
            if (chunked && file.size > chunkSize) {
                await this.uploadChunked(file, url, uploadInfo, chunkSize, headers, method, onProgress);
            } else {
                await this.uploadSingle(file, url, uploadInfo, headers, method, onProgress);
            }
            
            uploadInfo.status = 'completed';
            uploadInfo.progress = 100;
            
            if (onComplete) {
                onComplete(uploadInfo);
            }
            
            // Emit upload complete event
            if (typeof EventBus !== 'undefined') {
                EventBus.emit('upload:complete', uploadInfo);
            }
            
        } catch (error) {
            uploadInfo.status = 'error';
            uploadInfo.error = error;
            
            if (typeof Logger !== 'undefined') {
                Logger.error('Upload error', error);
            }
            
            if (onError) {
                onError(error, uploadInfo);
            }
            
            // Emit upload error event
            if (typeof EventBus !== 'undefined') {
                EventBus.emit('upload:error', { uploadInfo, error });
            }
            
            throw error;
        }
        
        return uploadInfo;
    },
    
    /**
     * Upload single file
     */
    async uploadSingle(file, url, uploadInfo, headers, method, onProgress) {
        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            
            xhr.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable) {
                    uploadInfo.progress = (e.loaded / e.total) * 100;
                    uploadInfo.loaded = e.loaded;
                    
                    if (onProgress) {
                        onProgress(uploadInfo);
                    }
                    
                    // Emit progress event
                    if (typeof EventBus !== 'undefined') {
                        EventBus.emit('upload:progress', uploadInfo);
                    }
                }
            });
            
            xhr.addEventListener('load', () => {
                if (xhr.status >= 200 && xhr.status < 300) {
                    resolve(xhr.response);
                } else {
                    reject(new Error(`Upload failed: ${xhr.status} ${xhr.statusText}`));
                }
            });
            
            xhr.addEventListener('error', () => {
                reject(new Error('Upload failed: Network error'));
            });
            
            xhr.addEventListener('abort', () => {
                reject(new Error('Upload aborted'));
            });
            
            xhr.open(method, url);
            
            // Set headers
            Object.keys(headers).forEach(key => {
                xhr.setRequestHeader(key, headers[key]);
            });
            
            // Send file
            const formData = new FormData();
            formData.append('file', file);
            xhr.send(formData);
        });
    },
    
    /**
     * Upload chunked file
     */
    async uploadChunked(file, url, uploadInfo, chunkSize, headers, method, onProgress) {
        const totalChunks = Math.ceil(file.size / chunkSize);
        const chunks = [];
        
        // Create chunks
        for (let i = 0; i < totalChunks; i++) {
            const start = i * chunkSize;
            const end = Math.min(start + chunkSize, file.size);
            chunks.push({
                index: i,
                start,
                end,
                blob: file.slice(start, end)
            });
        }
        
        // Upload chunks
        for (const chunk of chunks) {
            const chunkFormData = new FormData();
            chunkFormData.append('chunk', chunk.blob);
            chunkFormData.append('chunkIndex', chunk.index);
            chunkFormData.append('totalChunks', totalChunks);
            chunkFormData.append('fileName', file.name);
            chunkFormData.append('fileSize', file.size);
            
            await this.uploadChunk(chunkFormData, url, uploadInfo, chunk, headers, method, onProgress);
        }
    },
    
    /**
     * Upload single chunk
     */
    async uploadChunk(chunkFormData, url, uploadInfo, chunk, headers, method, onProgress) {
        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            
            xhr.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable) {
                    const chunkProgress = (e.loaded / e.total) * 100;
                    const overallProgress = ((chunk.index * chunk.blob.size + e.loaded) / uploadInfo.total) * 100;
                    
                    uploadInfo.progress = overallProgress;
                    uploadInfo.loaded = chunk.index * chunk.blob.size + e.loaded;
                    
                    if (onProgress) {
                        onProgress(uploadInfo);
                    }
                }
            });
            
            xhr.addEventListener('load', () => {
                if (xhr.status >= 200 && xhr.status < 300) {
                    resolve(xhr.response);
                } else {
                    reject(new Error(`Chunk upload failed: ${xhr.status}`));
                }
            });
            
            xhr.addEventListener('error', () => {
                reject(new Error('Chunk upload failed: Network error'));
            });
            
            xhr.open(method, url);
            
            // Set headers
            Object.keys(headers).forEach(key => {
                xhr.setRequestHeader(key, headers[key]);
            });
            
            xhr.send(chunkFormData);
        });
    },
    
    /**
     * Cancel upload
     */
    cancel(uploadId) {
        const uploadInfo = this.uploads.get(uploadId);
        if (!uploadInfo) {
            return false;
        }
        
        uploadInfo.status = 'cancelled';
        this.uploads.delete(uploadId);
        
        // Emit cancel event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('upload:cancelled', uploadInfo);
        }
        
        return true;
    },
    
    /**
     * Get upload status
     */
    getStatus(uploadId) {
        return this.uploads.get(uploadId);
    },
    
    /**
     * Get all uploads
     */
    getAll() {
        return Array.from(this.uploads.values());
    }
};

// Auto-initialize
if (typeof window !== 'undefined') {
    UploadManager.init();
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = UploadManager;
}

