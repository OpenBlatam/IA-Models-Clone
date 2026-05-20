/**
 * Download Manager Module
 * =======================
 * Advanced file download with progress tracking
 */

const DownloadManager = {
    /**
     * Active downloads
     */
    downloads: new Map(),
    
    /**
     * Initialize download manager
     */
    init() {
        if (typeof Logger !== 'undefined') {
            Logger.info('Download manager initialized');
        }
    },
    
    /**
     * Download file
     */
    async download(url, options = {}) {
        const {
            filename = null,
            onProgress = null,
            onComplete = null,
            onError = null,
            headers = {},
            method = 'GET'
        } = options;
        
        const downloadId = `download_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        
        const downloadInfo = {
            id: downloadId,
            url,
            filename,
            status: 'downloading',
            progress: 0,
            loaded: 0,
            total: 0,
            error: null
        };
        
        this.downloads.set(downloadId, downloadInfo);
        
        try {
            const blob = await this.downloadFile(url, downloadInfo, headers, method, onProgress);
            
            // Determine filename
            let finalFilename = filename;
            if (!finalFilename) {
                finalFilename = this.extractFilename(url) || 'download';
            }
            
            // Trigger download
            this.triggerDownload(blob, finalFilename);
            
            downloadInfo.status = 'completed';
            downloadInfo.progress = 100;
            
            if (onComplete) {
                onComplete(downloadInfo, blob);
            }
            
            // Emit download complete event
            if (typeof EventBus !== 'undefined') {
                EventBus.emit('download:complete', downloadInfo);
            }
            
        } catch (error) {
            downloadInfo.status = 'error';
            downloadInfo.error = error;
            
            if (typeof Logger !== 'undefined') {
                Logger.error('Download error', error);
            }
            
            if (onError) {
                onError(error, downloadInfo);
            }
            
            // Emit download error event
            if (typeof EventBus !== 'undefined') {
                EventBus.emit('download:error', { downloadInfo, error });
            }
            
            throw error;
        }
        
        return downloadInfo;
    },
    
    /**
     * Download file with progress
     */
    async downloadFile(url, downloadInfo, headers, method, onProgress) {
        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            
            xhr.addEventListener('progress', (e) => {
                if (e.lengthComputable) {
                    downloadInfo.progress = (e.loaded / e.total) * 100;
                    downloadInfo.loaded = e.loaded;
                    downloadInfo.total = e.total;
                    
                    if (onProgress) {
                        onProgress(downloadInfo);
                    }
                    
                    // Emit progress event
                    if (typeof EventBus !== 'undefined') {
                        EventBus.emit('download:progress', downloadInfo);
                    }
                }
            });
            
            xhr.addEventListener('load', () => {
                if (xhr.status >= 200 && xhr.status < 300) {
                    resolve(xhr.response);
                } else {
                    reject(new Error(`Download failed: ${xhr.status} ${xhr.statusText}`));
                }
            });
            
            xhr.addEventListener('error', () => {
                reject(new Error('Download failed: Network error'));
            });
            
            xhr.addEventListener('abort', () => {
                reject(new Error('Download aborted'));
            });
            
            xhr.open(method, url);
            xhr.responseType = 'blob';
            
            // Set headers
            Object.keys(headers).forEach(key => {
                xhr.setRequestHeader(key, headers[key]);
            });
            
            xhr.send();
        });
    },
    
    /**
     * Trigger download
     */
    triggerDownload(blob, filename) {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    },
    
    /**
     * Extract filename from URL
     */
    extractFilename(url) {
        try {
            const urlObj = new URL(url);
            const pathname = urlObj.pathname;
            const filename = pathname.split('/').pop();
            return filename || null;
        } catch (error) {
            return null;
        }
    },
    
    /**
     * Download from base64
     */
    downloadFromBase64(base64, filename, mimeType = 'application/octet-stream') {
        const byteCharacters = atob(base64);
        const byteNumbers = new Array(byteCharacters.length);
        
        for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        
        const byteArray = new Uint8Array(byteNumbers);
        const blob = new Blob([byteArray], { type: mimeType });
        
        this.triggerDownload(blob, filename);
    },
    
    /**
     * Download from data URL
     */
    downloadFromDataURL(dataURL, filename) {
        const blob = this.dataURLToBlob(dataURL);
        this.triggerDownload(blob, filename);
    },
    
    /**
     * Convert data URL to blob
     */
    dataURLToBlob(dataURL) {
        const arr = dataURL.split(',');
        const mime = arr[0].match(/:(.*?);/)[1];
        const bstr = atob(arr[1]);
        let n = bstr.length;
        const u8arr = new Uint8Array(n);
        
        while (n--) {
            u8arr[n] = bstr.charCodeAt(n);
        }
        
        return new Blob([u8arr], { type: mime });
    },
    
    /**
     * Cancel download
     */
    cancel(downloadId) {
        const downloadInfo = this.downloads.get(downloadId);
        if (!downloadInfo) {
            return false;
        }
        
        downloadInfo.status = 'cancelled';
        this.downloads.delete(downloadId);
        
        // Emit cancel event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('download:cancelled', downloadInfo);
        }
        
        return true;
    },
    
    /**
     * Get download status
     */
    getStatus(downloadId) {
        return this.downloads.get(downloadId);
    },
    
    /**
     * Get all downloads
     */
    getAll() {
        return Array.from(this.downloads.values());
    }
};

// Auto-initialize
if (typeof window !== 'undefined') {
    DownloadManager.init();
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DownloadManager;
}

