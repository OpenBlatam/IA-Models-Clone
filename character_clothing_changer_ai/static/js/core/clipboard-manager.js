/**
 * Clipboard Manager Module
 * ========================
 * Advanced clipboard operations
 */

const ClipboardManager = {
    /**
     * Initialize clipboard manager
     */
    init() {
        if (typeof Logger !== 'undefined') {
            Logger.info('Clipboard manager initialized');
        }
    },
    
    /**
     * Copy text to clipboard
     */
    async copyText(text) {
        try {
            if (navigator.clipboard && navigator.clipboard.writeText) {
                await navigator.clipboard.writeText(text);
                
                if (typeof Logger !== 'undefined') {
                    Logger.debug('Text copied to clipboard');
                }
                
                // Emit copy event
                if (typeof EventBus !== 'undefined') {
                    EventBus.emit('clipboard:copy', { type: 'text', data: text });
                }
                
                return true;
            } else {
                // Fallback for older browsers
                return this.copyTextFallback(text);
            }
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.error('Failed to copy text', error);
            }
            return false;
        }
    },
    
    /**
     * Copy text fallback
     */
    copyTextFallback(text) {
        const textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.select();
        
        try {
            const successful = document.execCommand('copy');
            document.body.removeChild(textarea);
            return successful;
        } catch (error) {
            document.body.removeChild(textarea);
            return false;
        }
    },
    
    /**
     * Paste text from clipboard
     */
    async pasteText() {
        try {
            if (navigator.clipboard && navigator.clipboard.readText) {
                const text = await navigator.clipboard.readText();
                
                // Emit paste event
                if (typeof EventBus !== 'undefined') {
                    EventBus.emit('clipboard:paste', { type: 'text', data: text });
                }
                
                return text;
            } else {
                throw new Error('Clipboard API not available');
            }
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.error('Failed to paste text', error);
            }
            throw error;
        }
    },
    
    /**
     * Copy image to clipboard
     */
    async copyImage(imageBlob) {
        try {
            if (navigator.clipboard && navigator.clipboard.write) {
                const clipboardItem = new ClipboardItem({
                    [imageBlob.type]: imageBlob
                });
                
                await navigator.clipboard.write([clipboardItem]);
                
                if (typeof Logger !== 'undefined') {
                    Logger.debug('Image copied to clipboard');
                }
                
                // Emit copy event
                if (typeof EventBus !== 'undefined') {
                    EventBus.emit('clipboard:copy', { type: 'image', data: imageBlob });
                }
                
                return true;
            } else {
                throw new Error('Clipboard API not available');
            }
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.error('Failed to copy image', error);
            }
            return false;
        }
    },
    
    /**
     * Copy image from canvas
     */
    async copyImageFromCanvas(canvas) {
        return new Promise((resolve, reject) => {
            canvas.toBlob(async (blob) => {
                if (blob) {
                    const success = await this.copyImage(blob);
                    if (success) {
                        resolve(true);
                    } else {
                        reject(new Error('Failed to copy image'));
                    }
                } else {
                    reject(new Error('Failed to create blob from canvas'));
                }
            });
        });
    },
    
    /**
     * Copy image from URL
     */
    async copyImageFromURL(url) {
        try {
            const response = await fetch(url);
            const blob = await response.blob();
            return await this.copyImage(blob);
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.error('Failed to copy image from URL', error);
            }
            return false;
        }
    },
    
    /**
     * Check clipboard permissions
     */
    async checkPermissions() {
        try {
            if (navigator.permissions && navigator.permissions.query) {
                const result = await navigator.permissions.query({ name: 'clipboard-read' });
                return {
                    read: result.state,
                    write: 'granted' // Write is usually granted
                };
            }
            return { read: 'unknown', write: 'unknown' };
        } catch (error) {
            return { read: 'unknown', write: 'unknown' };
        }
    },
    
    /**
     * Request clipboard permissions
     */
    async requestPermissions() {
        try {
            if (navigator.permissions && navigator.permissions.query) {
                const result = await navigator.permissions.query({ name: 'clipboard-read' });
                return result.state === 'granted' || result.state === 'prompt';
            }
            return true; // Assume granted if API not available
        } catch (error) {
            return false;
        }
    }
};

// Auto-initialize
if (typeof window !== 'undefined') {
    ClipboardManager.init();
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ClipboardManager;
}

