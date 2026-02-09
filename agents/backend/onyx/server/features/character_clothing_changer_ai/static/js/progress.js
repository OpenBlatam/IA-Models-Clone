/**
 * Progress Bar Module
 * ===================
 * Handles progress bar display with enhanced features
 */

const ProgressBar = {
    interval: null,
    currentProgress: 0,
    targetProgress: 0,

    /**
     * Start progress bar
     */
    start(initialProgress = 0) {
        const progressBar = document.getElementById('progressBar');
        if (!progressBar) return;

        this.currentProgress = initialProgress;
        this.targetProgress = 90; // Don't go to 100% until explicitly stopped
        
        this.updateDisplay(progressBar);
        
        this.interval = setInterval(() => {
            // Smooth progress increment
            const increment = Math.random() * 15;
            this.currentProgress = Math.min(
                this.currentProgress + increment,
                this.targetProgress
            );
            this.updateDisplay(progressBar);
        }, 500);
    },

    /**
     * Update progress bar display
     */
    updateDisplay(progressBar) {
        if (!progressBar) return;
        progressBar.style.width = this.currentProgress + '%';
        progressBar.textContent = Math.round(this.currentProgress) + '%';
    },

    /**
     * Set progress value
     */
    setProgress(value) {
        const progressBar = document.getElementById('progressBar');
        if (!progressBar) return;
        
        this.currentProgress = Math.max(0, Math.min(100, value));
        this.targetProgress = this.currentProgress;
        this.updateDisplay(progressBar);
    },

    /**
     * Stop progress bar
     */
    stop() {
        if (this.interval) {
            clearInterval(this.interval);
            this.interval = null;
        }

        const progressBar = document.getElementById('progressBar');
        if (progressBar) {
            this.currentProgress = 100;
            this.targetProgress = 100;
            this.updateDisplay(progressBar);
        }
    },

    /**
     * Reset progress bar
     */
    reset() {
        this.stop();
        this.currentProgress = 0;
        this.targetProgress = 0;
        const progressBar = document.getElementById('progressBar');
        if (progressBar) {
            this.updateDisplay(progressBar);
        }
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ProgressBar;
}


