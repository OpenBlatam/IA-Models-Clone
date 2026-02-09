/**
 * Drag and Drop Handler Module
 * ============================
 * Handles drag and drop functionality
 */

const DragDropHandler = {
    /**
     * Setup drag and drop for element
     */
    setup(element, options = {}) {
        const {
            onDrop,
            onDragEnter,
            onDragLeave,
            dragOverClass = 'drag-over'
        } = options;
        
        // Prevent default behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            element.addEventListener(eventName, this.preventDefaults, false);
        });
        
        // Add drag over class
        ['dragenter', 'dragover'].forEach(eventName => {
            element.addEventListener(eventName, () => {
                element.classList.add(dragOverClass);
                if (onDragEnter) onDragEnter();
            }, false);
        });
        
        // Remove drag over class
        ['dragleave', 'drop'].forEach(eventName => {
            element.addEventListener(eventName, () => {
                element.classList.remove(dragOverClass);
                if (onDragLeave) onDragLeave();
            }, false);
        });
        
        // Handle drop
        element.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0 && onDrop) {
                onDrop(files, e);
            }
        }, false);
    },

    /**
     * Prevent default drag and drop behavior
     */
    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    },

    /**
     * Setup for file input
     */
    setupForFileInput(dropZone, fileInput, options = {}) {
        this.setup(dropZone, {
            ...options,
            onDrop: (files) => {
                if (files.length > 0) {
                    fileInput.files = files;
                    fileInput.dispatchEvent(new Event('change', { bubbles: true }));
                }
            }
        });
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DragDropHandler;
}

