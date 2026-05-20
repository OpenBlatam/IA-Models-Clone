/**
 * Drag and Drop Manager Module
 * ============================
 * Advanced drag and drop functionality
 */

const DragDropManager = {
    /**
     * Active drag operations
     */
    activeDrags: new Map(),
    
    /**
     * Drop zones
     */
    dropZones: new Map(),
    
    /**
     * Initialize drag and drop manager
     */
    init() {
        if (typeof Logger !== 'undefined') {
            Logger.info('Drag and drop manager initialized');
        }
    },
    
    /**
     * Make element draggable
     */
    makeDraggable(element, options = {}) {
        const {
            data = {},
            onDragStart = null,
            onDrag = null,
            onDragEnd = null,
            handle = null,
            clone = false
        } = options;
        
        const dragElement = handle ? element.querySelector(handle) : element;
        
        if (!dragElement) {
            return false;
        }
        
        dragElement.draggable = true;
        dragElement.setAttribute('data-draggable', 'true');
        
        // Drag start
        dragElement.addEventListener('dragstart', (e) => {
            const dragId = `drag_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
            
            this.activeDrags.set(dragId, {
                element: clone ? element.cloneNode(true) : element,
                data,
                originalElement: element
            });
            
            e.dataTransfer.effectAllowed = 'move';
            e.dataTransfer.setData('text/plain', dragId);
            
            if (clone) {
                e.dataTransfer.setDragImage(element, 0, 0);
            }
            
            element.classList.add('dragging');
            
            if (onDragStart) {
                onDragStart(e, data);
            }
            
            // Emit drag start event
            if (typeof EventBus !== 'undefined') {
                EventBus.emit('drag:start', { dragId, element, data });
            }
        });
        
        // Drag
        dragElement.addEventListener('drag', (e) => {
            if (onDrag) {
                onDrag(e);
            }
        });
        
        // Drag end
        dragElement.addEventListener('dragend', (e) => {
            element.classList.remove('dragging');
            
            const dragId = e.dataTransfer.getData('text/plain');
            this.activeDrags.delete(dragId);
            
            if (onDragEnd) {
                onDragEnd(e);
            }
            
            // Emit drag end event
            if (typeof EventBus !== 'undefined') {
                EventBus.emit('drag:end', { dragId, element });
            }
        });
        
        return true;
    },
    
    /**
     * Make element a drop zone
     */
    makeDropZone(element, options = {}) {
        const {
            accept = null,
            onDragEnter = null,
            onDragOver = null,
            onDragLeave = null,
            onDrop = null,
            highlightClass = 'drag-over'
        } = options;
        
        const zoneId = `zone_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        
        this.dropZones.set(zoneId, {
            element,
            accept,
            highlightClass
        });
        
        // Drag enter
        element.addEventListener('dragenter', (e) => {
            e.preventDefault();
            e.stopPropagation();
            
            if (this.canDrop(e, accept)) {
                element.classList.add(highlightClass);
                
                if (onDragEnter) {
                    onDragEnter(e);
                }
            }
        });
        
        // Drag over
        element.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
            
            if (this.canDrop(e, accept)) {
                e.dataTransfer.dropEffect = 'move';
                
                if (onDragOver) {
                    onDragOver(e);
                }
            } else {
                e.dataTransfer.dropEffect = 'none';
            }
        });
        
        // Drag leave
        element.addEventListener('dragleave', (e) => {
            e.preventDefault();
            e.stopPropagation();
            
            element.classList.remove(highlightClass);
            
            if (onDragLeave) {
                onDragLeave(e);
            }
        });
        
        // Drop
        element.addEventListener('drop', (e) => {
            e.preventDefault();
            e.stopPropagation();
            
            element.classList.remove(highlightClass);
            
            if (this.canDrop(e, accept)) {
                const dragId = e.dataTransfer.getData('text/plain');
                const dragData = this.activeDrags.get(dragId);
                
                if (dragData) {
                    if (onDrop) {
                        onDrop(e, dragData.data, dragData.element);
                    }
                    
                    // Emit drop event
                    if (typeof EventBus !== 'undefined') {
                        EventBus.emit('drop:success', {
                            zoneId,
                            dragId,
                            data: dragData.data,
                            element: dragData.element
                        });
                    }
                }
            }
        });
        
        return zoneId;
    },
    
    /**
     * Check if can drop
     */
    canDrop(e, accept) {
        if (!accept) {
            return true;
        }
        
        const dragId = e.dataTransfer.getData('text/plain');
        const dragData = this.activeDrags.get(dragId);
        
        if (!dragData) {
            return false;
        }
        
        if (typeof accept === 'function') {
            return accept(dragData.data);
        }
        
        if (Array.isArray(accept)) {
            return accept.includes(dragData.data.type);
        }
        
        return true;
    },
    
    /**
     * Remove draggable
     */
    removeDraggable(element) {
        const dragElement = element.querySelector('[data-draggable="true"]') || element;
        dragElement.draggable = false;
        dragElement.removeAttribute('data-draggable');
    },
    
    /**
     * Remove drop zone
     */
    removeDropZone(zoneId) {
        this.dropZones.delete(zoneId);
    }
};

// Auto-initialize
if (typeof window !== 'undefined') {
    DragDropManager.init();
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DragDropManager;
}

