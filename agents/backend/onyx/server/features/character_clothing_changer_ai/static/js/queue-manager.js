/**
 * Queue Manager Module
 * ====================
 * Task queue management with priority and retry
 */

const QueueManager = {
    /**
     * Task queues by priority
     */
    queues: {
        high: [],
        normal: [],
        low: []
    },
    
    /**
     * Processing state
     */
    processing: false,
    
    /**
     * Max concurrent tasks
     */
    maxConcurrent: 3,
    
    /**
     * Active tasks
     */
    activeTasks: [],
    
    /**
     * Add task to queue
     */
    enqueue(task, priority = 'normal') {
        const queueTask = {
            id: Date.now() + Math.random(),
            task,
            priority,
            retries: 0,
            maxRetries: task.maxRetries || 3,
            createdAt: Date.now()
        };
        
        this.queues[priority].push(queueTask);
        
        // Emit task queued event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('queue:task:queued', queueTask);
        }
        
        // Start processing if not already
        if (!this.processing) {
            this.process();
        }
        
        return queueTask.id;
    },
    
    /**
     * Process queue
     */
    async process() {
        if (this.processing) return;
        
        this.processing = true;
        
        while (this.hasTasks() || this.activeTasks.length > 0) {
            // Wait if at max concurrent
            while (this.activeTasks.length >= this.maxConcurrent) {
                await this.waitForTask();
            }
            
            // Get next task
            const task = this.getNextTask();
            if (!task) {
                await new Promise(resolve => setTimeout(resolve, 100));
                continue;
            }
            
            // Execute task
            this.executeTask(task);
        }
        
        this.processing = false;
        
        // Emit queue empty event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('queue:empty');
        }
    },
    
    /**
     * Get next task
     */
    getNextTask() {
        // Priority order: high -> normal -> low
        for (const priority of ['high', 'normal', 'low']) {
            if (this.queues[priority].length > 0) {
                return this.queues[priority].shift();
            }
        }
        return null;
    },
    
    /**
     * Execute task
     */
    async executeTask(queueTask) {
        this.activeTasks.push(queueTask);
        
        try {
            const result = await queueTask.task();
            
            // Remove from active
            this.activeTasks = this.activeTasks.filter(t => t.id !== queueTask.id);
            
            // Emit task completed event
            if (typeof EventBus !== 'undefined') {
                EventBus.emit('queue:task:completed', { id: queueTask.id, result });
            }
            
            return result;
        } catch (error) {
            // Retry logic
            if (queueTask.retries < queueTask.maxRetries) {
                queueTask.retries++;
                
                // Re-queue with delay
                setTimeout(() => {
                    this.queues[queueTask.priority].push(queueTask);
                }, 1000 * queueTask.retries); // Exponential backoff
                
                if (typeof Logger !== 'undefined') {
                    Logger.warn(`Task ${queueTask.id} failed, retrying (${queueTask.retries}/${queueTask.maxRetries})`);
                }
            } else {
                // Max retries reached
                this.activeTasks = this.activeTasks.filter(t => t.id !== queueTask.id);
                
                if (typeof EventBus !== 'undefined') {
                    EventBus.emit('queue:task:failed', { id: queueTask.id, error });
                }
                
                if (typeof ErrorHandler !== 'undefined') {
                    ErrorHandler.handle(error, { context: 'queue task', taskId: queueTask.id });
                }
            }
        }
    },
    
    /**
     * Wait for a task to complete
     */
    async waitForTask() {
        return new Promise(resolve => {
            const check = () => {
                if (this.activeTasks.length < this.maxConcurrent) {
                    resolve();
                } else {
                    setTimeout(check, 100);
                }
            };
            check();
        });
    },
    
    /**
     * Check if has tasks
     */
    hasTasks() {
        return this.queues.high.length > 0 ||
               this.queues.normal.length > 0 ||
               this.queues.low.length > 0;
    },
    
    /**
     * Get queue status
     */
    getStatus() {
        return {
            high: this.queues.high.length,
            normal: this.queues.normal.length,
            low: this.queues.low.length,
            active: this.activeTasks.length,
            processing: this.processing
        };
    },
    
    /**
     * Clear queue
     */
    clear(priority = null) {
        if (priority) {
            this.queues[priority] = [];
        } else {
            this.queues = { high: [], normal: [], low: [] };
        }
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = QueueManager;
}

