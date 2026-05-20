/**
 * Queue Manager Module
 * ====================
 * Manages task queues with priority and concurrency control
 */

const QueueManager = {
    /**
     * Queues storage
     */
    queues: new Map(),
    
    /**
     * Create queue
     */
    create(name, options = {}) {
        const {
            concurrency = 1,
            priority = false,
            autoStart = true
        } = options;
        
        const queue = {
            name,
            tasks: [],
            running: 0,
            concurrency,
            priority,
            autoStart,
            paused: false,
            
            /**
             * Add task
             */
            add(task, priority = 0) {
                const taskItem = {
                    id: `task_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                    task,
                    priority,
                    addedAt: Date.now()
                };
                
                if (this.priority) {
                    // Insert in priority order
                    const index = this.tasks.findIndex(t => t.priority < priority);
                    if (index === -1) {
                        this.tasks.push(taskItem);
                    } else {
                        this.tasks.splice(index, 0, taskItem);
                    }
                } else {
                    this.tasks.push(taskItem);
                }
                
                if (this.autoStart && !this.paused) {
                    this.process();
                }
                
                return taskItem.id;
            },
            
            /**
             * Process queue
             */
            async process() {
                if (this.paused || this.running >= this.concurrency) {
                    return;
                }
                
                if (this.tasks.length === 0) {
                    return;
                }
                
                const taskItem = this.tasks.shift();
                this.running++;
                
                try {
                    const result = await taskItem.task();
                    
                    // Emit task complete event
                    if (typeof EventBus !== 'undefined') {
                        EventBus.emit('queue:task_complete', {
                            queue: this.name,
                            taskId: taskItem.id,
                            result
                        });
                    }
                } catch (error) {
                    // Emit task error event
                    if (typeof EventBus !== 'undefined') {
                        EventBus.emit('queue:task_error', {
                            queue: this.name,
                            taskId: taskItem.id,
                            error
                        });
                    }
                    
                    if (typeof Logger !== 'undefined') {
                        Logger.error(`Queue task error in ${this.name}`, error);
                    }
                } finally {
                    this.running--;
                    
                    // Process next task
                    if (this.tasks.length > 0 && !this.paused) {
                        this.process();
                    }
                }
            },
            
            /**
             * Pause queue
             */
            pause() {
                this.paused = true;
            },
            
            /**
             * Resume queue
             */
            resume() {
                this.paused = false;
                this.process();
            },
            
            /**
             * Clear queue
             */
            clear() {
                this.tasks = [];
            },
            
            /**
             * Get queue status
             */
            getStatus() {
                return {
                    name: this.name,
                    pending: this.tasks.length,
                    running: this.running,
                    paused: this.paused,
                    concurrency: this.concurrency
                };
            }
        };
        
        this.queues.set(name, queue);
        
        if (typeof Logger !== 'undefined') {
            Logger.info(`Queue created: ${name}`, { concurrency, priority });
        }
        
        return queue;
    },
    
    /**
     * Get queue
     */
    get(name) {
        return this.queues.get(name);
    },
    
    /**
     * Remove queue
     */
    remove(name) {
        const queue = this.queues.get(name);
        if (queue) {
            queue.clear();
            this.queues.delete(name);
            
            if (typeof Logger !== 'undefined') {
                Logger.info(`Queue removed: ${name}`);
            }
        }
    },
    
    /**
     * Get all queues
     */
    getAll() {
        return Array.from(this.queues.values());
    },
    
    /**
     * Get queue status
     */
    getStatus(name) {
        const queue = this.queues.get(name);
        return queue ? queue.getStatus() : null;
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = QueueManager;
}

