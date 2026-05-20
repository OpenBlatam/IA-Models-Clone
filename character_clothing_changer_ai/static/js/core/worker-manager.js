/**
 * Worker Manager Module
 * ====================
 * Manages Web Workers for background processing
 */

const WorkerManager = {
    /**
     * Active workers
     */
    workers: new Map(),
    
    /**
     * Worker pools
     */
    pools: new Map(),
    
    /**
     * Initialize worker manager
     */
    init() {
        if (typeof Logger !== 'undefined') {
            Logger.info('Worker manager initialized');
        }
    },
    
    /**
     * Create worker
     */
    create(name, scriptUrl, options = {}) {
        const {
            type = 'classic',
            credentials = 'same-origin',
            onMessage = null,
            onError = null,
            onMessageError = null
        } = options;
        
        try {
            const worker = new Worker(scriptUrl, {
                type,
                credentials
            });
            
            const workerInfo = {
                name,
                worker,
                scriptUrl,
                status: 'running',
                messageCount: 0,
                errorCount: 0,
                createdAt: Date.now()
            };
            
            // Setup message handler
            worker.onmessage = (event) => {
                workerInfo.messageCount++;
                
                if (onMessage) {
                    onMessage(event);
                }
                
                // Emit worker message event
                if (typeof EventBus !== 'undefined') {
                    EventBus.emit('worker:message', { name, data: event.data });
                }
            };
            
            // Setup error handler
            worker.onerror = (error) => {
                workerInfo.errorCount++;
                workerInfo.status = 'error';
                
                if (typeof Logger !== 'undefined') {
                    Logger.error(`Worker error: ${name}`, error);
                }
                
                if (onError) {
                    onError(error);
                }
                
                // Emit worker error event
                if (typeof EventBus !== 'undefined') {
                    EventBus.emit('worker:error', { name, error });
                }
            };
            
            // Setup message error handler
            worker.onmessageerror = (error) => {
                if (typeof Logger !== 'undefined') {
                    Logger.error(`Worker message error: ${name}`, error);
                }
                
                if (onMessageError) {
                    onMessageError(error);
                }
            };
            
            this.workers.set(name, workerInfo);
            
            if (typeof Logger !== 'undefined') {
                Logger.info(`Worker created: ${name}`, { scriptUrl });
            }
            
            return worker;
        } catch (error) {
            if (typeof Logger !== 'undefined') {
                Logger.error(`Failed to create worker: ${name}`, error);
            }
            throw error;
        }
    },
    
    /**
     * Post message to worker
     */
    postMessage(name, message, transfer = []) {
        const workerInfo = this.workers.get(name);
        if (!workerInfo) {
            throw new Error(`Worker not found: ${name}`);
        }
        
        if (workerInfo.status !== 'running') {
            throw new Error(`Worker not running: ${name}`);
        }
        
        workerInfo.worker.postMessage(message, transfer);
    },
    
    /**
     * Terminate worker
     */
    terminate(name) {
        const workerInfo = this.workers.get(name);
        if (!workerInfo) {
            return false;
        }
        
        workerInfo.worker.terminate();
        workerInfo.status = 'terminated';
        this.workers.delete(name);
        
        if (typeof Logger !== 'undefined') {
            Logger.info(`Worker terminated: ${name}`);
        }
        
        return true;
    },
    
    /**
     * Create worker pool
     */
    createPool(name, scriptUrl, poolSize = 4) {
        const pool = {
            name,
            scriptUrl,
            workers: [],
            queue: [],
            active: 0
        };
        
        // Create workers
        for (let i = 0; i < poolSize; i++) {
            const workerName = `${name}_${i}`;
            const worker = this.create(workerName, scriptUrl, {
                onMessage: (event) => {
                    this.handlePoolMessage(pool, workerName, event);
                }
            });
            pool.workers.push(workerName);
        }
        
        this.pools.set(name, pool);
        
        if (typeof Logger !== 'undefined') {
            Logger.info(`Worker pool created: ${name}`, { size: poolSize });
        }
        
        return pool;
    },
    
    /**
     * Handle pool message
     */
    handlePoolMessage(pool, workerName, event) {
        pool.active--;
        this.processPoolQueue(pool);
    },
    
    /**
     * Process pool queue
     */
    processPoolQueue(pool) {
        if (pool.queue.length === 0 || pool.active >= pool.workers.length) {
            return;
        }
        
        const task = pool.queue.shift();
        const workerName = pool.workers[pool.active];
        
        pool.active++;
        this.postMessage(workerName, task.message, task.transfer || []);
        
        // Process next
        this.processPoolQueue(pool);
    },
    
    /**
     * Add task to pool
     */
    addTaskToPool(poolName, message, transfer = []) {
        const pool = this.pools.get(poolName);
        if (!pool) {
            throw new Error(`Worker pool not found: ${poolName}`);
        }
        
        pool.queue.push({ message, transfer });
        this.processPoolQueue(pool);
    },
    
    /**
     * Get worker status
     */
    getStatus(name) {
        const workerInfo = this.workers.get(name);
        if (!workerInfo) {
            return null;
        }
        
        return {
            name: workerInfo.name,
            status: workerInfo.status,
            messageCount: workerInfo.messageCount,
            errorCount: workerInfo.errorCount,
            createdAt: workerInfo.createdAt
        };
    },
    
    /**
     * Get all workers
     */
    getAll() {
        return Array.from(this.workers.values());
    },
    
    /**
     * Get pool status
     */
    getPoolStatus(poolName) {
        const pool = this.pools.get(poolName);
        if (!pool) {
            return null;
        }
        
        return {
            name: pool.name,
            size: pool.workers.length,
            active: pool.active,
            queueLength: pool.queue.length
        };
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = WorkerManager;
}

