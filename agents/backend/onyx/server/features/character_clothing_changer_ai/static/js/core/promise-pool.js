/**
 * Promise Pool Module
 * ===================
 * Manages concurrent promise execution with pool size control
 */

const PromisePool = {
    /**
     * Execute promises with concurrency limit
     */
    async execute(tasks, concurrency = 5) {
        if (!Array.isArray(tasks) || tasks.length === 0) {
            return [];
        }
        
        const results = [];
        const executing = [];
        let index = 0;
        
        const executeNext = async () => {
            if (index >= tasks.length) {
                return;
            }
            
            const taskIndex = index++;
            const task = tasks[taskIndex];
            
            const promise = Promise.resolve(task())
                .then(result => {
                    results[taskIndex] = { success: true, result };
                    return result;
                })
                .catch(error => {
                    results[taskIndex] = { success: false, error };
                    throw error;
                })
                .finally(() => {
                    // Remove from executing
                    const execIndex = executing.indexOf(promise);
                    if (execIndex > -1) {
                        executing.splice(execIndex, 1);
                    }
                    
                    // Execute next
                    executeNext();
                });
            
            executing.push(promise);
            
            // Start next if under concurrency limit
            if (executing.length < concurrency) {
                executeNext();
            }
        };
        
        // Start initial batch
        const initialBatch = Math.min(concurrency, tasks.length);
        for (let i = 0; i < initialBatch; i++) {
            executeNext();
        }
        
        // Wait for all to complete
        await Promise.all(executing);
        
        return results;
    },
    
    /**
     * Execute with retry
     */
    async executeWithRetry(tasks, concurrency = 5, maxRetries = 3) {
        const results = [];
        
        for (let i = 0; i < tasks.length; i += concurrency) {
            const batch = tasks.slice(i, i + concurrency);
            
            const batchResults = await Promise.all(
                batch.map(async (task, batchIndex) => {
                    let lastError;
                    
                    for (let attempt = 0; attempt <= maxRetries; attempt++) {
                        try {
                            const result = await task();
                            return { success: true, result, attempts: attempt + 1 };
                        } catch (error) {
                            lastError = error;
                            
                            if (attempt < maxRetries) {
                                // Wait before retry (exponential backoff)
                                await new Promise(resolve => 
                                    setTimeout(resolve, Math.pow(2, attempt) * 1000)
                                );
                            }
                        }
                    }
                    
                    return { success: false, error: lastError, attempts: maxRetries + 1 };
                })
            );
            
            results.push(...batchResults);
        }
        
        return results;
    },
    
    /**
     * Execute with timeout
     */
    async executeWithTimeout(tasks, concurrency = 5, timeout = 5000) {
        const results = [];
        
        for (let i = 0; i < tasks.length; i += concurrency) {
            const batch = tasks.slice(i, i + concurrency);
            
            const batchResults = await Promise.allSettled(
                batch.map(task => {
                    return Promise.race([
                        Promise.resolve(task()),
                        new Promise((_, reject) => 
                            setTimeout(() => reject(new Error('Timeout')), timeout)
                        )
                    ]);
                })
            );
            
            results.push(...batchResults.map((result, index) => {
                if (result.status === 'fulfilled') {
                    return { success: true, result: result.value };
                } else {
                    return { success: false, error: result.reason };
                }
            }));
        }
        
        return results;
    },
    
    /**
     * Execute with progress callback
     */
    async executeWithProgress(tasks, concurrency = 5, onProgress = null) {
        const results = [];
        let completed = 0;
        const total = tasks.length;
        
        const updateProgress = () => {
            if (onProgress) {
                onProgress({
                    completed,
                    total,
                    percentage: (completed / total) * 100
                });
            }
        };
        
        for (let i = 0; i < tasks.length; i += concurrency) {
            const batch = tasks.slice(i, i + concurrency);
            
            const batchResults = await Promise.allSettled(
                batch.map(async task => {
                    try {
                        const result = await task();
                        completed++;
                        updateProgress();
                        return { success: true, result };
                    } catch (error) {
                        completed++;
                        updateProgress();
                        return { success: false, error };
                    }
                })
            );
            
            results.push(...batchResults.map(result => {
                if (result.status === 'fulfilled') {
                    return result.value;
                } else {
                    return { success: false, error: result.reason };
                }
            }));
        }
        
        return results;
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PromisePool;
}

