/**
 * Stream Manager Module
 * ====================
 * Manages data streams and streaming operations
 */

const StreamManager = {
    /**
     * Active streams
     */
    streams: new Map(),
    
    /**
     * Stream processors
     */
    processors: new Map(),
    
    /**
     * Initialize stream manager
     */
    init() {
        if (typeof Logger !== 'undefined') {
            Logger.info('Stream manager initialized');
        }
    },
    
    /**
     * Create stream
     */
    create(name, options = {}) {
        const {
            bufferSize = 1024,
            onData = null,
            onEnd = null,
            onError = null
        } = options;
        
        const stream = {
            name,
            buffer: [],
            bufferSize,
            status: 'active',
            dataCount: 0,
            errorCount: 0,
            listeners: new Set(),
            onData,
            onEnd,
            onError,
            createdAt: Date.now()
        };
        
        this.streams.set(name, stream);
        
        if (typeof Logger !== 'undefined') {
            Logger.info(`Stream created: ${name}`);
        }
        
        return stream;
    },
    
    /**
     * Write data to stream
     */
    write(name, data) {
        const stream = this.streams.get(name);
        if (!stream) {
            throw new Error(`Stream not found: ${name}`);
        }
        
        if (stream.status !== 'active') {
            throw new Error(`Stream not active: ${name}`);
        }
        
        stream.buffer.push(data);
        stream.dataCount++;
        
        // Process buffer if full
        if (stream.buffer.length >= stream.bufferSize) {
            this.flush(name);
        }
        
        // Execute onData callback
        if (stream.onData) {
            stream.onData(data);
        }
        
        // Notify listeners
        stream.listeners.forEach(listener => {
            try {
                listener({ type: 'data', data });
            } catch (error) {
                if (typeof Logger !== 'undefined') {
                    Logger.error('Stream listener error', error);
                }
            }
        });
        
        // Emit stream data event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('stream:data', { name, data });
        }
    },
    
    /**
     * Flush stream buffer
     */
    flush(name) {
        const stream = this.streams.get(name);
        if (!stream || stream.buffer.length === 0) {
            return;
        }
        
        const data = stream.buffer.splice(0);
        
        // Emit flush event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('stream:flush', { name, data });
        }
    },
    
    /**
     * End stream
     */
    end(name) {
        const stream = this.streams.get(name);
        if (!stream) {
            return false;
        }
        
        // Flush remaining buffer
        this.flush(name);
        
        stream.status = 'ended';
        
        // Execute onEnd callback
        if (stream.onEnd) {
            stream.onEnd();
        }
        
        // Emit stream end event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('stream:end', { name });
        }
        
        // Cleanup after delay
        setTimeout(() => {
            this.streams.delete(name);
        }, 5000);
        
        return true;
    },
    
    /**
     * Subscribe to stream
     */
    subscribe(name, listener) {
        const stream = this.streams.get(name);
        if (!stream) {
            throw new Error(`Stream not found: ${name}`);
        }
        
        stream.listeners.add(listener);
        
        // Return unsubscribe function
        return () => {
            stream.listeners.delete(listener);
        };
    },
    
    /**
     * Create processor
     */
    createProcessor(name, processorFn) {
        this.processors.set(name, processorFn);
        
        if (typeof Logger !== 'undefined') {
            Logger.info(`Stream processor created: ${name}`);
        }
    },
    
    /**
     * Process stream with processor
     */
    async processStream(streamName, processorName) {
        const stream = this.streams.get(streamName);
        const processor = this.processors.get(processorName);
        
        if (!stream) {
            throw new Error(`Stream not found: ${streamName}`);
        }
        if (!processor) {
            throw new Error(`Processor not found: ${processorName}`);
        }
        
        const results = [];
        
        for (const data of stream.buffer) {
            try {
                const result = await processor(data);
                results.push(result);
            } catch (error) {
                stream.errorCount++;
                if (typeof Logger !== 'undefined') {
                    Logger.error('Stream processing error', error);
                }
            }
        }
        
        return results;
    },
    
    /**
     * Pipe stream
     */
    pipe(sourceName, targetName) {
        const source = this.streams.get(sourceName);
        const target = this.streams.get(targetName);
        
        if (!source || !target) {
            throw new Error('Source or target stream not found');
        }
        
        const unsubscribe = this.subscribe(sourceName, ({ data }) => {
            this.write(targetName, data);
        });
        
        return unsubscribe;
    },
    
    /**
     * Get stream status
     */
    getStatus(name) {
        const stream = this.streams.get(name);
        if (!stream) {
            return null;
        }
        
        return {
            name: stream.name,
            status: stream.status,
            bufferLength: stream.buffer.length,
            dataCount: stream.dataCount,
            errorCount: stream.errorCount
        };
    },
    
    /**
     * Get all streams
     */
    getAll() {
        return Array.from(this.streams.values());
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = StreamManager;
}

