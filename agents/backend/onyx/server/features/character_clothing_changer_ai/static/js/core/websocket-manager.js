/**
 * WebSocket Manager Module
 * =======================
 * Manages WebSocket connections with reconnection and message queuing
 */

const WebSocketManager = {
    /**
     * WebSocket connections
     */
    connections: new Map(),
    
    /**
     * Message queues
     */
    queues: new Map(),
    
    /**
     * Reconnection settings
     */
    reconnectSettings: {
        maxAttempts: 5,
        initialDelay: 1000,
        maxDelay: 30000,
        backoffMultiplier: 2
    },
    
    /**
     * Create WebSocket connection
     */
    create(name, url, options = {}) {
        const {
            protocols = [],
            autoReconnect = true,
            queueMessages = true,
            onOpen = null,
            onMessage = null,
            onError = null,
            onClose = null
        } = options;
        
        const connection = {
            name,
            url,
            ws: null,
            status: 'disconnected',
            reconnectAttempts: 0,
            reconnectTimer: null,
            messageQueue: [],
            listeners: new Set(),
            autoReconnect,
            queueMessages,
            onOpen,
            onMessage,
            onError,
            onClose
        };
        
        this.connections.set(name, connection);
        this.queues.set(name, []);
        
        // Connect
        this.connect(name);
        
        if (typeof Logger !== 'undefined') {
            Logger.info(`WebSocket connection created: ${name}`, { url });
        }
        
        return connection;
    },
    
    /**
     * Connect WebSocket
     */
    connect(name) {
        const connection = this.connections.get(name);
        if (!connection) {
            throw new Error(`WebSocket connection not found: ${name}`);
        }
        
        if (connection.status === 'connecting' || connection.status === 'connected') {
            return;
        }
        
        try {
            connection.status = 'connecting';
            connection.ws = new WebSocket(connection.url, connection.protocols);
            
            connection.ws.onopen = (event) => {
                connection.status = 'connected';
                connection.reconnectAttempts = 0;
                
                if (typeof Logger !== 'undefined') {
                    Logger.info(`WebSocket connected: ${name}`);
                }
                
                // Execute onOpen callback
                if (connection.onOpen) {
                    connection.onOpen(event);
                }
                
                // Notify listeners
                connection.listeners.forEach(listener => {
                    try {
                        listener({ type: 'open', event });
                    } catch (error) {
                        if (typeof Logger !== 'undefined') {
                            Logger.error('WebSocket listener error', error);
                        }
                    }
                });
                
                // Emit connection open event
                if (typeof EventBus !== 'undefined') {
                    EventBus.emit('websocket:open', { name, connection });
                }
                
                // Process queued messages
                this.processQueue(name);
            };
            
            connection.ws.onmessage = (event) => {
                if (typeof Logger !== 'undefined') {
                    Logger.debug(`WebSocket message received: ${name}`, event.data);
                }
                
                // Execute onMessage callback
                if (connection.onMessage) {
                    connection.onMessage(event);
                }
                
                // Notify listeners
                connection.listeners.forEach(listener => {
                    try {
                        listener({ type: 'message', event });
                    } catch (error) {
                        if (typeof Logger !== 'undefined') {
                            Logger.error('WebSocket listener error', error);
                        }
                    }
                });
                
                // Emit message event
                if (typeof EventBus !== 'undefined') {
                    EventBus.emit('websocket:message', { name, data: event.data });
                }
            };
            
            connection.ws.onerror = (event) => {
                if (typeof Logger !== 'undefined') {
                    Logger.error(`WebSocket error: ${name}`, event);
                }
                
                // Execute onError callback
                if (connection.onError) {
                    connection.onError(event);
                }
                
                // Emit error event
                if (typeof EventBus !== 'undefined') {
                    EventBus.emit('websocket:error', { name, event });
                }
            };
            
            connection.ws.onclose = (event) => {
                connection.status = 'disconnected';
                
                if (typeof Logger !== 'undefined') {
                    Logger.info(`WebSocket closed: ${name}`, { code: event.code, reason: event.reason });
                }
                
                // Execute onClose callback
                if (connection.onClose) {
                    connection.onClose(event);
                }
                
                // Emit close event
                if (typeof EventBus !== 'undefined') {
                    EventBus.emit('websocket:close', { name, event });
                }
                
                // Auto reconnect if enabled
                if (connection.autoReconnect && !event.wasClean) {
                    this.scheduleReconnect(name);
                }
            };
            
        } catch (error) {
            connection.status = 'error';
            
            if (typeof Logger !== 'undefined') {
                Logger.error(`WebSocket connection error: ${name}`, error);
            }
            
            if (connection.autoReconnect) {
                this.scheduleReconnect(name);
            }
        }
    },
    
    /**
     * Schedule reconnection
     */
    scheduleReconnect(name) {
        const connection = this.connections.get(name);
        if (!connection) {
            return;
        }
        
        if (connection.reconnectAttempts >= this.reconnectSettings.maxAttempts) {
            if (typeof Logger !== 'undefined') {
                Logger.error(`WebSocket max reconnection attempts reached: ${name}`);
            }
            return;
        }
        
        const delay = Math.min(
            this.reconnectSettings.initialDelay * Math.pow(this.reconnectSettings.backoffMultiplier, connection.reconnectAttempts),
            this.reconnectSettings.maxDelay
        );
        
        connection.reconnectAttempts++;
        
        if (typeof Logger !== 'undefined') {
            Logger.info(`WebSocket reconnecting: ${name} (attempt ${connection.reconnectAttempts})`, { delay });
        }
        
        connection.reconnectTimer = setTimeout(() => {
            this.connect(name);
        }, delay);
    },
    
    /**
     * Send message
     */
    send(name, message) {
        const connection = this.connections.get(name);
        if (!connection) {
            throw new Error(`WebSocket connection not found: ${name}`);
        }
        
        if (connection.status === 'connected' && connection.ws.readyState === WebSocket.OPEN) {
            connection.ws.send(typeof message === 'string' ? message : JSON.stringify(message));
            return true;
        }
        
        // Queue message if enabled
        if (connection.queueMessages) {
            this.queues.get(name).push(message);
            if (typeof Logger !== 'undefined') {
                Logger.debug(`WebSocket message queued: ${name}`);
            }
        }
        
        return false;
    },
    
    /**
     * Process message queue
     */
    processQueue(name) {
        const queue = this.queues.get(name);
        if (!queue || queue.length === 0) {
            return;
        }
        
        const connection = this.connections.get(name);
        if (connection.status !== 'connected') {
            return;
        }
        
        while (queue.length > 0) {
            const message = queue.shift();
            this.send(name, message);
        }
    },
    
    /**
     * Close connection
     */
    close(name) {
        const connection = this.connections.get(name);
        if (!connection) {
            return false;
        }
        
        connection.autoReconnect = false;
        
        if (connection.reconnectTimer) {
            clearTimeout(connection.reconnectTimer);
        }
        
        if (connection.ws) {
            connection.ws.close();
        }
        
        this.connections.delete(name);
        this.queues.delete(name);
        
        if (typeof Logger !== 'undefined') {
            Logger.info(`WebSocket connection closed: ${name}`);
        }
        
        return true;
    },
    
    /**
     * Subscribe to connection events
     */
    subscribe(name, listener) {
        const connection = this.connections.get(name);
        if (!connection) {
            throw new Error(`WebSocket connection not found: ${name}`);
        }
        
        connection.listeners.add(listener);
        
        // Return unsubscribe function
        return () => {
            connection.listeners.delete(listener);
        };
    },
    
    /**
     * Get connection status
     */
    getStatus(name) {
        const connection = this.connections.get(name);
        if (!connection) {
            return null;
        }
        
        return {
            name: connection.name,
            status: connection.status,
            url: connection.url,
            reconnectAttempts: connection.reconnectAttempts,
            queueLength: this.queues.get(name)?.length || 0
        };
    },
    
    /**
     * Get all connections
     */
    getAll() {
        return Array.from(this.connections.values());
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = WebSocketManager;
}

