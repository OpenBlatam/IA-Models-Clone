/**
 * TypeScript Type Definitions
 * ===========================
 * Type definitions for the Character Clothing Changer AI frontend
 */

/**
 * API Response Types
 */
interface APIResponse<T = any> {
    success: boolean;
    data?: T;
    error?: string;
}

/**
 * Health Check Response
 */
interface HealthResponse {
    status: 'healthy' | 'unhealthy';
    model_initialized: boolean;
    using_deepseek_fallback: boolean;
    model_type: 'Flux2' | 'DeepSeek' | 'unknown';
}

/**
 * Model Info Response
 */
interface ModelInfoResponse {
    status: 'initialized' | 'not_initialized';
    primary_model: string;
    fallback_mode: boolean;
    device?: string;
    dtype?: string;
}

/**
 * Clothing Change Result
 */
interface ClothingChangeResult {
    clothing_description: string;
    character_name?: string;
    style?: string;
    changed: boolean;
    prompt_used?: string;
    negative_prompt_used?: string;
    image_base64?: string;
    image_url?: string;
    image_path?: string;
    saved_path?: string;
    saved: boolean;
    quality_metrics?: QualityMetrics;
}

/**
 * Quality Metrics
 */
interface QualityMetrics {
    similarity?: number;
    quality?: number;
    [key: string]: any;
}

/**
 * Gallery Item
 */
interface GalleryItem {
    id: number;
    image_base64?: string;
    image_url?: string;
    result_image?: string;
    clothing_description: string;
    character_name?: string;
    prompt_used?: string;
    negative_prompt_used?: string;
    quality_metrics?: QualityMetrics;
    saved_path?: string;
    timestamp: string;
    favorite?: boolean;
}

/**
 * History Item
 */
interface HistoryItem {
    id: number;
    timestamp: string;
    originalImage?: string;
    resultImage?: string;
    image_base64?: string;
    image_url?: string;
    clothing_description: string;
    clothingDescription?: string;
    character_name?: string;
    characterName?: string;
    prompt_used?: string;
    negative_prompt_used?: string;
    quality_metrics?: QualityMetrics;
    saved_path?: string;
}

/**
 * Form Data
 */
interface FormData {
    image: File;
    clothing_description: string;
    character_name?: string;
    prompt?: string;
    negative_prompt?: string;
    num_inference_steps?: number;
    guidance_scale?: number;
    strength?: number;
    save_tensor?: boolean;
}

/**
 * Validation Result
 */
interface ValidationResult {
    valid: boolean;
    errors: string[];
}

/**
 * Cache Item
 */
interface CacheItem<T = any> {
    value: T;
    expiry: number;
    createdAt: number;
}

/**
 * Log Entry
 */
interface LogEntry {
    timestamp: string;
    level: 'DEBUG' | 'INFO' | 'WARN' | 'ERROR';
    message: string;
    args?: any[];
}

/**
 * State
 */
interface AppState {
    currentResult: ClothingChangeResult | null;
    currentImage: string | null;
    currentTab: 'result' | 'comparison' | 'gallery' | 'history' | 'stats';
    theme: 'default' | 'dark' | 'purple' | 'blue';
    isProcessing: boolean;
    serverStatus: 'unknown' | 'healthy' | 'unhealthy' | 'loading';
    modelInfo: ModelInfoResponse | null;
}

/**
 * Event Bus Listener
 */
type EventListener = (...args: any[]) => void;

/**
 * State Change Listener
 */
type StateChangeListener<T = any> = (newValue: T, oldValue: T, key: string) => void;

/**
 * Module Exports
 */
declare const CONFIG: {
    API_BASE: string;
    STORAGE_KEYS: {
        HISTORY: string;
        GALLERY: string;
        THEME: string;
    };
    LIMITS: {
        MAX_HISTORY: number;
        MAX_GALLERY: number;
    };
    DEFAULT_VALUES: {
        NUM_STEPS: number;
        GUIDANCE_SCALE: number;
        STRENGTH: number;
        SAVE_TENSOR: string;
    };
};

declare const CONSTANTS: {
    API: {
        BASE_URL: string;
        ENDPOINTS: Record<string, string>;
        TIMEOUT: number;
        RETRY_ATTEMPTS: number;
        RETRY_DELAY: number;
    };
    STORAGE: Record<string, string>;
    LIMITS: Record<string, number>;
    DEFAULTS: Record<string, any>;
    IMAGE: Record<string, any>;
    VALIDATION: Record<string, string>;
    ERRORS: Record<string, string>;
    SUCCESS: Record<string, string>;
    STATUS: Record<string, string>;
    CACHE: Record<string, number>;
    LOGGER: Record<string, any>;
    THEMES: Record<string, string>;
    ANIMATION: Record<string, number>;
    POLLING: Record<string, number>;
};

declare const API: {
    checkHealth(): Promise<APIResponse<HealthResponse>>;
    getModelInfo(): Promise<APIResponse<ModelInfoResponse>>;
    initializeModel(): Promise<APIResponse<ModelInfoResponse>>;
    changeClothing(formData: FormData): Promise<APIResponse<ClothingChangeResult>>;
    listTensors(): Promise<APIResponse<any[]>>;
    getTensor(tensorId: string): Promise<Blob>;
    getImage(imageId: string): Promise<Blob>;
};

declare const UI: {
    updateStatus(status: string, text: string): void;
    toggleAdvanced(): void;
    switchTab(tabName: string): void;
    showLoading(message?: string): string;
    showError(message: string): string;
    showResult(data: ClothingChangeResult): string;
    toggleThemeMenu(): void;
    setTheme(theme: string): void;
};

declare const StateManager: {
    get<T = any>(key: string): T;
    set<T = any>(key: string, value: T): void;
    update(updates: Partial<AppState>): void;
    subscribe(key: string, callback: StateChangeListener): () => void;
    unsubscribe(key: string, callback: StateChangeListener): void;
    reset(): void;
    getState(): AppState;
};

declare const EventBus: {
    on(event: string, callback: EventListener, context?: any): () => void;
    once(event: string, callback: EventListener, context?: any): () => void;
    off(event: string, listenerId?: string | number): void;
    emit(event: string, ...args: any[]): void;
    getEvents(): string[];
    getListenerCount(event: string): number;
    clear(): void;
};

declare const Logger: {
    debug(message: string, ...args: any[]): void;
    info(message: string, ...args: any[]): void;
    warn(message: string, ...args: any[]): void;
    error(message: string, ...args: any[]): void;
    setLevel(level: string | number): void;
    getHistory(filterLevel?: string | number): LogEntry[];
    clearHistory(): void;
    exportHistory(): string;
};

declare const Cache: {
    set<T = any>(key: string, value: T, ttl?: number): void;
    get<T = any>(key: string): T | null;
    has(key: string): boolean;
    delete(key: string): boolean;
    clear(): void;
    clearExpired(): number;
    getStats(): {
        total: number;
        active: number;
        expired: number;
        size: number;
    };
    keys(): string[];
};

declare const FormDataBuilder: {
    build(formId?: string): FormData;
    validate(formId?: string): ValidationResult;
    reset(formId?: string): void;
};

declare const ErrorHandler: {
    handleApiError(error: any, context?: string): string;
    showError(message: string, context?: string): void;
    handleNetworkError(error: Error): string;
    handleValidationError(field: string, message: string): string;
    clearValidationErrors(): void;
    formatError(error: any): string;
};

declare const ItemRenderer: {
    renderGalleryItem(item: GalleryItem, index?: number): string;
    renderHistoryItem(item: HistoryItem, index?: number): string;
    viewItem(index: number): void;
    downloadItem(index: number): void;
    reuseItem(index: number): void;
    renderEmptyState(type?: 'gallery' | 'history'): string;
};

declare const ModalViewer: {
    show(item: GalleryItem | HistoryItem): void;
    close(): void;
    download(): void;
};

declare const Storage: {
    saveHistory(items: HistoryItem[]): void;
    getHistory(): HistoryItem[];
    saveGallery(items: GalleryItem[]): void;
    getGallery(): GalleryItem[];
    saveTheme(theme: string): void;
    getTheme(): string | null;
    saveFavorites(favorites: number[]): void;
    getFavorites(): number[];
};

declare const GalleryManager: {
    items: GalleryItem[];
    init(): void;
    add(data: ClothingChangeResult): void;
    load(): void;
    viewImage(index: number): void;
    toggleFavorite(itemId: number): void;
};

declare const HistoryManager: {
    items: HistoryItem[];
    init(): void;
    add(data: ClothingChangeResult): void;
    load(): void;
    loadItem(id: number): void;
    toggleFavorite(itemId: number): void;
};

declare const Form: {
    currentImage: string | null;
    init(): void;
    setupFileUpload(): void;
    setupDragAndDrop(): void;
    setupFormSubmit(): void;
    buildFormData(): FormData;
};

declare const Notifications: {
    success(message: string, duration?: number): void;
    error(message: string, duration?: number): void;
    warning(message: string, duration?: number): void;
    info(message: string, duration?: number): void;
};

declare const ProgressBar: {
    start(): void;
    stop(): void;
    update(percentage: number): void;
};

declare const Comparison: {
    update(before: string, after: string): void;
};

declare const ImageAnalyzer: {
    analyze(imageData: string): void;
};

declare const Stats: {
    getStats(): any;
    display(): string;
};

declare const Filters: {
    filterGallery(query: string): void;
    filterHistory(query: string): void;
    sortGallery(order: 'asc' | 'desc'): void;
};

declare const Favorites: {
    init(): void;
    toggle(itemId: number): void;
    getAll(): number[];
    isFavorite(itemId: number): boolean;
};

declare const Shortcuts: {
    init(): void;
};

declare const AppState: {
    currentResult: ClothingChangeResult | null;
};

