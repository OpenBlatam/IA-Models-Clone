/**
 * Form Submission Handler Module
 * ==============================
 * Handles form submission logic and state management
 */

const FormSubmissionHandler = {
    /**
     * Prepare form for submission
     */
    prepareSubmission(submitBtn, resultsElement, resultContentElement) {
        submitBtn.disabled = true;
        submitBtn.textContent = '⏳ Procesando...';
        if (resultsElement) {
            resultsElement.classList.add('show');
        }
        if (resultContentElement && typeof UI !== 'undefined') {
            resultContentElement.innerHTML = UI.showLoading('Generando nueva ropa... Esto puede tomar varios minutos.');
        }
        
        const formStartTime = Date.now();
        if (typeof StateManager !== 'undefined') {
            StateManager.set('isProcessing', true);
            StateManager.set('formStartTime', formStartTime);
        }
        
        // Track form submission start
        if (typeof AdvancedAnalytics !== 'undefined') {
            AdvancedAnalytics.trackUserAction('form_submit_start', {
                form: 'clothing_change'
            });
        }
        
        // Emit form submitted event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('form:submitted');
        }
        
        if (typeof ProgressBar !== 'undefined') {
            ProgressBar.start();
        }
        
        return formStartTime;
    },

    /**
     * Handle successful submission
     */
    handleSuccess(data, currentImage, submitBtn, resultContentElement) {
        // Update state
        if (typeof StateManager !== 'undefined') {
            StateManager.set('currentResult', data);
            StateManager.set('isProcessing', false);
        } else if (typeof AppState !== 'undefined') {
            AppState.currentResult = data;
        }
        
        // Save to history and gallery
        if (typeof HistoryManager !== 'undefined') {
            HistoryManager.add(data);
        }
        if (typeof GalleryManager !== 'undefined') {
            GalleryManager.add(data);
        }
        
        // Emit form completed event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('form:completed', data);
        }
        
        // Show result
        if (resultContentElement && typeof UI !== 'undefined') {
            resultContentElement.innerHTML = UI.showResult(data);
        }
        
        // Update comparison
        if (typeof Comparison !== 'undefined' && currentImage) {
            const resultImage = data.image_base64 || data.image_url || data.result_image;
            Comparison.update(currentImage, resultImage);
        }
        
        // Refresh views
        if (typeof GalleryManager !== 'undefined') {
            GalleryManager.load();
        }
        if (typeof HistoryManager !== 'undefined') {
            HistoryManager.load();
        }
        if (typeof UI !== 'undefined') {
            UI.switchTab('result');
        }
        
        // Show success notification
        if (typeof Notifications !== 'undefined') {
            Notifications.success('¡Ropa cambiada exitosamente!');
        }
        
        // Track successful form submission
        this.trackSuccess();
        
        // Reset button
        this.resetButton(submitBtn);
    },

    /**
     * Handle submission error
     */
    handleError(error, submitBtn, resultContentElement) {
        // Update state on error
        if (typeof StateManager !== 'undefined') {
            StateManager.set('isProcessing', false);
        }
        
        // Emit form error event
        if (typeof EventBus !== 'undefined') {
            EventBus.emit('form:error', error);
        }
        
        // Handle error display
        this.displayError(error, resultContentElement);
        
        // Show error notification
        if (typeof Notifications !== 'undefined') {
            Notifications.error('Error al procesar la solicitud');
        }
        
        // Track failed form submission
        this.trackFailure(error);
        
        // Reset button
        this.resetButton(submitBtn);
    },

    /**
     * Display error
     */
    displayError(error, resultContentElement) {
        if (!resultContentElement) return;
        
        const errorMsg = typeof ErrorHandler !== 'undefined' 
            ? ErrorHandler.formatError(error) 
            : (error.message || error.toString());
        
        // Check if error is about Flux2 but fallback might work
        if (errorMsg.includes('Failed to build Flux2 model') || 
            errorMsg.includes('Cannot load model black-forest-labs/flux2-dev')) {
            // Try to check if DeepSeek fallback is available
            if (typeof API !== 'undefined') {
                API.checkHealth().then(health => {
                    if (health.success && health.data.using_deepseek_fallback) {
                        resultContentElement.innerHTML = `
                            <div class="info-message">
                                <h3>🤖 Usando DeepSeek (Modo Fallback)</h3>
                                <p>Flux2 no está disponible, pero el sistema está funcionando con DeepSeek.</p>
                                <p>Por favor, intenta nuevamente. El procesamiento funcionará con DeepSeek.</p>
                                <button class="btn" onclick="document.getElementById('clothingForm').dispatchEvent(new Event('submit', { bubbles: true }))">
                                    🔄 Reintentar con DeepSeek
                                </button>
                            </div>
                        `;
                    } else {
                        if (typeof UI !== 'undefined') {
                            resultContentElement.innerHTML = UI.showError(errorMsg);
                        }
                        if (typeof ErrorHandler !== 'undefined') {
                            ErrorHandler.handleApiError(error, 'Form Submission');
                        }
                    }
                }).catch(() => {
                    if (typeof UI !== 'undefined') {
                        resultContentElement.innerHTML = UI.showError(errorMsg);
                    }
                    if (typeof ErrorHandler !== 'undefined') {
                        ErrorHandler.handleApiError(error, 'Health Check');
                    }
                });
            }
        } else {
            if (typeof UI !== 'undefined') {
                resultContentElement.innerHTML = UI.showError(errorMsg);
            }
            if (typeof ErrorHandler !== 'undefined') {
                ErrorHandler.handleApiError(error, 'Form Submission');
            }
        }
    },

    /**
     * Track success
     */
    trackSuccess() {
        if (typeof AdvancedAnalytics !== 'undefined') {
            const processingTime = Date.now() - (typeof StateManager !== 'undefined' ? StateManager.get('formStartTime') : Date.now());
            AdvancedAnalytics.trackFormSubmission('clothing_change', true, processingTime);
            
            const imageInput = document.getElementById('imageInput');
            AdvancedAnalytics.trackImageProcessing(
                imageInput?.files[0]?.size || 0,
                processingTime,
                true
            );
        }
    },

    /**
     * Track failure
     */
    trackFailure(error) {
        if (typeof AdvancedAnalytics !== 'undefined') {
            const processingTime = Date.now() - (typeof StateManager !== 'undefined' ? StateManager.get('formStartTime') : Date.now());
            AdvancedAnalytics.trackFormSubmission('clothing_change', false, processingTime);
            AdvancedAnalytics.trackError(error, { context: 'form_submission' });
        }
        
        if (typeof Logger !== 'undefined') {
            Logger.error('Form submission error', error);
        }
    },

    /**
     * Reset button state
     */
    resetButton(submitBtn) {
        if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.textContent = '🚀 Cambiar Ropa';
        }
        
        if (typeof ProgressBar !== 'undefined') {
            ProgressBar.stop();
        }
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FormSubmissionHandler;
}

