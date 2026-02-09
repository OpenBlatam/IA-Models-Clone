/**
 * Carousel Manager Module
 * =======================
 * Advanced carousel management with autoplay, navigation, and touch support
 */

const CarouselManager = {
    /**
     * Active carousels
     */
    carousels: new Map(),
    
    /**
     * Default options
     */
    defaultOptions: {
        autoplay: false,
        autoplayInterval: 3000,
        loop: true,
        animation: true,
        animationDuration: 500,
        touch: true,
        keyboard: true,
        indicators: true,
        navigation: true
    },
    
    /**
     * Initialize carousel manager
     */
    init() {
        if (typeof Logger !== 'undefined') {
            Logger.info('Carousel manager initialized');
        }
    },
    
    /**
     * Initialize carousel container
     */
    initContainer(container, options = {}) {
        const config = {
            ...this.defaultOptions,
            ...options
        };
        
        const carouselId = `carousel_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        
        const items = container.querySelectorAll('[data-carousel-item]');
        const track = container.querySelector('[data-carousel-track]') || container;
        
        if (items.length === 0) {
            return null;
        }
        
        let currentIndex = 0;
        let autoplayTimer = null;
        let touchStartX = 0;
        let touchEndX = 0;
        
        // Setup track
        track.style.position = 'relative';
        track.style.overflow = 'hidden';
        
        // Setup items
        items.forEach((item, index) => {
            item.style.display = index === 0 ? 'block' : 'none';
            item.setAttribute('data-carousel-index', index);
        });
        
        // Show item
        const showItem = (index) => {
            items.forEach((item, i) => {
                if (i === index) {
                    if (config.animation && typeof AnimationManager !== 'undefined') {
                        item.style.display = 'block';
                        AnimationManager.fadeIn(item, config.animationDuration);
                    } else {
                        item.style.display = 'block';
                    }
                } else {
                    item.style.display = 'none';
                }
            });
            
            currentIndex = index;
            
            // Update indicators
            if (config.indicators) {
                this.updateIndicators(container, index, items.length);
            }
            
            // Emit slide change event
            if (typeof EventBus !== 'undefined') {
                EventBus.emit('carousel:slide', { carouselId, index, item: items[index] });
            }
        };
        
        // Next slide
        const nextSlide = () => {
            let nextIndex = currentIndex + 1;
            if (nextIndex >= items.length) {
                nextIndex = config.loop ? 0 : currentIndex;
            }
            showItem(nextIndex);
        };
        
        // Previous slide
        const prevSlide = () => {
            let prevIndex = currentIndex - 1;
            if (prevIndex < 0) {
                prevIndex = config.loop ? items.length - 1 : currentIndex;
            }
            showItem(prevIndex);
        };
        
        // Setup navigation buttons
        if (config.navigation) {
            const prevBtn = container.querySelector('[data-carousel-prev]');
            const nextBtn = container.querySelector('[data-carousel-next]');
            
            if (prevBtn) {
                prevBtn.addEventListener('click', prevSlide);
            }
            
            if (nextBtn) {
                nextBtn.addEventListener('click', nextSlide);
            }
        }
        
        // Setup indicators
        if (config.indicators) {
            this.createIndicators(container, items.length, (index) => {
                showItem(index);
            });
        }
        
        // Setup keyboard navigation
        if (config.keyboard) {
            container.addEventListener('keydown', (e) => {
                if (e.key === 'ArrowLeft') {
                    prevSlide();
                } else if (e.key === 'ArrowRight') {
                    nextSlide();
                }
            });
        }
        
        // Setup touch support
        if (config.touch) {
            track.addEventListener('touchstart', (e) => {
                touchStartX = e.touches[0].clientX;
            });
            
            track.addEventListener('touchend', (e) => {
                touchEndX = e.changedTouches[0].clientX;
                handleSwipe();
            });
            
            const handleSwipe = () => {
                const swipeThreshold = 50;
                const diff = touchStartX - touchEndX;
                
                if (Math.abs(diff) > swipeThreshold) {
                    if (diff > 0) {
                        nextSlide();
                    } else {
                        prevSlide();
                    }
                }
            };
        }
        
        // Setup autoplay
        if (config.autoplay) {
            const startAutoplay = () => {
                autoplayTimer = setInterval(nextSlide, config.autoplayInterval);
            };
            
            const stopAutoplay = () => {
                if (autoplayTimer) {
                    clearInterval(autoplayTimer);
                    autoplayTimer = null;
                }
            };
            
            startAutoplay();
            
            // Pause on hover
            container.addEventListener('mouseenter', stopAutoplay);
            container.addEventListener('mouseleave', startAutoplay);
        }
        
        // Store carousel info
        const carouselInfo = {
            id: carouselId,
            container,
            items: Array.from(items),
            track,
            currentIndex: 0,
            config,
            showItem,
            nextSlide,
            prevSlide,
            autoplayTimer
        };
        
        this.carousels.set(carouselId, carouselInfo);
        
        return carouselId;
    },
    
    /**
     * Create indicators
     */
    createIndicators(container, count, onClick) {
        const indicatorsContainer = document.createElement('div');
        indicatorsContainer.className = 'carousel-indicators';
        
        for (let i = 0; i < count; i++) {
            const indicator = document.createElement('button');
            indicator.className = 'carousel-indicator';
            indicator.setAttribute('data-carousel-indicator', i);
            indicator.setAttribute('aria-label', `Go to slide ${i + 1}`);
            
            if (i === 0) {
                indicator.classList.add('active');
            }
            
            indicator.addEventListener('click', () => {
                onClick(i);
            });
            
            indicatorsContainer.appendChild(indicator);
        }
        
        container.appendChild(indicatorsContainer);
    },
    
    /**
     * Update indicators
     */
    updateIndicators(container, activeIndex, total) {
        const indicators = container.querySelectorAll('.carousel-indicator');
        indicators.forEach((indicator, index) => {
            if (index === activeIndex) {
                indicator.classList.add('active');
            } else {
                indicator.classList.remove('active');
            }
        });
    },
    
    /**
     * Get carousel info
     */
    get(carouselId) {
        return this.carousels.get(carouselId);
    },
    
    /**
     * Get all carousels
     */
    getAll() {
        return Array.from(this.carousels.values());
    }
};

// Auto-initialize
if (typeof window !== 'undefined') {
    CarouselManager.init();
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CarouselManager;
}

