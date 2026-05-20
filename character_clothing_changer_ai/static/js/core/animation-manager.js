/**
 * Animation Manager Module
 * =======================
 * Manages animations with performance optimization
 */

const AnimationManager = {
    /**
     * Active animations
     */
    animations: new Map(),
    
    /**
     * Animation queue
     */
    queue: [],
    
    /**
     * Initialize animation manager
     */
    init() {
        // Check for reduced motion preference
        if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
            this.disableAnimations();
        }
        
        if (typeof Logger !== 'undefined') {
            Logger.info('Animation manager initialized');
        }
    },
    
    /**
     * Animate element
     */
    animate(element, keyframes, options = {}) {
        if (!element) {
            return null;
        }
        
        const {
            duration = 300,
            easing = 'ease',
            fill = 'forwards',
            delay = 0,
            iterations = 1,
            direction = 'normal'
        } = options;
        
        const animation = element.animate(keyframes, {
            duration,
            easing,
            fill,
            delay,
            iterations,
            direction
        });
        
        const animationId = `anim_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        this.animations.set(animationId, { animation, element });
        
        animation.onfinish = () => {
            this.animations.delete(animationId);
        };
        
        return animationId;
    },
    
    /**
     * Fade in
     */
    fadeIn(element, duration = 300) {
        return this.animate(element, [
            { opacity: 0 },
            { opacity: 1 }
        ], { duration });
    },
    
    /**
     * Fade out
     */
    fadeOut(element, duration = 300) {
        return this.animate(element, [
            { opacity: 1 },
            { opacity: 0 }
        ], { duration });
    },
    
    /**
     * Slide in
     */
    slideIn(element, direction = 'right', duration = 300) {
        const transforms = {
            right: ['translateX(100%)', 'translateX(0)'],
            left: ['translateX(-100%)', 'translateX(0)'],
            up: ['translateY(100%)', 'translateY(0)'],
            down: ['translateY(-100%)', 'translateY(0)']
        };
        
        return this.animate(element, [
            { transform: transforms[direction][0], opacity: 0 },
            { transform: transforms[direction][1], opacity: 1 }
        ], { duration });
    },
    
    /**
     * Slide out
     */
    slideOut(element, direction = 'right', duration = 300) {
        const transforms = {
            right: ['translateX(0)', 'translateX(100%)'],
            left: ['translateX(0)', 'translateX(-100%)'],
            up: ['translateY(0)', 'translateY(-100%)'],
            down: ['translateY(0)', 'translateY(100%)']
        };
        
        return this.animate(element, [
            { transform: transforms[direction][0], opacity: 1 },
            { transform: transforms[direction][1], opacity: 0 }
        ], { duration });
    },
    
    /**
     * Scale
     */
    scale(element, from = 0, to = 1, duration = 300) {
        return this.animate(element, [
            { transform: `scale(${from})`, opacity: 0 },
            { transform: `scale(${to})`, opacity: 1 }
        ], { duration });
    },
    
    /**
     * Shake
     */
    shake(element, duration = 500) {
        return this.animate(element, [
            { transform: 'translateX(0)' },
            { transform: 'translateX(-10px)' },
            { transform: 'translateX(10px)' },
            { transform: 'translateX(-10px)' },
            { transform: 'translateX(10px)' },
            { transform: 'translateX(0)' }
        ], { duration, iterations: 1 });
    },
    
    /**
     * Pulse
     */
    pulse(element, duration = 1000) {
        return this.animate(element, [
            { transform: 'scale(1)', opacity: 1 },
            { transform: 'scale(1.1)', opacity: 0.8 },
            { transform: 'scale(1)', opacity: 1 }
        ], { duration, iterations: Infinity });
    },
    
    /**
     * Stop animation
     */
    stop(animationId) {
        const anim = this.animations.get(animationId);
        if (anim) {
            anim.animation.cancel();
            this.animations.delete(animationId);
        }
    },
    
    /**
     * Stop all animations
     */
    stopAll() {
        this.animations.forEach(({ animation }) => {
            animation.cancel();
        });
        this.animations.clear();
    },
    
    /**
     * Disable animations
     */
    disableAnimations() {
        document.documentElement.classList.add('reduced-motion');
    },
    
    /**
     * Enable animations
     */
    enableAnimations() {
        document.documentElement.classList.remove('reduced-motion');
    },
    
    /**
     * Queue animation
     */
    queueAnimation(element, keyframes, options) {
        this.queue.push({ element, keyframes, options });
        this.processQueue();
    },
    
    /**
     * Process animation queue
     */
    processQueue() {
        if (this.queue.length === 0) {
            return;
        }
        
        const { element, keyframes, options } = this.queue.shift();
        const animationId = this.animate(element, keyframes, options);
        
        if (animationId) {
            const anim = this.animations.get(animationId);
            if (anim) {
                anim.animation.onfinish = () => {
                    this.animations.delete(animationId);
                    this.processQueue();
                };
            }
        }
    }
};

// Auto-initialize
if (typeof window !== 'undefined') {
    AnimationManager.init();
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AnimationManager;
}

