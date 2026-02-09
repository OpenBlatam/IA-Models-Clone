/**
 * Animation utilities and easing functions
 * @module robot-3d-view/utils/animations
 */

/**
 * Easing function type
 */
export type EasingFunction = (t: number) => number;

/**
 * Easing functions
 */
export const Easing = {
  /**
   * Linear easing (no easing)
   */
  linear: (t: number): number => t,

  /**
   * Ease in (slow start)
   */
  easeIn: (t: number): number => t * t,

  /**
   * Ease out (slow end)
   */
  easeOut: (t: number): number => t * (2 - t),

  /**
   * Ease in-out (slow start and end)
   */
  easeInOut: (t: number): number =>
    t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t,

  /**
   * Ease in cubic
   */
  easeInCubic: (t: number): number => t * t * t,

  /**
   * Ease out cubic
   */
  easeOutCubic: (t: number): number => --t * t * t + 1,

  /**
   * Ease in-out cubic
   */
  easeInOutCubic: (t: number): number =>
    t < 0.5 ? 4 * t * t * t : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1,

  /**
   * Elastic easing
   */
  elastic: (t: number): number =>
    t === 0 || t === 1
      ? t
      : -Math.pow(2, 10 * (t - 1)) * Math.sin((t - 1.1) * 5 * Math.PI),

  /**
   * Bounce easing
   */
  bounce: (t: number): number => {
    if (t < 1 / 2.75) {
      return 7.5625 * t * t;
    } else if (t < 2 / 2.75) {
      return 7.5625 * (t -= 1.5 / 2.75) * t + 0.75;
    } else if (t < 2.5 / 2.75) {
      return 7.5625 * (t -= 2.25 / 2.75) * t + 0.9375;
    } else {
      return 7.5625 * (t -= 2.625 / 2.75) * t + 0.984375;
    }
  },
};

/**
 * Animation options
 */
export interface AnimationOptions {
  duration: number;
  easing?: EasingFunction;
  delay?: number;
  onStart?: () => void;
  onUpdate?: (progress: number) => void;
  onComplete?: () => void;
}

/**
 * Animates a value from start to end
 * 
 * @param start - Start value
 * @param end - End value
 * @param options - Animation options
 * @returns Promise that resolves when animation completes
 */
export function animate(
  start: number,
  end: number,
  options: AnimationOptions
): Promise<void> {
  return new Promise((resolve) => {
    const {
      duration,
      easing = Easing.easeInOut,
      delay = 0,
      onStart,
      onUpdate,
      onComplete,
    } = options;

    let startTime: number | null = null;
    let animationFrameId: number;

    const animateFrame = (currentTime: number) => {
      if (startTime === null) {
        startTime = currentTime;
        if (delay > 0) {
          startTime += delay;
        }
        onStart?.();
      }

      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const easedProgress = easing(progress);
      const value = start + (end - start) * easedProgress;

      onUpdate?.(value);

      if (progress < 1) {
        animationFrameId = requestAnimationFrame(animateFrame);
      } else {
        onComplete?.();
        resolve();
      }
    };

    animationFrameId = requestAnimationFrame(animateFrame);
  });
}

/**
 * Interpolates between two values
 * 
 * @param start - Start value
 * @param end - End value
 * @param t - Progress (0-1)
 * @param easing - Easing function
 * @returns Interpolated value
 */
export function lerp(
  start: number,
  end: number,
  t: number,
  easing?: EasingFunction
): number {
  const easedT = easing ? easing(t) : t;
  return start + (end - start) * easedT;
}

/**
 * Smooth step interpolation
 * 
 * @param t - Progress (0-1)
 * @returns Smoothed value
 */
export function smoothStep(t: number): number {
  return t * t * (3 - 2 * t);
}

/**
 * Creates a spring animation
 * 
 * @param target - Target value
 * @param current - Current value
 * @param velocity - Current velocity
 * @param stiffness - Spring stiffness
 * @param damping - Spring damping
 * @param deltaTime - Time delta
 * @returns New value and velocity
 */
export function spring(
  target: number,
  current: number,
  velocity: number,
  stiffness: number,
  damping: number,
  deltaTime: number
): { value: number; velocity: number } {
  const force = (target - current) * stiffness;
  const dampingForce = velocity * damping;
  const acceleration = (force - dampingForce) / 1; // mass = 1

  const newVelocity = velocity + acceleration * deltaTime;
  const newValue = current + newVelocity * deltaTime;

  return { value: newValue, velocity: newVelocity };
}



