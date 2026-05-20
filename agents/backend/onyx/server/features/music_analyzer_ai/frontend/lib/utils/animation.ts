/**
 * Animation utility functions.
 * Provides helper functions for animations and transitions.
 */

/**
 * Easing functions for animations.
 */
export const easing = {
  linear: (t: number) => t,
  easeIn: (t: number) => t * t,
  easeOut: (t: number) => t * (2 - t),
  easeInOut: (t: number) => (t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t),
  easeInQuad: (t: number) => t * t,
  easeOutQuad: (t: number) => t * (2 - t),
  easeInOutQuad: (t: number) =>
    t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t,
  easeInCubic: (t: number) => t * t * t,
  easeOutCubic: (t: number) => --t * t * t + 1,
  easeInOutCubic: (t: number) =>
    t < 0.5 ? 4 * t * t * t : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1,
} as const;

/**
 * Animates a value from start to end.
 * @param start - Start value
 * @param end - End value
 * @param duration - Duration in milliseconds
 * @param easingFn - Easing function
 * @param onUpdate - Update callback
 * @returns Promise that resolves when animation completes
 */
export function animate(
  start: number,
  end: number,
  duration: number,
  easingFn: (t: number) => number = easing.easeInOut,
  onUpdate: (value: number) => void
): Promise<void> {
  return new Promise((resolve) => {
    const startTime = performance.now();
    const difference = end - start;

    const update = (currentTime: number) => {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = easingFn(progress);
      const value = start + difference * eased;

      onUpdate(value);

      if (progress < 1) {
        requestAnimationFrame(update);
      } else {
        resolve();
      }
    };

    requestAnimationFrame(update);
  });
}

/**
 * Fades in an element.
 * @param element - Element to fade in
 * @param duration - Duration in milliseconds
 * @returns Promise that resolves when animation completes
 */
export function fadeIn(
  element: HTMLElement,
  duration: number = 300
): Promise<void> {
  element.style.opacity = '0';
  element.style.display = 'block';

  return animate(0, 1, duration, easing.easeInOut, (value) => {
    element.style.opacity = String(value);
  });
}

/**
 * Fades out an element.
 * @param element - Element to fade out
 * @param duration - Duration in milliseconds
 * @returns Promise that resolves when animation completes
 */
export function fadeOut(
  element: HTMLElement,
  duration: number = 300
): Promise<void> {
  return animate(1, 0, duration, easing.easeInOut, (value) => {
    element.style.opacity = String(value);
  }).then(() => {
    element.style.display = 'none';
  });
}

/**
 * Slides an element in.
 * @param element - Element to slide in
 * @param direction - Slide direction
 * @param duration - Duration in milliseconds
 * @returns Promise that resolves when animation completes
 */
export function slideIn(
  element: HTMLElement,
  direction: 'up' | 'down' | 'left' | 'right' = 'up',
  duration: number = 300
): Promise<void> {
  const directions = {
    up: { start: 'translateY(100%)', end: 'translateY(0)' },
    down: { start: 'translateY(-100%)', end: 'translateY(0)' },
    left: { start: 'translateX(100%)', end: 'translateX(0)' },
    right: { start: 'translateX(-100%)', end: 'translateX(0)' },
  };

  const { start, end } = directions[direction];
  element.style.transform = start;
  element.style.display = 'block';

  return animate(0, 1, duration, easing.easeOut, (value) => {
    const current = value === 1 ? end : start;
    element.style.transform = current;
    element.style.opacity = String(value);
  });
}

/**
 * Slides an element out.
 * @param element - Element to slide out
 * @param direction - Slide direction
 * @param duration - Duration in milliseconds
 * @returns Promise that resolves when animation completes
 */
export function slideOut(
  element: HTMLElement,
  direction: 'up' | 'down' | 'left' | 'right' = 'down',
  duration: number = 300
): Promise<void> {
  const directions = {
    up: { end: 'translateY(-100%)' },
    down: { end: 'translateY(100%)' },
    left: { end: 'translateX(-100%)' },
    right: { end: 'translateX(100%)' },
  };

  const { end } = directions[direction];

  return animate(1, 0, duration, easing.easeIn, (value) => {
    element.style.transform = value === 0 ? end : 'translate(0)';
    element.style.opacity = String(value);
  }).then(() => {
    element.style.display = 'none';
  });
}

