/**
 * Modal utilities
 */

/**
 * Prevent body scroll (for web compatibility)
 */
export const preventBodyScroll = (prevent: boolean) => {
  // This is mainly for web compatibility
  // In React Native, modals handle this automatically
  if (typeof document !== 'undefined') {
    if (prevent) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'unset';
    }
  }
};

/**
 * Get modal animation config
 */
export const getModalAnimationConfig = (
  animationType: 'fade' | 'slide' | 'scale'
) => {
  switch (animationType) {
    case 'fade':
      return {
        opacity: { from: 0, to: 1 },
        duration: 300,
      };
    case 'slide':
      return {
        translateY: { from: 100, to: 0 },
        duration: 300,
      };
    case 'scale':
      return {
        scale: { from: 0.9, to: 1 },
        opacity: { from: 0, to: 1 },
        duration: 300,
      };
    default:
      return {
        opacity: { from: 0, to: 1 },
        duration: 300,
      };
  }
};

/**
 * Calculate modal position
 */
export const calculateModalPosition = (
  screenWidth: number,
  screenHeight: number,
  modalWidth: number,
  modalHeight: number,
  position: 'center' | 'top' | 'bottom' | 'left' | 'right'
) => {
  switch (position) {
    case 'center':
      return {
        left: (screenWidth - modalWidth) / 2,
        top: (screenHeight - modalHeight) / 2,
      };
    case 'top':
      return {
        left: (screenWidth - modalWidth) / 2,
        top: 0,
      };
    case 'bottom':
      return {
        left: (screenWidth - modalWidth) / 2,
        top: screenHeight - modalHeight,
      };
    case 'left':
      return {
        left: 0,
        top: (screenHeight - modalHeight) / 2,
      };
    case 'right':
      return {
        left: screenWidth - modalWidth,
        top: (screenHeight - modalHeight) / 2,
      };
    default:
      return {
        left: (screenWidth - modalWidth) / 2,
        top: (screenHeight - modalHeight) / 2,
      };
  }
};

