export const preventDefault = (e: Event): void => {
  e.preventDefault();
};

export const stopPropagation = (e: Event): void => {
  e.stopPropagation();
};

export const stopImmediatePropagation = (e: Event): void => {
  e.stopImmediatePropagation();
};

export const getEventTarget = (e: Event): EventTarget | null => {
  return e.target;
};

export const getEventCurrentTarget = (e: Event): EventTarget | null => {
  return e.currentTarget;
};

export const isKeyboardEvent = (e: Event): e is KeyboardEvent => {
  return e instanceof KeyboardEvent;
};

export const isMouseEvent = (e: Event): e is MouseEvent => {
  return e instanceof MouseEvent;
};

export const isTouchEvent = (e: Event): e is TouchEvent => {
  return e instanceof TouchEvent;
};

export const getKeyCode = (e: KeyboardEvent): string => {
  return e.key || e.code;
};

export const isEnterKey = (e: KeyboardEvent): boolean => {
  return e.key === 'Enter' || e.keyCode === 13;
};

export const isEscapeKey = (e: KeyboardEvent): boolean => {
  return e.key === 'Escape' || e.keyCode === 27;
};

export const isSpaceKey = (e: KeyboardEvent): boolean => {
  return e.key === ' ' || e.keyCode === 32;
};

export const isArrowKey = (e: KeyboardEvent): boolean => {
  return ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key);
};



