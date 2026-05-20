export const eventUtils = {
  preventDefault: (e: Event | React.SyntheticEvent) => {
    e.preventDefault()
  },

  stopPropagation: (e: Event | React.SyntheticEvent) => {
    e.stopPropagation()
  },

  stopImmediatePropagation: (e: Event) => {
    e.stopImmediatePropagation()
  },

  getEventTarget: (e: Event): EventTarget | null => {
    return e.target
  },

  isKeyboardEvent: (e: Event): e is KeyboardEvent => {
    return e.type === 'keydown' || e.type === 'keyup' || e.type === 'keypress'
  },

  isMouseEvent: (e: Event): e is MouseEvent => {
    return e.type === 'click' || e.type === 'mousedown' || e.type === 'mouseup'
  },

  getKey: (e: KeyboardEvent): string => {
    return e.key
  },

  getModifierKeys: (e: KeyboardEvent | MouseEvent) => {
    return {
      ctrl: e.ctrlKey,
      shift: e.shiftKey,
      alt: e.altKey,
      meta: e.metaKey,
    }
  },
}

