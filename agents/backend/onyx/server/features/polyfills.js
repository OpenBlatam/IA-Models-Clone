if (typeof global !== 'undefined') {
  if (typeof global.self === 'undefined') {
    global.self = global;
  }
  
  if (typeof global.window === 'undefined') {
    global.window = {
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => {},
      location: { href: '', origin: '', pathname: '/' },
      history: { pushState: () => {}, replaceState: () => {} },
      navigator: { userAgent: 'node' },
      document: {
        createElement: () => ({}),
        getElementById: () => null,
        querySelector: () => null,
        addEventListener: () => {},
        removeEventListener: () => {}
      },
      requestAnimationFrame: (cb) => setTimeout(cb, 16),
      cancelAnimationFrame: (id) => clearTimeout(id),
      getComputedStyle: () => ({}),
      matchMedia: () => ({ matches: false, addListener: () => {}, removeListener: () => {} })
    };
  }
  
  if (typeof global.document === 'undefined') {
    global.document = global.window.document;
  }
  
  if (typeof global.navigator === 'undefined') {
    global.navigator = global.window.navigator;
  }
  
  if (typeof global.HTMLElement === 'undefined') {
    global.HTMLElement = class HTMLElement {};
  }
  
  if (typeof global.Element === 'undefined') {
    global.Element = class Element {};
  }
}
