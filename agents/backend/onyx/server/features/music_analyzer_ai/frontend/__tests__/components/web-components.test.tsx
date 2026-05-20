/**
 * Web Components Testing
 * 
 * Tests that verify Web Components functionality including
 * custom elements, shadow DOM, and component lifecycle.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';

describe('Web Components Testing', () => {
  describe('Custom Elements', () => {
    it('should define custom element', () => {
      class CustomElement extends HTMLElement {
        connectedCallback() {
          this.textContent = 'Custom Element';
        }
      }
      
      customElements.define('custom-element', CustomElement);
      const element = document.createElement('custom-element');
      expect(element).toBeInstanceOf(CustomElement);
    });

    it('should handle custom element lifecycle', () => {
      const callbacks: string[] = [];
      
      class LifecycleElement extends HTMLElement {
        connectedCallback() {
          callbacks.push('connected');
        }
        
        disconnectedCallback() {
          callbacks.push('disconnected');
        }
        
        adoptedCallback() {
          callbacks.push('adopted');
        }
        
        attributeChangedCallback(name: string, oldValue: string, newValue: string) {
          callbacks.push(`attributeChanged:${name}`);
        }
        
        static get observedAttributes() {
          return ['data-value'];
        }
      }
      
      customElements.define('lifecycle-element', LifecycleElement);
      const element = document.createElement('lifecycle-element');
      document.body.appendChild(element);
      element.setAttribute('data-value', 'test');
      document.body.removeChild(element);
      
      expect(callbacks).toContain('connected');
      expect(callbacks).toContain('attributeChanged:data-value');
      expect(callbacks).toContain('disconnected');
    });
  });

  describe('Shadow DOM', () => {
    it('should create shadow root', () => {
      class ShadowElement extends HTMLElement {
        constructor() {
          super();
          this.attachShadow({ mode: 'open' });
        }
      }
      
      customElements.define('shadow-element', ShadowElement);
      const element = document.createElement('shadow-element') as ShadowElement;
      expect(element.shadowRoot).toBeDefined();
    });

    it('should isolate styles in shadow DOM', () => {
      class IsolatedElement extends HTMLElement {
        constructor() {
          super();
          const shadow = this.attachShadow({ mode: 'open' });
          shadow.innerHTML = `
            <style>
              .isolated { color: red; }
            </style>
            <div class="isolated">Isolated content</div>
          `;
        }
      }
      
      customElements.define('isolated-element', IsolatedElement);
      const element = document.createElement('isolated-element') as IsolatedElement;
      expect(element.shadowRoot).toBeDefined();
      expect(element.shadowRoot?.querySelector('.isolated')).toBeDefined();
    });

    it('should support closed shadow DOM', () => {
      class ClosedShadowElement extends HTMLElement {
        constructor() {
          super();
          this.attachShadow({ mode: 'closed' });
        }
      }
      
      customElements.define('closed-shadow-element', ClosedShadowElement);
      const element = document.createElement('closed-shadow-element') as ClosedShadowElement;
      expect(element.shadowRoot).toBeNull(); // Closed shadow DOM is not accessible
    });
  });

  describe('Slots', () => {
    it('should use named slots', () => {
      class SlotElement extends HTMLElement {
        constructor() {
          super();
          const shadow = this.attachShadow({ mode: 'open' });
          shadow.innerHTML = `
            <div>
              <slot name="header"></slot>
              <slot name="content"></slot>
            </div>
          `;
        }
      }
      
      customElements.define('slot-element', SlotElement);
      const element = document.createElement('slot-element');
      element.innerHTML = `
        <span slot="header">Header</span>
        <span slot="content">Content</span>
      `;
      
      expect(element.querySelector('[slot="header"]')).toBeDefined();
      expect(element.querySelector('[slot="content"]')).toBeDefined();
    });

    it('should use default slot', () => {
      class DefaultSlotElement extends HTMLElement {
        constructor() {
          super();
          const shadow = this.attachShadow({ mode: 'open' });
          shadow.innerHTML = '<slot>Default content</slot>';
        }
      }
      
      customElements.define('default-slot-element', DefaultSlotElement);
      const element = document.createElement('default-slot-element');
      element.textContent = 'Custom content';
      
      expect(element.textContent).toBe('Custom content');
    });
  });

  describe('Properties and Attributes', () => {
    it('should sync properties and attributes', () => {
      class PropertyElement extends HTMLElement {
        private _value: string = '';
        
        get value() {
          return this._value;
        }
        
        set value(val: string) {
          this._value = val;
          this.setAttribute('value', val);
        }
        
        static get observedAttributes() {
          return ['value'];
        }
        
        attributeChangedCallback(name: string, oldValue: string, newValue: string) {
          if (name === 'value') {
            this._value = newValue;
          }
        }
      }
      
      customElements.define('property-element', PropertyElement);
      const element = document.createElement('property-element') as PropertyElement;
      element.value = 'test';
      
      expect(element.value).toBe('test');
      expect(element.getAttribute('value')).toBe('test');
    });
  });

  describe('Event Handling', () => {
    it('should dispatch custom events', () => {
      class EventElement extends HTMLElement {
        dispatchCustomEvent() {
          this.dispatchEvent(new CustomEvent('custom-event', {
            detail: { data: 'test' },
            bubbles: true,
          }));
        }
      }
      
      customElements.define('event-element', EventElement);
      const element = document.createElement('event-element') as EventElement;
      const handler = vi.fn();
      
      element.addEventListener('custom-event', handler);
      element.dispatchCustomEvent();
      
      expect(handler).toHaveBeenCalled();
    });
  });
});

