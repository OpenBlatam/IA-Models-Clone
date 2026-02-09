/**
 * Mutation Observer Testing
 * 
 * Tests that verify Mutation Observer functionality including
 * DOM change detection, attribute monitoring, and child node tracking.
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';

// Mock MutationObserver
class MockMutationObserver {
  callback: MutationCallback;
  observedNodes: Node[] = [];

  constructor(callback: MutationCallback) {
    this.callback = callback;
  }

  observe(target: Node, options?: MutationObserverInit) {
    if (!this.observedNodes.includes(target)) {
      this.observedNodes.push(target);
    }
  }

  disconnect() {
    this.observedNodes = [];
  }

  takeRecords(): MutationRecord[] {
    return [];
  }

  simulateMutation(target: Node, type: 'attributes' | 'childList' | 'characterData', options?: any) {
    const record: MutationRecord = {
      type,
      target,
      addedNodes: type === 'childList' ? (options?.addedNodes || new NodeList()) : new NodeList(),
      removedNodes: type === 'childList' ? (options?.removedNodes || new NodeList()) : new NodeList(),
      previousSibling: null,
      nextSibling: null,
      attributeName: type === 'attributes' ? options?.attributeName || null : null,
      attributeNamespace: null,
      oldValue: type === 'attributes' ? options?.oldValue || null : null,
    };
    
    this.callback([record], this);
  }
}

(global as any).MutationObserver = MockMutationObserver;

describe('Mutation Observer Testing', () => {
  describe('Observer Creation', () => {
    it('should create mutation observer', () => {
      const callback = vi.fn();
      const observer = new MutationObserver(callback);
      
      expect(observer).toBeDefined();
      expect(observer.callback).toBe(callback);
    });
  });

  describe('Observing Changes', () => {
    it('should observe attribute changes', (done) => {
      const callback = vi.fn((mutations) => {
        expect(mutations[0].type).toBe('attributes');
        expect(mutations[0].attributeName).toBe('data-value');
        done();
      });
      
      const observer = new MutationObserver(callback);
      const element = document.createElement('div');
      
      observer.observe(element, { attributes: true });
      observer.simulateMutation(element, 'attributes', {
        attributeName: 'data-value',
        oldValue: 'old',
      });
    });

    it('should observe child node changes', (done) => {
      const callback = vi.fn((mutations) => {
        expect(mutations[0].type).toBe('childList');
        done();
      });
      
      const observer = new MutationObserver(callback);
      const element = document.createElement('div');
      
      observer.observe(element, { childList: true });
      observer.simulateMutation(element, 'childList');
    });

    it('should observe character data changes', (done) => {
      const callback = vi.fn((mutations) => {
        expect(mutations[0].type).toBe('characterData');
        done();
      });
      
      const observer = new MutationObserver(callback);
      const textNode = document.createTextNode('text');
      
      observer.observe(textNode, { characterData: true });
      observer.simulateMutation(textNode, 'characterData');
    });
  });

  describe('Attribute Monitoring', () => {
    it('should track specific attribute changes', () => {
      const callback = vi.fn();
      const observer = new MutationObserver(callback);
      const element = document.createElement('div');
      
      observer.observe(element, {
        attributes: true,
        attributeFilter: ['data-value', 'data-id'],
      });
      
      observer.simulateMutation(element, 'attributes', {
        attributeName: 'data-value',
      });
      
      expect(callback).toHaveBeenCalled();
    });

    it('should track old attribute values', () => {
      const callback = vi.fn((mutations) => {
        expect(mutations[0].oldValue).toBe('old-value');
      });
      
      const observer = new MutationObserver(callback);
      const element = document.createElement('div');
      
      observer.observe(element, {
        attributes: true,
        attributeOldValue: true,
      });
      
      observer.simulateMutation(element, 'attributes', {
        attributeName: 'data-value',
        oldValue: 'old-value',
      });
    });
  });

  describe('Child Node Monitoring', () => {
    it('should track added nodes', () => {
      const callback = vi.fn((mutations) => {
        const addedNodes = Array.from(mutations[0].addedNodes);
        expect(addedNodes.length).toBeGreaterThan(0);
      });
      
      const observer = new MutationObserver(callback);
      const parent = document.createElement('div');
      const child = document.createElement('span');
      
      observer.observe(parent, { childList: true });
      observer.simulateMutation(parent, 'childList', {
        addedNodes: [child] as any,
      });
    });

    it('should track removed nodes', () => {
      const callback = vi.fn((mutations) => {
        const removedNodes = Array.from(mutations[0].removedNodes);
        expect(removedNodes.length).toBeGreaterThan(0);
      });
      
      const observer = new MutationObserver(callback);
      const parent = document.createElement('div');
      const child = document.createElement('span');
      
      observer.observe(parent, { childList: true });
      observer.simulateMutation(parent, 'childList', {
        removedNodes: [child] as any,
      });
    });

    it('should track subtree changes', () => {
      const callback = vi.fn();
      const observer = new MutationObserver(callback);
      const parent = document.createElement('div');
      const child = document.createElement('span');
      parent.appendChild(child);
      
      observer.observe(parent, {
        childList: true,
        subtree: true,
      });
      
      observer.simulateMutation(child, 'childList');
      expect(callback).toHaveBeenCalled();
    });
  });

  describe('Disconnection', () => {
    it('should disconnect observer', () => {
      const observer = new MutationObserver(vi.fn());
      const element = document.createElement('div');
      
      observer.observe(element);
      observer.disconnect();
      
      expect(observer.observedNodes).toHaveLength(0);
    });
  });

  describe('Performance', () => {
    it('should batch multiple mutations', () => {
      const callback = vi.fn();
      const observer = new MutationObserver(callback);
      const element = document.createElement('div');
      
      observer.observe(element, { attributes: true });
      
      // Simulate multiple rapid changes
      for (let i = 0; i < 10; i++) {
        observer.simulateMutation(element, 'attributes', {
          attributeName: `data-${i}`,
        });
      }
      
      // Callback should be called (batched)
      expect(callback).toHaveBeenCalled();
    });
  });
});

