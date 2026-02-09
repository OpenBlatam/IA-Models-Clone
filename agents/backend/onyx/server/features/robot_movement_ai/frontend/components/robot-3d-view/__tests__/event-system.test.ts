/**
 * Tests for event system
 * @module robot-3d-view/__tests__/event-system
 */

import { describe, it, expect, beforeEach } from '@jest/globals';
import { EventManager, Events } from '../utils/event-system';

describe('EventManager', () => {
  let eventManager: EventManager;

  beforeEach(() => {
    eventManager = new EventManager();
  });

  describe('on', () => {
    it('should subscribe to events', () => {
      let called = false;
      eventManager.on(Events.CONFIG_CHANGED, () => {
        called = true;
      });

      eventManager.emit(Events.CONFIG_CHANGED, {});
      expect(called).toBe(true);
    });

    it('should allow multiple handlers', () => {
      let callCount = 0;
      eventManager.on(Events.CONFIG_CHANGED, () => {
        callCount++;
      });
      eventManager.on(Events.CONFIG_CHANGED, () => {
        callCount++;
      });

      eventManager.emit(Events.CONFIG_CHANGED, {});
      expect(callCount).toBe(2);
    });
  });

  describe('once', () => {
    it('should call handler only once', () => {
      let callCount = 0;
      eventManager.once(Events.CONFIG_CHANGED, () => {
        callCount++;
      });

      eventManager.emit(Events.CONFIG_CHANGED, {});
      eventManager.emit(Events.CONFIG_CHANGED, {});
      expect(callCount).toBe(1);
    });
  });

  describe('off', () => {
    it('should unsubscribe from events', () => {
      let called = false;
      const handler = () => {
        called = true;
      };

      eventManager.on(Events.CONFIG_CHANGED, handler);
      eventManager.off(Events.CONFIG_CHANGED, handler);
      eventManager.emit(Events.CONFIG_CHANGED, {});

      expect(called).toBe(false);
    });
  });

  describe('emit', () => {
    it('should pass data to handlers', async () => {
      let receivedData: unknown;
      eventManager.on(Events.CONFIG_CHANGED, (data) => {
        receivedData = data;
      });

      const testData = { test: 'value' };
      await eventManager.emit(Events.CONFIG_CHANGED, testData);
      expect(receivedData).toEqual(testData);
    });

    it('should handle async handlers', async () => {
      let resolved = false;
      eventManager.on(Events.CONFIG_CHANGED, async () => {
        await new Promise((resolve) => setTimeout(resolve, 10));
        resolved = true;
      });

      await eventManager.emit(Events.CONFIG_CHANGED, {});
      expect(resolved).toBe(true);
    });
  });

  describe('removeAllListeners', () => {
    it('should remove all listeners for an event', () => {
      let called = false;
      eventManager.on(Events.CONFIG_CHANGED, () => {
        called = true;
      });

      eventManager.removeAllListeners(Events.CONFIG_CHANGED);
      eventManager.emit(Events.CONFIG_CHANGED, {});
      expect(called).toBe(false);
    });

    it('should remove all listeners if no event specified', () => {
      let called = false;
      eventManager.on(Events.CONFIG_CHANGED, () => {
        called = true;
      });

      eventManager.removeAllListeners();
      eventManager.emit(Events.CONFIG_CHANGED, {});
      expect(called).toBe(false);
    });
  });

  describe('listenerCount', () => {
    it('should return correct listener count', () => {
      eventManager.on(Events.CONFIG_CHANGED, () => {});
      eventManager.on(Events.CONFIG_CHANGED, () => {});
      eventManager.once(Events.CONFIG_CHANGED, () => {});

      expect(eventManager.listenerCount(Events.CONFIG_CHANGED)).toBe(3);
    });
  });
});



