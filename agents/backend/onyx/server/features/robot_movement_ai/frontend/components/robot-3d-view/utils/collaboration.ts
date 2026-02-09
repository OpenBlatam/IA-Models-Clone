/**
 * Basic collaboration utilities
 * @module robot-3d-view/utils/collaboration
 */

import type { SceneConfig } from '../schemas/validation-schemas';

/**
 * Collaboration event types
 */
export type CollaborationEventType =
  | 'config-changed'
  | 'position-changed'
  | 'user-joined'
  | 'user-left';

/**
 * Collaboration event
 */
export interface CollaborationEvent {
  type: CollaborationEventType;
  userId: string;
  timestamp: number;
  data?: unknown;
}

/**
 * Collaboration user
 */
export interface CollaborationUser {
  id: string;
  name: string;
  color: string;
  cursor?: {
    x: number;
    y: number;
  };
}

/**
 * Collaboration manager (basic implementation)
 * 
 * Note: This is a basic implementation. For real-time collaboration,
 * you would integrate with WebSockets, WebRTC, or a service like Socket.io.
 */
export class CollaborationManager {
  private users: Map<string, CollaborationUser> = new Map();
  private events: CollaborationEvent[] = [];
  private maxEvents = 1000;
  private listeners: Map<CollaborationEventType, Set<(event: CollaborationEvent) => void>> = new Map();

  /**
   * Adds a user
   */
  addUser(user: CollaborationUser): void {
    this.users.set(user.id, user);
    this.emitEvent({
      type: 'user-joined',
      userId: user.id,
      timestamp: Date.now(),
      data: user,
    });
  }

  /**
   * Removes a user
   */
  removeUser(userId: string): void {
    const user = this.users.get(userId);
    if (user) {
      this.users.delete(userId);
      this.emitEvent({
        type: 'user-left',
        userId,
        timestamp: Date.now(),
        data: user,
      });
    }
  }

  /**
   * Gets all users
   */
  getUsers(): CollaborationUser[] {
    return Array.from(this.users.values());
  }

  /**
   * Gets a user
   */
  getUser(userId: string): CollaborationUser | undefined {
    return this.users.get(userId);
  }

  /**
   * Updates user cursor
   */
  updateUserCursor(userId: string, cursor: { x: number; y: number }): void {
    const user = this.users.get(userId);
    if (user) {
      user.cursor = cursor;
    }
  }

  /**
   * Broadcasts config change
   */
  broadcastConfigChange(userId: string, config: SceneConfig): void {
    this.emitEvent({
      type: 'config-changed',
      userId,
      timestamp: Date.now(),
      data: config,
    });
  }

  /**
   * Broadcasts position change
   */
  broadcastPositionChange(userId: string, position: number[]): void {
    this.emitEvent({
      type: 'position-changed',
      userId,
      timestamp: Date.now(),
      data: position,
    });
  }

  /**
   * Emits an event
   */
  private emitEvent(event: CollaborationEvent): void {
    this.events.push(event);
    if (this.events.length > this.maxEvents) {
      this.events.shift();
    }

    const listeners = this.listeners.get(event.type);
    if (listeners) {
      listeners.forEach((listener) => {
        try {
          listener(event);
        } catch (error) {
          console.error('Collaboration event listener error:', error);
        }
      });
    }
  }

  /**
   * Subscribes to events
   */
  on(eventType: CollaborationEventType, listener: (event: CollaborationEvent) => void): () => void {
    if (!this.listeners.has(eventType)) {
      this.listeners.set(eventType, new Set());
    }
    this.listeners.get(eventType)!.add(listener);

    // Return unsubscribe function
    return () => {
      this.listeners.get(eventType)?.delete(listener);
    };
  }

  /**
   * Gets recent events
   */
  getRecentEvents(count = 100): CollaborationEvent[] {
    return this.events.slice(-count);
  }

  /**
   * Clears all events
   */
  clearEvents(): void {
    this.events = [];
  }
}

/**
 * Global collaboration manager instance
 */
export const collaborationManager = new CollaborationManager();



