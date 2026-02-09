/**
 * Telemetry and monitoring system
 * @module robot-3d-view/utils/telemetry
 */

/**
 * Telemetry event
 */
export interface TelemetryEvent {
  name: string;
  timestamp: number;
  properties?: Record<string, unknown>;
  metrics?: Record<string, number>;
}

/**
 * Telemetry session
 */
export interface TelemetrySession {
  id: string;
  startTime: number;
  endTime?: number;
  events: TelemetryEvent[];
  properties?: Record<string, unknown>;
}

/**
 * Telemetry Manager class
 */
export class TelemetryManager {
  private sessions: Map<string, TelemetrySession> = new Map();
  private currentSession: TelemetrySession | null = null;
  private enabled = true;
  private maxEventsPerSession = 10000;
  private maxSessions = 100;

  /**
   * Starts a new session
   */
  startSession(properties?: Record<string, unknown>): string {
    const sessionId = `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    const session: TelemetrySession = {
      id: sessionId,
      startTime: Date.now(),
      events: [],
      properties,
    };

    this.sessions.set(sessionId, session);
    this.currentSession = session;

    // Limit sessions
    if (this.sessions.size > this.maxSessions) {
      const oldest = Array.from(this.sessions.values())
        .sort((a, b) => a.startTime - b.startTime)[0];
      this.sessions.delete(oldest.id);
    }

    this.track('session:started', { sessionId }, properties);
    return sessionId;
  }

  /**
   * Ends current session
   */
  endSession(): void {
    if (this.currentSession) {
      this.currentSession.endTime = Date.now();
      this.track('session:ended', {
        duration: this.currentSession.endTime - this.currentSession.startTime,
      });
      this.currentSession = null;
    }
  }

  /**
   * Tracks an event
   */
  track(
    name: string,
    properties?: Record<string, unknown>,
    metrics?: Record<string, number>
  ): void {
    if (!this.enabled || !this.currentSession) return;

    const event: TelemetryEvent = {
      name,
      timestamp: Date.now(),
      properties,
      metrics,
    };

    this.currentSession.events.push(event);

    // Limit events
    if (this.currentSession.events.length > this.maxEventsPerSession) {
      this.currentSession.events.shift();
    }
  }

  /**
   * Tracks a page view
   */
  trackPageView(page: string, properties?: Record<string, unknown>): void {
    this.track('page:view', { page, ...properties });
  }

  /**
   * Tracks user action
   */
  trackAction(action: string, properties?: Record<string, unknown>): void {
    this.track('action', { action, ...properties });
  }

  /**
   * Tracks performance metric
   */
  trackPerformance(metric: string, value: number, properties?: Record<string, unknown>): void {
    this.track('performance', properties, { [metric]: value });
  }

  /**
   * Tracks error
   */
  trackError(error: Error, properties?: Record<string, unknown>): void {
    this.track('error', {
      message: error.message,
      stack: error.stack,
      name: error.name,
      ...properties,
    });
  }

  /**
   * Gets current session
   */
  getCurrentSession(): TelemetrySession | null {
    return this.currentSession;
  }

  /**
   * Gets all sessions
   */
  getAllSessions(): TelemetrySession[] {
    return Array.from(this.sessions.values());
  }

  /**
   * Exports session data
   */
  exportSession(sessionId: string): string | null {
    const session = this.sessions.get(sessionId);
    if (!session) return null;

    return JSON.stringify(session, null, 2);
  }

  /**
   * Exports all sessions
   */
  exportAllSessions(): string {
    return JSON.stringify(Array.from(this.sessions.values()), null, 2);
  }

  /**
   * Clears all sessions
   */
  clear(): void {
    this.sessions.clear();
    this.currentSession = null;
  }

  /**
   * Enables/disables telemetry
   */
  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
  }

  /**
   * Checks if telemetry is enabled
   */
  isEnabled(): boolean {
    return this.enabled;
  }
}

/**
 * Global telemetry manager instance
 */
export const telemetryManager = new TelemetryManager();

// Auto-start session
telemetryManager.startSession({
  userAgent: navigator.userAgent,
  language: navigator.language,
  platform: navigator.platform,
});



