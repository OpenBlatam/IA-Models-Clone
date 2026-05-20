/**
 * State machine utility functions.
 * Provides helper functions for state machine implementation.
 */

/**
 * State machine transition.
 */
export interface StateTransition<TState, TEvent> {
  from: TState;
  event: TEvent;
  to: TState;
  guard?: () => boolean;
  action?: () => void;
}

/**
 * State machine configuration.
 */
export interface StateMachineConfig<TState, TEvent> {
  initialState: TState;
  transitions: StateTransition<TState, TEvent>[];
  onStateChange?: (from: TState, to: TState, event: TEvent) => void;
}

/**
 * State machine class.
 */
export class StateMachine<TState extends string, TEvent extends string> {
  private currentState: TState;
  private transitions: Map<string, StateTransition<TState, TEvent>>;
  private onStateChange?: (from: TState, to: TState, event: TEvent) => void;

  constructor(config: StateMachineConfig<TState, TEvent>) {
    this.currentState = config.initialState;
    this.onStateChange = config.onStateChange;
    this.transitions = new Map();

    for (const transition of config.transitions) {
      const key = `${transition.from}:${transition.event}`;
      this.transitions.set(key, transition);
    }
  }

  /**
   * Gets current state.
   */
  get state(): TState {
    return this.currentState;
  }

  /**
   * Checks if transition is possible.
   */
  canTransition(event: TEvent): boolean {
    const key = `${this.currentState}:${event}`;
    const transition = this.transitions.get(key);

    if (!transition) {
      return false;
    }

    if (transition.guard && !transition.guard()) {
      return false;
    }

    return true;
  }

  /**
   * Transitions to new state.
   */
  transition(event: TEvent): boolean {
    const key = `${this.currentState}:${event}`;
    const transition = this.transitions.get(key);

    if (!transition) {
      return false;
    }

    if (transition.guard && !transition.guard()) {
      return false;
    }

    const fromState = this.currentState;
    this.currentState = transition.to;

    if (transition.action) {
      transition.action();
    }

    if (this.onStateChange) {
      this.onStateChange(fromState, this.currentState, event);
    }

    return true;
  }

  /**
   * Gets available transitions for current state.
   */
  getAvailableTransitions(): TEvent[] {
    const available: TEvent[] = [];

    for (const [key, transition] of this.transitions.entries()) {
      if (transition.from === this.currentState) {
        if (!transition.guard || transition.guard()) {
          available.push(transition.event);
        }
      }
    }

    return available;
  }

  /**
   * Resets state machine to initial state.
   */
  reset(): void {
    this.currentState = this.transitions.values().next().value?.from || this.currentState;
  }
}

/**
 * Creates a state machine.
 */
export function createStateMachine<TState extends string, TEvent extends string>(
  config: StateMachineConfig<TState, TEvent>
): StateMachine<TState, TEvent> {
  return new StateMachine(config);
}

