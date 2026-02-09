/**
 * State machine utilities
 * 
 * Provides finite state machine implementation
 */

/**
 * State transition
 */
export interface StateTransition<TState extends string, TEvent extends string> {
  readonly from: TState;
  readonly to: TState;
  readonly event: TEvent;
  readonly guard?: () => boolean | Promise<boolean>;
  readonly action?: () => void | Promise<void>;
}

/**
 * State machine configuration
 */
export interface StateMachineConfig<TState extends string, TEvent extends string> {
  readonly initialState: TState;
  readonly states: readonly TState[];
  readonly transitions: readonly StateTransition<TState, TEvent>[];
  readonly onStateChange?: (from: TState, to: TState, event: TEvent) => void | Promise<void>;
}

/**
 * State machine class
 */
export class StateMachine<TState extends string, TEvent extends string> {
  private currentState: TState;
  private readonly transitions: Map<string, StateTransition<TState, TEvent>>;
  private readonly onStateChange?: (from: TState, to: TState, event: TEvent) => void | Promise<void>;

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
   * Gets current state
   */
  getState(): TState {
    return this.currentState;
  }

  /**
   * Checks if transition is possible
   */
  canTransition(event: TEvent): boolean {
    const key = `${this.currentState}:${event}`;
    const transition = this.transitions.get(key);
    return transition !== undefined;
  }

  /**
   * Gets possible next states for an event
   */
  getNextStates(event: TEvent): TState[] {
    const key = `${this.currentState}:${event}`;
    const transition = this.transitions.get(key);
    return transition ? [transition.to] : [];
  }

  /**
   * Transitions to a new state
   */
  async transition(event: TEvent): Promise<boolean> {
    const key = `${this.currentState}:${event}`;
    const transition = this.transitions.get(key);

    if (!transition) {
      return false;
    }

    // Check guard
    if (transition.guard) {
      const canTransition = await transition.guard();
      if (!canTransition) {
        return false;
      }
    }

    const fromState = this.currentState;
    const toState = transition.to;

    // Execute action
    if (transition.action) {
      await transition.action();
    }

    // Change state
    this.currentState = toState;

    // Call onStateChange callback
    if (this.onStateChange) {
      await this.onStateChange(fromState, toState, event);
    }

    return true;
  }

  /**
   * Resets to initial state
   */
  reset(): void {
    this.currentState = (this as unknown as { initialState: TState }).initialState;
  }
}

/**
 * Creates a state machine
 */
export function createStateMachine<TState extends string, TEvent extends string>(
  config: StateMachineConfig<TState, TEvent>
): StateMachine<TState, TEvent> {
  return new StateMachine(config);
}




