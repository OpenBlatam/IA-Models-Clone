/**
 * Custom hook for state machine
 * 
 * Provides React integration for state machines
 */

import { useMemo, useCallback } from "react";
import {
  StateMachine,
  type StateMachineConfig,
  type StateTransition,
} from "../utils/state/state-machine";

/**
 * Return type for useStateMachine hook
 */
export interface UseStateMachineReturn<TState extends string, TEvent extends string> {
  /** Current state */
  readonly state: TState;
  /** Transition to a new state */
  readonly transition: (event: TEvent) => Promise<boolean>;
  /** Check if transition is possible */
  readonly canTransition: (event: TEvent) => boolean;
  /** Get possible next states */
  readonly getNextStates: (event: TEvent) => TState[];
  /** Reset to initial state */
  readonly reset: () => void;
}

/**
 * Custom hook for state machine
 * 
 * @param config - State machine configuration
 * @returns State machine operations
 * 
 * @example
 * ```typescript
 * type AgentState = "idle" | "running" | "paused" | "stopped";
 * type AgentEvent = "start" | "pause" | "resume" | "stop";
 * 
 * const { state, transition, canTransition } = useStateMachine<AgentState, AgentEvent>({
 *   initialState: "idle",
 *   states: ["idle", "running", "paused", "stopped"],
 *   transitions: [
 *     { from: "idle", to: "running", event: "start" },
 *     { from: "running", to: "paused", event: "pause" },
 *     { from: "paused", to: "running", event: "resume" },
 *     { from: "running", to: "stopped", event: "stop" },
 *   ],
 * });
 * 
 * await transition("start");
 * ```
 */
export function useStateMachine<TState extends string, TEvent extends string>(
  config: StateMachineConfig<TState, TEvent>
): UseStateMachineReturn<TState, TEvent> {
  const machine = useMemo(() => new StateMachine(config), [config]);

  const transition = useCallback(
    async (event: TEvent): Promise<boolean> => {
      return machine.transition(event);
    },
    [machine]
  );

  const canTransition = useCallback(
    (event: TEvent): boolean => {
      return machine.canTransition(event);
    },
    [machine]
  );

  const getNextStates = useCallback(
    (event: TEvent): TState[] => {
      return machine.getNextStates(event);
    },
    [machine]
  );

  const reset = useCallback(() => {
    machine.reset();
  }, [machine]);

  return {
    state: machine.getState(),
    transition,
    canTransition,
    getNextStates,
    reset,
  };
}




