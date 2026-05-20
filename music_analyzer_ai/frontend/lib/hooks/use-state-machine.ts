/**
 * Custom hook for state machine.
 * Provides reactive state machine functionality.
 */

import { useState, useCallback, useRef } from 'react';
import {
  StateMachine,
  createStateMachine,
  StateMachineConfig,
} from '../utils/state-machine';

/**
 * Custom hook for state machine.
 * Provides reactive state machine functionality.
 *
 * @param config - State machine configuration
 * @returns State machine operations
 */
export function useStateMachine<TState extends string, TEvent extends string>(
  config: StateMachineConfig<TState, TEvent>
) {
  const machineRef = useRef<StateMachine<TState, TEvent>>(
    createStateMachine(config)
  );
  const [state, setState] = useState<TState>(machineRef.current.state);

  const transition = useCallback(
    (event: TEvent): boolean => {
      const result = machineRef.current.transition(event);
      if (result) {
        setState(machineRef.current.state);
      }
      return result;
    },
    []
  );

  const canTransition = useCallback(
    (event: TEvent): boolean => {
      return machineRef.current.canTransition(event);
    },
    []
  );

  const getAvailableTransitions = useCallback((): TEvent[] => {
    return machineRef.current.getAvailableTransitions();
  }, []);

  const reset = useCallback(() => {
    machineRef.current.reset();
    setState(machineRef.current.state);
  }, []);

  return {
    state,
    transition,
    canTransition,
    getAvailableTransitions,
    reset,
  };
}

