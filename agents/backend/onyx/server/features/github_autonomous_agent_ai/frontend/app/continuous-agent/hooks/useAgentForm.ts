/**
 * Custom hook for managing agent form state and validation
 * 
 * Features:
 * - Form state management for all agent fields
 * - Real-time validation with error messages
 * - Type-safe form operations
 * - Config generation for API requests
 */

import { useState, useCallback, useMemo } from "react";
import type { AgentConfig, TaskType } from "../types";
import { FORM_DEFAULTS, TASK_TYPES } from "../constants";
import {
  validateName,
  validateDescription,
  validateFrequency,
  validateJSON,
  validateGoal,
  parseJSON,
} from "../utils/validation";

/**
 * Return type for useAgentForm hook
 */
type UseAgentFormReturn = {
  readonly name: string;
  readonly description: string;
  readonly taskType: TaskType;
  readonly frequency: number;
  readonly parameters: string;
  readonly goal?: string;
  readonly errors: {
    readonly name: string | null;
    readonly description: string | null;
    readonly frequency: string | null;
    readonly parameters: string | null;
    readonly goal: string | null;
  };
  readonly isValid: boolean;
  readonly setName: (value: string) => void;
  readonly setDescription: (value: string) => void;
  readonly setTaskType: (value: TaskType) => void;
  readonly setFrequency: (value: number) => void;
  readonly setParameters: (value: string) => void;
  readonly setGoal: (value: string) => void;
  readonly reset: () => void;
  readonly getConfig: () => AgentConfig;
  readonly validate: () => boolean;
};

/**
 * Custom hook for managing agent form state and validation
 * 
 * @returns Form state, setters, validators, and config generator
 */
export const useAgentForm = (): UseAgentFormReturn => {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [taskType, setTaskType] = useState<TaskType>(FORM_DEFAULTS.TASK_TYPE);
  const [frequency, setFrequency] = useState(FORM_DEFAULTS.FREQUENCY);
  const [parameters, setParameters] = useState(FORM_DEFAULTS.PARAMETERS);
  const [goal, setGoal] = useState<string>(FORM_DEFAULTS.GOAL);
  const [errors, setErrors] = useState<{
    readonly name: string | null;
    readonly description: string | null;
    readonly frequency: string | null;
    readonly parameters: string | null;
    readonly goal: string | null;
  }>({
    name: null,
    description: null,
    frequency: null,
    parameters: null,
    goal: null,
  });

  /**
   * Validates all form fields and updates error state
   * @returns True if all fields are valid, false otherwise
   */
  const validate = useCallback((): boolean => {
    const nameValidation = validateName(name);
    const descriptionValidation = validateDescription(description);
    const frequencyValidation = validateFrequency(frequency);
    const parametersValidation = validateJSON(parameters);
    const goalValidation = validateGoal(goal);

    setErrors({
      name: nameValidation.error,
      description: descriptionValidation.error,
      frequency: frequencyValidation.error,
      parameters: parametersValidation.error,
      goal: goalValidation.error,
    });

    return (
      nameValidation.isValid &&
      descriptionValidation.isValid &&
      frequencyValidation.isValid &&
      parametersValidation.isValid &&
      goalValidation.isValid
    );
  }, [name, description, frequency, parameters, goal]);

  /**
   * Generates AgentConfig from current form state
   * @returns AgentConfig object ready for API submission
   */
  const getConfig = useCallback((): AgentConfig => {
    const config: AgentConfig = {
      taskType,
      frequency,
      parameters: parseJSON(parameters),
    };
    
    // Only include goal if it's not empty
    if (goal?.trim()) {
      config.goal = goal.trim();
    }
    
    return config;
  }, [taskType, frequency, parameters, goal]);

  /**
   * Resets form to initial state
   */
  const reset = useCallback((): void => {
    setName("");
    setDescription("");
    setTaskType(FORM_DEFAULTS.TASK_TYPE);
    setFrequency(FORM_DEFAULTS.FREQUENCY);
    setParameters(FORM_DEFAULTS.PARAMETERS);
    setGoal(FORM_DEFAULTS.GOAL);
    setErrors({
      name: null,
      description: null,
      frequency: null,
      parameters: null,
      goal: null,
    });
  }, []);

  // Memoize isValid to prevent unnecessary recalculations
  const isValid = useMemo(
    () => name.trim().length > 0 && description.trim().length > 0,
    [name, description]
  );

  return {
    name,
    description,
    taskType,
    frequency,
    parameters,
    goal,
    errors,
    isValid,
    setName,
    setDescription,
    setTaskType,
    setFrequency,
    setParameters,
    setGoal,
    reset,
    getConfig,
    validate,
  };
};

