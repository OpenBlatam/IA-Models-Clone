"use client";

import { useEffect, useCallback, useState } from "react";
import { useForm, type UseFormReturn } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { toast } from "sonner";
import type { CreateAgentRequest, ContinuousAgent } from "../types";
import { createAgentRequestSchema, type AgentFormValues } from "../utils/validation/zod-schemas";
import { FORM_DEFAULTS } from "../constants";
import { ERROR_MESSAGES } from "../constants/messages";
import { useJSONFormatter } from "./useJSONFormatter";

type UseAgentFormModalOptions = {
  readonly initialAgent?: ContinuousAgent | null;
  readonly onSubmit: (data: AgentFormValues) => Promise<void>;
  readonly onSuccess?: () => void;
};

type UseAgentFormModalReturn = {
  readonly form: UseFormReturn<AgentFormValues>;
  readonly isSubmitting: boolean;
  readonly submitError: string | null;
  readonly formatJSON: () => void;
  readonly handleSubmit: (e?: React.BaseSyntheticEvent) => Promise<void>;
  readonly resetForm: () => void;
  readonly setSubmitError: (error: string | null) => void;
};

const defaultValues: AgentFormValues = {
  name: "",
  description: "",
  config: {
    taskType: FORM_DEFAULTS.TASK_TYPE,
    frequency: FORM_DEFAULTS.FREQUENCY,
    parameters: {},
    goal: "",
  },
};

/**
 * Custom hook for managing agent form modals (create/edit)
 * 
 * Features:
 * - React Hook Form integration
 * - Zod validation
 * - JSON formatting
 * - Error handling
 * - Pre-fill support for edit mode
 */
export const useAgentFormModal = ({
  initialAgent,
  onSubmit,
  onSuccess,
}: UseAgentFormModalOptions): UseAgentFormModalReturn => {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const form = useForm<AgentFormValues>({
    resolver: zodResolver(createAgentRequestSchema),
    defaultValues,
    mode: "onChange",
    reValidateMode: "onChange",
  });

  const parametersValue = form.watch("config.parameters");

  // Pre-fill form when editing
  useEffect(() => {
    if (initialAgent) {
      form.reset({
        name: initialAgent.name,
        description: initialAgent.description,
        config: {
          taskType: initialAgent.config.taskType,
          frequency: initialAgent.config.frequency,
          parameters: initialAgent.config.parameters || {},
          goal: initialAgent.config.goal || "",
        },
      });
    } else {
      form.reset(defaultValues);
    }
    setSubmitError(null);
  }, [initialAgent, form]);

  // Format JSON parameters
  const { formatJSON: formatJSONValue } = useJSONFormatter({
    onFormat: (formatted) => {
      try {
        const parsed = JSON.parse(formatted);
        form.setValue("config.parameters", parsed, { shouldValidate: true });
        form.clearErrors("config.parameters");
      } catch {
        // Error already handled by hook
      }
    },
  });

  const formatJSON = useCallback(() => {
    formatJSONValue(parametersValue || {});
  }, [formatJSONValue, parametersValue]);

  const handleSubmit = useCallback(
    async (e?: React.BaseSyntheticEvent): Promise<void> => {
      e?.preventDefault();

      if (isSubmitting) {
        return;
      }

      setSubmitError(null);
      setIsSubmitting(true);

      try {
        const data = form.getValues();
        await onSubmit(data);
        toast.success(
          initialAgent ? "Agente actualizado exitosamente" : "Agente creado exitosamente"
        );
        onSuccess?.();
      } catch (error) {
        const errorMessage =
          error instanceof Error
            ? error.message
            : initialAgent
            ? ERROR_MESSAGES.UPDATE_AGENT
            : ERROR_MESSAGES.CREATE_AGENT;
        setSubmitError(errorMessage);
        toast.error(errorMessage);
      } finally {
        setIsSubmitting(false);
      }
    },
    [form, onSubmit, onSuccess, isSubmitting, initialAgent]
  );

  const resetForm = useCallback(() => {
    form.reset(initialAgent ? {
      name: initialAgent.name,
      description: initialAgent.description,
      config: {
        taskType: initialAgent.config.taskType,
        frequency: initialAgent.config.frequency,
        parameters: initialAgent.config.parameters || {},
        goal: initialAgent.config.goal || "",
      },
    } : defaultValues);
    setSubmitError(null);
    setIsSubmitting(false);
  }, [form, initialAgent]);

  return {
    form,
    isSubmitting,
    submitError,
    formatJSON,
    handleSubmit,
    resetForm,
    setSubmitError,
  };
};



