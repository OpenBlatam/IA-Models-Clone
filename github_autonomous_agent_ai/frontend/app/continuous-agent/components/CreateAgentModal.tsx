"use client";

import React, { useEffect, useCallback } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { toast } from "sonner";
import type { CreateAgentRequest } from "../types";
import { createAgentRequestSchema, type AgentFormValues } from "../utils/validation/zod-schemas";
import { FORM_DEFAULTS } from "../constants";
import { AnimatedDialog } from "./ui/AnimatedDialog";
import { CreateAgentFormFields } from "./CreateAgentModal/FormFields";
import { FormModalFooter } from "./CreateAgentModal/FormModalFooter";
import { useJSONFormatter } from "../hooks/useJSONFormatter";
import { useModalKeyboardShortcuts } from "../hooks/useModalKeyboardShortcuts";
import { UI_MESSAGES, ERROR_MESSAGES } from "../constants/messages";

type CreateAgentModalProps = {
  readonly open: boolean;
  readonly onClose: () => void;
  readonly onCreate: (request: CreateAgentRequest) => Promise<void> | void;
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
 * Modal component for creating a new continuous agent
 * 
 * Features:
 * - React Hook Form for better performance and validation
 * - Radix UI Dialog for accessibility
 * - Framer Motion animations
 * - Sonner toast notifications
 * - Real-time validation with Zod
 * - Goal/prompt field support
 */
export const CreateAgentModal = ({
  open,
  onClose,
  onCreate,
}: CreateAgentModalProps): JSX.Element => {
  const form = useForm<AgentFormValues>({
    resolver: zodResolver(createAgentRequestSchema),
    defaultValues,
    mode: "onChange",
    reValidateMode: "onChange",
  });

  const {
    handleSubmit,
    formState: { errors, isSubmitting, isValid },
    reset,
    watch,
    setValue,
    clearErrors,
  } = form;

  const parametersValue = watch("config.parameters");

  // Reset form when modal closes
  useEffect(() => {
    if (!open) {
      reset(defaultValues);
    }
  }, [open, reset]);

  // Format JSON parameters
  const { formatJSON: formatJSONValue } = useJSONFormatter({
    onFormat: (formatted) => {
      try {
        const parsed = JSON.parse(formatted);
        setValue("config.parameters", parsed, { shouldValidate: true });
        clearErrors("config.parameters");
      } catch {
        // Error already handled by hook
      }
    },
  });

  const formatJSON = useCallback(() => {
    formatJSONValue(parametersValue || {});
  }, [formatJSONValue, parametersValue]);

  // Handle form submission
  const onSubmit = useCallback(
    async (data: AgentFormValues) => {
      try {
        await onCreate(data);
        toast.success("Agente creado exitosamente");
        onClose();
      } catch (error) {
        const errorMessage =
          error instanceof Error ? error.message : ERROR_MESSAGES.CREATE_AGENT;
        toast.error(errorMessage);
      }
    },
    [onCreate, onClose]
  );

  // Keyboard shortcuts
  useModalKeyboardShortcuts({
    open,
    onSubmit: () => handleSubmit(onSubmit)(),
    onFormatJSON: formatJSON,
    canSubmit: isValid,
  });

  return (
    <AnimatedDialog open={open} onOpenChange={onClose}>
      <h2 className="text-2xl font-bold mb-4">
        {UI_MESSAGES.CREATE_NEW_AGENT}
      </h2>

      <form onSubmit={handleSubmit(onSubmit)} className="space-y-4" noValidate>
        <CreateAgentFormFields form={form} onFormatJSON={formatJSON} />

        <FormModalFooter
          onCancel={onClose}
          isSubmitting={isSubmitting}
          isValid={isValid}
          submitLabel="Crear Agente"
          cancelLabel="Cancelar"
        />
      </form>
    </AnimatedDialog>
  );
};
