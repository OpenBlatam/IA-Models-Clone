"use client";

import React, { useEffect } from "react";
import type { ContinuousAgent } from "../types";
import type { AgentFormValues } from "../utils/validation/zod-schemas";
import { AnimatedDialog } from "./ui/AnimatedDialog";
import { ErrorAlert } from "./ui/ErrorAlert";
import { EditAgentFormFields } from "./EditAgentModal/EditAgentFormFields";
import { FormModalFooter } from "./CreateAgentModal/FormModalFooter";
import { useAgentFormModal } from "../hooks/useAgentFormModal";
import { useModalKeyboardShortcuts } from "../hooks/useModalKeyboardShortcuts";
import { updateAgent } from "../services/agentService";

type EditAgentModalProps = {
  readonly open: boolean;
  readonly agent: ContinuousAgent | null;
  readonly onClose: () => void;
  readonly onUpdate: (agent: ContinuousAgent) => Promise<void> | void;
};

/**
 * Modal component for editing an existing continuous agent
 * 
 * Features:
 * - React Hook Form for better performance and validation
 * - Radix UI Dialog for accessibility
 * - Framer Motion animations
 * - Pre-fills form with existing agent data
 * - Real-time validation with Zod
 */
export const EditAgentModal = ({
  open,
  agent,
  onClose,
  onUpdate,
}: EditAgentModalProps): JSX.Element | null => {
  const {
    form,
    isSubmitting,
    submitError,
    formatJSON,
    handleSubmit,
    resetForm,
    setSubmitError,
  } = useAgentFormModal({
    initialAgent: agent,
    onSubmit: async (data: AgentFormValues) => {
      if (!agent) {
        throw new Error("Agent is required");
      }
      const updatedAgent = await updateAgent(agent.id, data);
      await onUpdate(updatedAgent);
      onClose();
    },
    onSuccess: () => {
      onClose();
    },
  });

  const {
    formState: { errors, isValid },
    watch,
    handleSubmit: formHandleSubmit,
  } = form;

  const parametersValue = watch("config.parameters");

  // Reset form when modal closes
  useEffect(() => {
    if (!open) {
      resetForm();
    }
  }, [open, resetForm]);

  // Keyboard shortcuts
  useModalKeyboardShortcuts({
    open,
    onSubmit: handleSubmit,
    onFormatJSON: formatJSON,
    canSubmit: isValid,
  });

  if (!agent) {
    return null;
  }

  return (
    <AnimatedDialog open={open} onOpenChange={onClose}>
      <h2 className="text-2xl font-bold mb-4">
        Editar Agente: {agent.name}
      </h2>

      {submitError && (
        <div
          className="mb-4 animate-in slide-in-from-top-2 duration-300"
          role="alert"
          aria-live="assertive"
        >
          <ErrorAlert message={submitError} />
        </div>
      )}

      <form onSubmit={formHandleSubmit(handleSubmit)} className="space-y-4" noValidate>
        <EditAgentFormFields form={form} onFormatJSON={formatJSON} />

        <FormModalFooter
          onCancel={onClose}
          isSubmitting={isSubmitting}
          isValid={isValid}
          submitLabel="Guardar Cambios"
          cancelLabel="Cancelar"
        />
      </form>
    </AnimatedDialog>
  );
};
