"use client";

import React from "react";
import type { UseFormReturn } from "react-hook-form";
import type { AgentFormValues } from "../../utils/validation/zod-schemas";
import { CreateAgentFormFields } from "../CreateAgentModal/FormFields";

type EditAgentFormFieldsProps = {
  readonly form: UseFormReturn<AgentFormValues>;
  readonly onFormatJSON: () => void;
};

/**
 * Form fields component for EditAgentModal
 * Reuses CreateAgentFormFields for consistency
 */
export const EditAgentFormFields = ({
  form,
  onFormatJSON,
}: EditAgentFormFieldsProps): JSX.Element => {
  return <CreateAgentFormFields form={form} onFormatJSON={onFormatJSON} />;
};

