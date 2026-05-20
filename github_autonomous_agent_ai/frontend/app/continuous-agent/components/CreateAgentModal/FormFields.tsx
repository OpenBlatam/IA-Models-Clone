"use client";

import React from "react";
import { FormProvider } from "react-hook-form";
import type { UseFormReturn } from "react-hook-form";
import type { AgentFormValues } from "../../utils/validation/zod-schemas";
import { NameField } from "./fields/NameField";
import { DescriptionField } from "./fields/DescriptionField";
import { TaskTypeField } from "./fields/TaskTypeField";
import { FrequencyField } from "./fields/FrequencyField";
import { ParametersField } from "./fields/ParametersField";
import { GoalField } from "./fields/GoalField";

type FormFieldsProps = {
  readonly form: UseFormReturn<AgentFormValues>;
  readonly onFormatJSON: () => void;
};

/**
 * Form fields component for CreateAgentModal
 * Uses FormProvider to share form context with individual field components
 */
export const CreateAgentFormFields = ({
  form,
  onFormatJSON,
}: FormFieldsProps): JSX.Element => {
  return (
    <FormProvider {...form}>
      <NameField />
      <DescriptionField />
      <TaskTypeField />
      <FrequencyField />
      <ParametersField onFormatJSON={onFormatJSON} />
      <GoalField />
    </FormProvider>
  );
};

