"use client";

import React from "react";
import { useFormContext, Controller } from "react-hook-form";
import type { AgentFormValues } from "../../../utils/validation/zod-schemas";
import { FormField } from "../../ui/FormField";
import { Select } from "../../ui/Select";
import { TASK_TYPES } from "../../../constants";

/**
 * Task type field component
 */
export const TaskTypeField = (): JSX.Element => {
  const {
    control,
    formState: { errors },
  } = useFormContext<AgentFormValues>();

  return (
    <FormField
      label="Tipo de Tarea"
      htmlFor="agent-task-type"
      error={errors.config?.taskType?.message}
    >
      <Controller
        name="config.taskType"
        control={control}
        render={({ field }) => (
          <Select
            id="agent-task-type"
            value={field.value}
            onChange={(e) => field.onChange(e.target.value)}
            options={Object.entries(TASK_TYPES).map(([key, value]) => ({
              value,
              label: value,
            }))}
            error={errors.config?.taskType?.message}
          />
        )}
      />
    </FormField>
  );
};



