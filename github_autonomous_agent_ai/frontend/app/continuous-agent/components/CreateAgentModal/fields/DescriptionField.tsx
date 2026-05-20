"use client";

import React from "react";
import { useFormContext } from "react-hook-form";
import type { AgentFormValues } from "../../../utils/validation/zod-schemas";
import { FormField } from "../../ui/FormField";
import { Textarea } from "../../ui/Textarea";
import { CharacterCount } from "../../ui/CharacterCount";
import { VALIDATION_LIMITS } from "../../../constants";

/**
 * Description field component with character count
 */
export const DescriptionField = (): JSX.Element => {
  const {
    register,
    formState: { errors },
    watch,
  } = useFormContext<AgentFormValues>();

  const descriptionValue = watch("description") || "";

  return (
    <FormField
      label="Descripción"
      htmlFor="agent-description"
      required
      error={errors.description?.message}
    >
      <div className="relative">
        <Textarea
          id="agent-description"
          {...register("description")}
          placeholder="Describe qué hace este agente..."
          maxLength={VALIDATION_LIMITS.MAX_DESCRIPTION_LENGTH}
          error={errors.description?.message}
          rows={3}
        />
        <div className="absolute right-3 top-2">
          <CharacterCount
            current={descriptionValue.length}
            max={VALIDATION_LIMITS.MAX_DESCRIPTION_LENGTH}
            status={descriptionValue.length > VALIDATION_LIMITS.MAX_DESCRIPTION_LENGTH ? "error" : "ok"}
          />
        </div>
      </div>
    </FormField>
  );
};



