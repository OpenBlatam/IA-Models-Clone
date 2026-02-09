"use client";

import React from "react";
import { useFormContext } from "react-hook-form";
import type { AgentFormValues } from "../../../utils/validation/zod-schemas";
import { FormField } from "../../ui/FormField";
import { Textarea } from "../../ui/Textarea";
import { CharacterCount } from "../../ui/CharacterCount";
import { VALIDATION_LIMITS } from "../../../constants";

/**
 * Goal field component with character count
 */
export const GoalField = (): JSX.Element => {
  const {
    register,
    formState: { errors },
    watch,
  } = useFormContext<AgentFormValues>();

  const goalValue = watch("config.goal") || "";

  return (
    <FormField
      label="Objetivo/Prompt (Opcional)"
      htmlFor="agent-goal"
      error={errors.config?.goal?.message}
    >
      <div className="relative">
        <Textarea
          id="agent-goal"
          {...register("config.goal")}
          placeholder="Define el objetivo o prompt del agente..."
          maxLength={VALIDATION_LIMITS.MAX_GOAL_LENGTH}
          error={errors.config?.goal?.message}
          rows={6}
        />
        {goalValue.length > 0 && (
          <div className="absolute right-3 top-2">
            <CharacterCount
              current={goalValue.length}
              max={VALIDATION_LIMITS.MAX_GOAL_LENGTH}
              status={goalValue.length > VALIDATION_LIMITS.MAX_GOAL_LENGTH ? "error" : "ok"}
            />
          </div>
        )}
      </div>
    </FormField>
  );
};



