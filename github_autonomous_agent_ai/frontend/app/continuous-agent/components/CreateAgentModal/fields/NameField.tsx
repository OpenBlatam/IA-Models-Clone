"use client";

import React from "react";
import { useFormContext } from "react-hook-form";
import type { AgentFormValues } from "../../../utils/validation/zod-schemas";
import { FormField } from "../../ui/FormField";
import { Input } from "../../ui/Input";
import { CharacterCount } from "../../ui/CharacterCount";
import { VALIDATION_LIMITS } from "../../../constants";

/**
 * Name field component with character count
 */
export const NameField = (): JSX.Element => {
  const {
    register,
    formState: { errors },
    watch,
  } = useFormContext<AgentFormValues>();

  const nameValue = watch("name") || "";

  return (
    <FormField
      label="Nombre del Agente"
      htmlFor="agent-name"
      required
      error={errors.name?.message}
    >
      <div className="relative">
        <Input
          id="agent-name"
          {...register("name")}
          placeholder="Ej: Generador de contenido automático"
          maxLength={VALIDATION_LIMITS.MAX_NAME_LENGTH}
          error={errors.name?.message}
        />
        <div className="absolute right-3 top-1/2 -translate-y-1/2">
          <CharacterCount
            current={nameValue.length}
            max={VALIDATION_LIMITS.MAX_NAME_LENGTH}
            min={VALIDATION_LIMITS.MIN_NAME_LENGTH}
            status={
              nameValue.length >= VALIDATION_LIMITS.MIN_NAME_LENGTH ? "ok" : "warning"
            }
          />
        </div>
      </div>
      {nameValue.length < VALIDATION_LIMITS.MIN_NAME_LENGTH && !errors.name && (
        <p className="text-xs text-yellow-600 mt-1" role="status">
          Mínimo {VALIDATION_LIMITS.MIN_NAME_LENGTH} caracteres
        </p>
      )}
    </FormField>
  );
};



