import React from "react";
import { FormField } from "../ui/FormField";
import { Select } from "../ui/Select";
import { TASK_TYPE_OPTIONS } from "../../constants";
import type { UseAgentFormReturn } from "../../hooks/useAgentForm";

type AgentTaskTypeFieldProps = {
  readonly form: UseAgentFormReturn;
  readonly onErrorChange?: (error: string | null) => void;
};

export const AgentTaskTypeField = ({
  form,
  onErrorChange,
}: AgentTaskTypeFieldProps): JSX.Element => {
  const handleChange = (event: React.ChangeEvent<HTMLSelectElement>): void => {
    form.setTaskType(event.target.value as typeof form.taskType);
    onErrorChange?.(null);
  };

  return (
    <FormField label="Tipo de Tarea" htmlFor="task-type" required>
      <Select
        id="task-type"
        value={form.taskType}
        onChange={handleChange}
        options={TASK_TYPE_OPTIONS}
        required
        ariaLabel="Seleccionar tipo de tarea"
      />
    </FormField>
  );
};





