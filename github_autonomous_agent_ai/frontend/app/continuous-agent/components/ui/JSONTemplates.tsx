"use client";

import React from "react";
import { QuickButton } from "./QuickButton";
import { cn } from "../../utils/classNames";

type JSONTemplate = {
  readonly name: string;
  readonly value: string;
  readonly description?: string;
};

type JSONTemplatesProps = {
  readonly templates: readonly JSONTemplate[];
  readonly onSelect: (template: JSONTemplate) => void;
  readonly className?: string;
};

export const JSONTemplates = ({
  templates,
  onSelect,
  className,
}: JSONTemplatesProps): JSX.Element => {
  if (templates.length === 0) {
    return null;
  }

  return (
    <div
      className={cn(
        "border border-gray-200 rounded-lg p-3 bg-gray-50 space-y-2",
        className
      )}
      role="group"
      aria-label="Plantillas JSON"
    >
      <p className="text-xs font-medium text-gray-700">Plantillas rápidas:</p>
      <div className="flex flex-wrap gap-2">
        {templates.map((template, index) => (
          <QuickButton
            key={index}
            label={template.name}
            onClick={() => onSelect(template)}
            title={template.description || template.name}
            ariaLabel={`Aplicar plantilla ${template.name}`}
            variant="secondary"
          />
        ))}
      </div>
    </div>
  );
};

