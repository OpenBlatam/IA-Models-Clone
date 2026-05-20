'use client';

import { UseFormRegisterReturn } from 'react-hook-form';

interface FormOptionsProps {
  registerSafety: UseFormRegisterReturn;
  registerTools: UseFormRegisterReturn;
  registerMaterials: UseFormRegisterReturn;
}

export const FormOptions = ({
  registerSafety,
  registerTools,
  registerMaterials,
}: FormOptionsProps): JSX.Element => {
  return (
    <div className="flex items-center space-x-4">
      <label className="flex items-center space-x-2">
        <input
          type="checkbox"
          {...registerSafety}
          className="rounded"
          aria-label="Incluir advertencias de seguridad"
        />
        <span className="text-sm">Incluir advertencias de seguridad</span>
      </label>
      <label className="flex items-center space-x-2">
        <input
          type="checkbox"
          {...registerTools}
          className="rounded"
          aria-label="Incluir herramientas"
        />
        <span className="text-sm">Incluir herramientas</span>
      </label>
      <label className="flex items-center space-x-2">
        <input
          type="checkbox"
          {...registerMaterials}
          className="rounded"
          aria-label="Incluir materiales"
        />
        <span className="text-sm">Incluir materiales</span>
      </label>
    </div>
  );
};

