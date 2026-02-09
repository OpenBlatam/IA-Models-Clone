/**
 * Component to compare multiple validations
 */

'use client';

import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent, Button, Select, Badge } from '@/components/ui';
import { useValidations } from '@/hooks/useValidations';
import { PersonalityChart } from './PersonalityChart';
import { Compare, X } from 'lucide-react';
import type { ValidationRead } from '@/lib/types';

export const ComparisonView: React.FC = () => {
  const { data: validations } = useValidations();
  const [selectedIds, setSelectedIds] = useState<string[]>([]);

  if (!validations || validations.length < 2) {
    return (
      <Card>
        <CardContent className="py-12">
          <p className="text-center text-muted-foreground">
            Se necesitan al menos 2 validaciones para comparar
          </p>
        </CardContent>
      </Card>
    );
  }

  const handleSelectValidation = (id: string) => {
    if (selectedIds.includes(id)) {
      setSelectedIds(selectedIds.filter((selectedId) => selectedId !== id));
    } else if (selectedIds.length < 3) {
      setSelectedIds([...selectedIds, id]);
    }
  };

  const handleRemoveValidation = (id: string) => {
    setSelectedIds(selectedIds.filter((selectedId) => selectedId !== id));
  };

  const handleClearAll = () => {
    setSelectedIds([]);
  };

  const selectedValidations = validations.filter((v) => selectedIds.includes(v.id));

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Comparar Validaciones</CardTitle>
            {selectedIds.length > 0 && (
              <Button variant="ghost" size="sm" onClick={handleClearAll}>
                Limpiar selección
              </Button>
            )}
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <p className="text-sm text-muted-foreground">
              Selecciona hasta 3 validaciones para comparar (máximo 3)
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {validations.map((validation) => {
                const isSelected = selectedIds.includes(validation.id);
                const canSelect = !isSelected && selectedIds.length < 3;

                return (
                  <button
                    key={validation.id}
                    type="button"
                    onClick={() => handleSelectValidation(validation.id)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        handleSelectValidation(validation.id);
                      }
                    }}
                    disabled={!canSelect && !isSelected}
                    className={`
                      p-4 border-2 rounded-lg text-left transition-all
                      ${isSelected ? 'border-primary bg-primary/10' : 'border-input hover:border-primary/50'}
                      ${!canSelect && !isSelected ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
                    `}
                    aria-pressed={isSelected}
                    aria-label={`${isSelected ? 'Deseleccionar' : 'Seleccionar'} validación ${validation.id.slice(0, 8)}`}
                    tabIndex={canSelect || isSelected ? 0 : -1}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-medium">
                        Validación #{validation.id.slice(0, 8)}
                      </span>
                      {isSelected && (
                        <Badge variant="primary">Seleccionada</Badge>
                      )}
                    </div>
                    <div className="text-sm text-muted-foreground">
                      <p>Estado: {validation.status}</p>
                      <p>Plataformas: {validation.connected_platforms.length}</p>
                    </div>
                  </button>
                );
              })}
            </div>

            {selectedValidations.length > 0 && (
              <div className="mt-6 space-y-4">
                <h3 className="text-lg font-semibold">Validaciones Seleccionadas</h3>
                <div className="flex flex-wrap gap-2">
                  {selectedValidations.map((validation) => (
                    <div
                      key={validation.id}
                      className="flex items-center gap-2 px-3 py-2 bg-accent rounded-md"
                    >
                      <span className="text-sm font-medium">
                        #{validation.id.slice(0, 8)}
                      </span>
                      <button
                        type="button"
                        onClick={() => handleRemoveValidation(validation.id)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' || e.key === ' ') {
                            e.preventDefault();
                            handleRemoveValidation(validation.id);
                          }
                        }}
                        className="p-1 hover:bg-destructive/10 rounded"
                        aria-label={`Remover validación ${validation.id.slice(0, 8)}`}
                        tabIndex={0}
                      >
                        <X className="h-4 w-4" aria-hidden="true" />
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {selectedValidations.length >= 2 && (
        <Card>
          <CardHeader>
            <CardTitle>Comparación</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <p className="text-sm text-muted-foreground">
                Comparando {selectedValidations.length} validaciones
              </p>
              {/* Aquí se pueden agregar gráficos comparativos */}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};




