/**
 * Validation stepper component showing validation progress
 */

'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent, Stepper } from '@/components/ui';
import { FileText, Play, CheckCircle2, BarChart3 } from 'lucide-react';
import type { ValidationRead } from '@/lib/types';

export interface ValidationStepperProps {
  validation: ValidationRead;
}

export const ValidationStepper: React.FC<ValidationStepperProps> = ({ validation }) => {
  const getCurrentStep = (): number => {
    if (validation.status === 'completed') {
      return 3;
    }
    if (validation.has_profile || validation.has_report) {
      return 2;
    }
    if (validation.status === 'running') {
      return 1;
    }
    return 0;
  };

  const steps = [
    {
      id: 'created',
      label: 'Creada',
      description: 'Validación inicializada',
      icon: <FileText className="h-5 w-5" aria-hidden="true" />,
    },
    {
      id: 'running',
      label: 'En Proceso',
      description: 'Analizando datos',
      icon: <Play className="h-5 w-5" aria-hidden="true" />,
    },
    {
      id: 'processing',
      label: 'Procesando',
      description: 'Generando perfil y reporte',
      icon: <BarChart3 className="h-5 w-5" aria-hidden="true" />,
    },
    {
      id: 'completed',
      label: 'Completada',
      description: 'Análisis finalizado',
      icon: <CheckCircle2 className="h-5 w-5" aria-hidden="true" />,
    },
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle>Progreso de la Validación</CardTitle>
      </CardHeader>
      <CardContent>
        <Stepper steps={steps} currentStep={getCurrentStep()} orientation="horizontal" />
      </CardContent>
    </Card>
  );
};



