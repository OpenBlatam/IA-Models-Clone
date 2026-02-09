/**
 * Validation rating component
 */

'use client';

import React from 'react';
import { Card, CardHeader, CardTitle, CardContent, Rating } from '@/components/ui';
import { useValidations } from '@/hooks/useValidations';
import { useMemo } from 'react';

export const ValidationRating: React.FC = () => {
  const { data: validations } = useValidations();

  const averageRating = useMemo(() => {
    if (!validations || validations.length === 0) {
      return 0;
    }

    const completed = validations.filter((v) => v.status === 'completed');
    if (completed.length === 0) {
      return 0;
    }

    // Calcular rating basado en éxito y completación
    const successRate = completed.length / validations.length;
    const hasProfileRate =
      completed.filter((v) => v.has_profile).length / completed.length;
    const hasReportRate =
      completed.filter((v) => v.has_report).length / completed.length;

    // Rating de 1 a 5 basado en estos factores
    const rating = (successRate * 2 + hasProfileRate * 1.5 + hasReportRate * 1.5) / 5 * 5;
    return Math.min(5, Math.max(1, rating));
  }, [validations]);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Calificación General</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col items-center justify-center py-4">
          <Rating value={averageRating} max={5} readonly size="lg" showValue />
          <p className="text-sm text-muted-foreground mt-4 text-center">
            Basado en la tasa de éxito, completación y calidad de los análisis
          </p>
        </div>
      </CardContent>
    </Card>
  );
};



