/**
 * Validation detail page
 */

'use client';

import React from 'react';
import { useParams } from 'next/navigation';
import { useValidation, useRunValidation } from '@/hooks/useValidations';
import { useQuery } from '@tanstack/react-query';
import { profilesApi, reportsApi } from '@/lib/api';
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  Button,
  LoadingSpinner,
  ErrorMessage,
  Tabs,
} from '@/components/ui';
import { ProfileDisplay } from '@/components/features/ProfileDisplay';
import { ReportViewer } from '@/components/features/ReportViewer';
import { ValidationActions } from '@/components/features/ValidationActions';
import { ValidationTabs } from '@/components/features/ValidationTabs';
import { ValidationDetails } from '@/components/features/ValidationDetails';
import { ValidationStepper } from '@/components/features/ValidationStepper';
import { NotesSection } from '@/components/features/NotesSection';
import { PrintButton } from '@/components/features/PrintButton';
import { ExportButton } from '@/components/features/ExportButton';
import { ConfirmDialog, Breadcrumbs } from '@/components/ui';
import { ArrowLeft, Play } from 'lucide-react';
import Link from 'next/link';
import { format } from 'date-fns';
import { useState } from 'react';

export default function ValidationDetailPage() {
  const params = useParams();
  const id = params.id as string;
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);

  const { data: validation, isLoading } = useValidation(id);
  const runValidation = useRunValidation();

  const { data: profile } = useQuery({
    queryKey: ['profile', id],
    queryFn: () => profilesApi.getByValidationId(id),
    enabled: !!validation?.has_profile && !!id,
  });

  const { data: report } = useQuery({
    queryKey: ['report', id],
    queryFn: () => reportsApi.getByValidationId(id),
    enabled: !!validation?.has_report && !!id,
  });

  const handleRunValidation = () => {
    if (!id) {
      return;
    }
    runValidation.mutate(id);
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLButtonElement>) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      handleRunValidation();
    }
  };

  const handleDelete = () => {
    // Implementar lógica de eliminación
    setShowDeleteDialog(false);
  };

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center" role="status" aria-live="polite">
        <LoadingSpinner size="lg" />
        <span className="sr-only">Cargando validación...</span>
      </div>
    );
  }

  if (!validation) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Card>
          <CardContent>
            <ErrorMessage
              message="Validación no encontrada. La validación que buscas no existe o ha sido eliminada."
              title="Validación no encontrada"
            />
            <Link href="/" className="block mt-4">
              <Button variant="outline" className="w-full">
                Volver al inicio
              </Button>
            </Link>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <Link href="/" className="flex items-center gap-2 text-muted-foreground hover:text-foreground">
              <ArrowLeft className="h-4 w-4" />
              <span>Volver</span>
            </Link>
            <div className="flex items-center gap-2">
              {validation.status === 'pending' && (
                <Button
                  onClick={handleRunValidation}
                  onKeyDown={handleKeyDown}
                  isLoading={runValidation.isPending}
                  aria-label="Ejecutar análisis de validación"
                >
                  <Play className="h-4 w-4 mr-2" aria-hidden="true" />
                  Ejecutar Análisis
                </Button>
              )}
              <ValidationActions
                validation={validation}
                onDelete={() => setShowDeleteDialog(true)}
              />
              <PrintButton />
              {validation.has_report && (
                <ExportButton validationId={id} fileName={`validacion-${id.slice(0, 8)}`} />
              )}
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <Breadcrumbs
          items={[
            { label: 'Validaciones', href: '/' },
            { label: `Validación ${id.slice(0, 8)}` },
          ]}
        />

        <div className="space-y-6 mt-6">
          <ValidationStepper validation={validation} />
          
          <ValidationDetails validation={validation} />

          <ValidationTabs validation={validation} />

          <NotesSection validationId={validation.id} />
        </div>
      </main>

      <ConfirmDialog
        isOpen={showDeleteDialog}
        onClose={() => setShowDeleteDialog(false)}
        onConfirm={handleDelete}
        title="Eliminar Validación"
        message="¿Estás seguro de que deseas eliminar esta validación? Esta acción no se puede deshacer."
        confirmText="Eliminar"
        cancelText="Cancelar"
        variant="destructive"
      />
    </div>
  );
}

