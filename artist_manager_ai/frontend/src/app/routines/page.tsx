'use client';

import { useRoutines, usePendingRoutines, useCompleteRoutine, useDeleteRoutine } from '@/hooks/use-routines';
import { useArtist } from '@/hooks/use-artist';
import { useDeleteConfirmation } from '@/hooks/use-delete-confirmation';
import { formatTime, getRoutineTypeLabel, getDaysOfWeekNames } from '@/lib/utils';
import { PageLayout, PageHeader } from '@/components/layout';
import { LoadingSpinner, EmptyState, Section, Button, Card, CardHeader, CardTitle, CardContent, ActionButtons } from '@/components/ui';
import { toast } from '@/lib/toast';
import { Clock, CheckCircle } from 'lucide-react';

const RoutinesPage = () => {
  const { artistId } = useArtist();
  const { data: routines, isLoading } = useRoutines(artistId);
  const { data: pendingRoutines } = usePendingRoutines(artistId);
  const completeRoutine = useCompleteRoutine(artistId);
  const deleteRoutine = useDeleteRoutine(artistId);

  const handleComplete = async (taskId: string) => {
    try {
      await completeRoutine.mutateAsync({
        taskId,
        completion: { status: 'completed' },
      });
      toast.success('Rutina completada exitosamente');
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Error al completar la rutina');
    }
  };

  const handleDelete = useDeleteConfirmation<string>({
    onConfirm: async (taskId) => {
      if (taskId) {
        await deleteRoutine.mutateAsync(taskId);
      }
    },
    message: '¿Estás seguro de que deseas eliminar esta rutina?',
    successMessage: 'Rutina eliminada exitosamente',
    errorMessage: 'Error al eliminar la rutina',
  });

  if (isLoading) {
    return <LoadingSpinner message="Cargando rutinas..." fullScreen />;
  }

  return (
    <PageLayout>
      <PageHeader title="Rutinas" actionLabel="Nueva Rutina" actionHref="/routines/new" />

      {pendingRoutines && pendingRoutines.length > 0 && (
        <Section title="Rutinas Pendientes">
          <DataGrid columns={3} gap="md">
            {pendingRoutines.map((routine) => (
              <Card key={routine.id} className="border-orange-200 bg-orange-50">
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <CardTitle>{routine.title}</CardTitle>
                    <Badge variant="warning" size="sm">Pendiente</Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2 mb-4">
                    <div className="flex items-center text-sm text-gray-600">
                      <Clock className="w-4 h-4 mr-2" />
                      {formatTime(routine.scheduled_time)}
                    </div>
                    <p className="text-sm text-gray-600">
                      Duración: {routine.duration_minutes} minutos
                    </p>
                  </div>
                  <Button
                    variant="primary"
                    size="sm"
                    onClick={() => handleComplete(routine.id)}
                    className="w-full"
                  >
                    <CheckCircle className="w-4 h-4 mr-2" />
                    Completar
                  </Button>
                </CardContent>
              </Card>
            ))}
          </DataGrid>
        </Section>
      )}

      <Section title="Todas las Rutinas">
        {!routines || routines.length === 0 ? (
          <EmptyState
            icon={Clock}
            title="No hay rutinas creadas"
            description="Comienza creando tu primera rutina"
            actionLabel="Crear Rutina"
            actionHref="/routines/new"
          />
        ) : (
          <DataGrid columns={3} gap="lg">
            {routines.map((routine) => (
              <Card key={routine.id} className="hover:shadow-lg transition-shadow">
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <CardTitle className="text-lg">{routine.title}</CardTitle>
                    <Badge variant="info" size="sm">
                      {getRoutineTypeLabel(routine.routine_type)}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3 mb-4">
                    <div className="flex items-center text-sm text-gray-600">
                      <Clock className="w-4 h-4 mr-2" />
                      {formatTime(routine.scheduled_time)}
                    </div>
                    <div className="text-sm">
                      <p className="font-medium text-gray-700 mb-1">Duración</p>
                      <p className="text-gray-600">{routine.duration_minutes} minutos</p>
                    </div>
                    <div className="text-sm">
                      <p className="font-medium text-gray-700 mb-1">Prioridad</p>
                      <p className="text-gray-600">{routine.priority}/10</p>
                    </div>
                    <div className="text-sm">
                      <p className="font-medium text-gray-700 mb-1">Días</p>
                      <p className="text-gray-600">{getDaysOfWeekNames(routine.days_of_week).join(', ')}</p>
                    </div>
                    {routine.description && (
                      <p className="text-sm text-gray-700 line-clamp-2">{routine.description}</p>
                    )}
                  </div>
                  <div className="flex gap-2">
                    <Button
                      variant="primary"
                      size="sm"
                      onClick={() => handleComplete(routine.id)}
                      className="flex-1"
                    >
                      <CheckCircle className="w-4 h-4 mr-2" />
                      Completar
                    </Button>
                    <ActionButtons
                      onDelete={() => handleDelete(routine.id)}
                      deleteLabel="Eliminar rutina"
                      showView={false}
                    />
                  </div>
                </CardContent>
              </Card>
            ))}
          </DataGrid>
        )}
      </Section>
    </PageLayout>
  );
};

export default RoutinesPage;

