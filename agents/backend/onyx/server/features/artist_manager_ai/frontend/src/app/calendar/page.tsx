'use client';

import { useEvents, useDeleteEvent } from '@/hooks/use-calendar';
import { useArtist } from '@/hooks/use-artist';
import { useDeleteConfirmation } from '@/hooks/use-delete-confirmation';
import { formatDate, formatTime, formatDateTime, getEventTypeLabel } from '@/lib/utils';
import { PageLayout, PageHeader } from '@/components/layout';
import { LoadingSpinner, EmptyState, DataGrid, Card, CardHeader, CardTitle, CardContent, Badge, ActionButtons } from '@/components/ui';
import { Calendar } from 'lucide-react';

const CalendarPage = () => {
  const { artistId } = useArtist();
  const { data: events, isLoading } = useEvents(artistId, { days: 30 });
  const deleteEvent = useDeleteEvent(artistId);

  const handleDelete = useDeleteConfirmation<string>({
    onConfirm: async (eventId) => {
      if (eventId) {
        await deleteEvent.mutateAsync(eventId);
      }
    },
    message: '¿Estás seguro de que deseas eliminar este evento?',
    successMessage: 'Evento eliminado exitosamente',
    errorMessage: 'Error al eliminar el evento',
  });

  if (isLoading) {
    return <LoadingSpinner message="Cargando eventos..." fullScreen />;
  }

  return (
    <PageLayout>
      <PageHeader title="Calendario" actionLabel="Nuevo Evento" actionHref="/calendar/new" />

      {!events || events.length === 0 ? (
        <EmptyState
          icon={Calendar}
          title="No hay eventos programados"
          description="Comienza creando tu primer evento"
          actionLabel="Crear Evento"
          actionHref="/calendar/new"
        />
      ) : (
        <DataGrid columns={3} gap="lg">
          {events.map((event) => (
            <Card key={event.id} className="hover:shadow-lg transition-shadow">
              <CardHeader>
                <div className="flex items-start justify-between">
                  <CardTitle className="text-lg">{event.title}</CardTitle>
                  <Badge variant="info" size="sm">
                    {getEventTypeLabel(event.event_type)}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-3 mb-4">
                  <div className="text-sm">
                    <p className="font-medium text-gray-700 mb-1">Fecha y Hora</p>
                    <p className="text-gray-600">{formatDateTime(event.start_time)}</p>
                    <p className="text-gray-600 text-xs">Hasta {formatTime(event.end_time)}</p>
                  </div>
                  {event.location && (
                    <div className="text-sm">
                      <p className="font-medium text-gray-700 mb-1">Ubicación</p>
                      <p className="text-gray-600">{event.location}</p>
                    </div>
                  )}
                  {event.description && (
                    <p className="text-sm text-gray-700 line-clamp-2">{event.description}</p>
                  )}
                </div>
                <ActionButtons
                  viewHref={`/calendar/${event.id}`}
                  onDelete={() => handleDelete(event.id)}
                  deleteLabel="Eliminar evento"
                  showEdit={false}
                />
              </CardContent>
            </Card>
          ))}
        </DataGrid>
      )}
    </PageLayout>
  );
};

export default CalendarPage;

