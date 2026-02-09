'use client';

import { useRouter } from 'next/navigation';
import { useEvent, useWardrobeRecommendation, useDeleteEvent } from '@/hooks/use-calendar';
import { useArtist } from '@/hooks/use-artist';
import { useDeleteConfirmation } from '@/hooks/use-delete-confirmation';
import { formatDateTime, getEventTypeLabel } from '@/lib/utils';
import { PageLayout } from '@/components/layout/page-layout';
import { BackButton, LoadingSpinner, EmptyState, Card, CardHeader, CardTitle, CardContent, Button, DataList, Badge, Divider } from '@/components/ui';
import { Shirt, Calendar as CalendarIcon } from 'lucide-react';
import { useParams } from 'next/navigation';

const EventDetailPage = () => {
  const router = useRouter();
  const params = useParams();
  const eventId = params.id as string;
  const { artistId } = useArtist();
  const { data: event, isLoading } = useEvent(artistId, eventId);
  const { data: recommendation, isLoading: recommendationLoading } = useWardrobeRecommendation(
    artistId,
    eventId
  );
  const deleteEvent = useDeleteEvent(artistId);

  const handleDelete = useDeleteConfirmation<string>({
    onConfirm: async () => {
      await deleteEvent.mutateAsync(eventId);
      router.push('/calendar');
    },
    message: '¿Estás seguro de que deseas eliminar este evento?',
    successMessage: 'Evento eliminado exitosamente',
    errorMessage: 'Error al eliminar el evento',
  });

  if (isLoading) {
    return <LoadingSpinner message="Cargando evento..." fullScreen />;
  }

  if (!event) {
    return (
      <PageLayout>
        <EmptyState
          icon={CalendarIcon}
          title="Evento no encontrado"
          description="El evento que buscas no existe"
          actionLabel="Volver al Calendario"
          actionHref="/calendar"
        />
      </PageLayout>
    );
  }

  return (
    <PageLayout>
      <div className="max-w-3xl mx-auto">
        <BackButton href="/calendar" label="Volver al Calendario" className="mb-6" />

        <Card className="mb-6">
          <CardHeader>
            <div className="flex items-start justify-between">
              <div>
                <CardTitle className="text-2xl mb-2">{event.title}</CardTitle>
                <Badge variant="info" size="md" className="flex items-center w-fit">
                  <CalendarIcon className="w-4 h-4 mr-1" />
                  {getEventTypeLabel(event.event_type)}
                </Badge>
              </div>
              <Button variant="danger" size="sm" onClick={() => handleDelete(eventId)} aria-label="Eliminar evento">
                Eliminar
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <DataList
              items={[
                { label: 'Descripción', value: event.description },
                { label: 'Fecha y Hora de Inicio', value: formatDateTime(event.start_time) },
                { label: 'Fecha y Hora de Fin', value: formatDateTime(event.end_time) },
                ...(event.location ? [{ label: 'Ubicación', value: event.location }] : []),
                ...(event.attendees && event.attendees.length > 0
                  ? [{ label: 'Asistentes', value: <ul className="list-disc list-inside">{event.attendees.map((a, i) => <li key={i}>{a}</li>)}</ul> }]
                  : []),
                ...(event.protocol_requirements && event.protocol_requirements.length > 0
                  ? [{ label: 'Requisitos de Protocolo', value: <ul className="list-disc list-inside">{event.protocol_requirements.map((r, i) => <li key={i}>{r}</li>)}</ul> }]
                  : []),
                ...(event.wardrobe_requirements ? [{ label: 'Requisitos de Vestimenta', value: event.wardrobe_requirements }] : []),
                ...(event.notes ? [{ label: 'Notas', value: event.notes }] : []),
              ]}
            />
          </CardContent>
        </Card>

        {recommendationLoading ? (
          <Card>
            <CardContent className="text-center py-8">
              <p className="text-gray-500">Cargando recomendación...</p>
            </CardContent>
          </Card>
        ) : recommendation ? (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Shirt className="w-5 h-5 mr-2" />
                Recomendación de Vestimenta
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <p className="text-sm font-medium text-gray-700 mb-1">Código de Vestimenta</p>
                  <p className="text-gray-900">{recommendation.dress_code}</p>
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-700 mb-1">Razonamiento</p>
                  <p className="text-gray-900">{recommendation.reasoning}</p>
                </div>
                {recommendation.recommended_items && recommendation.recommended_items.length > 0 && (
                  <div>
                    <p className="text-sm font-medium text-gray-700 mb-1">Items Recomendados</p>
                    <ul className="list-disc list-inside text-gray-900">
                      {recommendation.recommended_items.map((item, index) => (
                        <li key={index}>{item}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        ) : null}
      </div>
    </PageLayout>
  );
};

export default EventDetailPage;

