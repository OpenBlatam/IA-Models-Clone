'use client';

import { useRouter } from 'next/navigation';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { useCreateEvent } from '@/hooks/use-calendar';
import { useArtist } from '@/hooks/use-artist';
import { eventSchema, type EventFormData } from '@/lib/validations';
import { EventType } from '@/types';
import { EVENT_TYPE_OPTIONS } from '@/lib/constants';
import { FormLayout } from '@/components/forms/form-layout';
import { FormActions } from '@/components/forms/form-actions';
import { Input, Textarea, Select } from '@/components/ui';
import { toast } from '@/lib/toast';

const NewEventPage = () => {
  const router = useRouter();
  const { artistId } = useArtist();
  const createEvent = useCreateEvent(artistId);

  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm<EventFormData>({
    resolver: zodResolver(eventSchema),
    defaultValues: {
      event_type: EventType.OTHER,
      attendees: [],
      protocol_requirements: [],
    },
  });

  const onSubmit = async (data: EventFormData) => {
    try {
      await createEvent.mutateAsync(data);
      toast.success('Evento creado exitosamente');
      router.push('/calendar');
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Error al crear el evento');
    }
  };

  return (
    <FormLayout title="Nuevo Evento" backHref="/calendar" backLabel="Volver al Calendario">
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
              <Input
                label="Título"
                {...register('title')}
                error={errors.title?.message}
                placeholder="Ej: Concierto en el Estadio"
                required
              />

              <Textarea
                label="Descripción"
                {...register('description')}
                error={errors.description?.message}
                placeholder="Descripción del evento"
                rows={4}
                required
              />

              <Select
                label="Tipo de Evento"
                {...register('event_type')}
                error={errors.event_type?.message}
                options={EVENT_TYPE_OPTIONS}
                required
              />

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Input
                  label="Fecha y Hora de Inicio"
                  type="datetime-local"
                  {...register('start_time')}
                  error={errors.start_time?.message}
                  required
                />

                <Input
                  label="Fecha y Hora de Fin"
                  type="datetime-local"
                  {...register('end_time')}
                  error={errors.end_time?.message}
                  required
                />
              </div>

              <Input
                label="Ubicación"
                {...register('location')}
                error={errors.location?.message}
                placeholder="Ej: Estadio Nacional"
              />

              <Textarea
                label="Requisitos de Protocolo"
                {...register('protocol_requirements')}
                error={errors.protocol_requirements?.message}
                placeholder="Cada requisito en una línea"
                rows={3}
              />

              <Textarea
                label="Requisitos de Vestimenta"
                {...register('wardrobe_requirements')}
                error={errors.wardrobe_requirements?.message}
                placeholder="Descripción de la vestimenta requerida"
                rows={2}
              />

              <Textarea
                label="Notas"
                {...register('notes')}
                error={errors.notes?.message}
                placeholder="Notas adicionales"
                rows={3}
              />

              <FormActions
                submitLabel="Crear Evento"
                cancelHref="/calendar"
                isSubmitting={isSubmitting}
              />
            </form>
    </FormLayout>
  );
};

export default NewEventPage;

