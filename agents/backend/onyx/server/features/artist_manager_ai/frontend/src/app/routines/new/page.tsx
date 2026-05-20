'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { useCreateRoutine } from '@/hooks/use-routines';
import { useArtist } from '@/hooks/use-artist';
import { routineSchema, type RoutineFormData } from '@/lib/validations';
import { RoutineType } from '@/types';
import { ROUTINE_TYPE_OPTIONS, DAY_OPTIONS } from '@/lib/constants';
import { FormLayout } from '@/components/forms/form-layout';
import { FormActions } from '@/components/forms/form-actions';
import { MultiSelectButtons } from '@/components/forms/multi-select-buttons';
import { Input, Textarea, Select } from '@/components/ui';
import { toast } from '@/lib/toast';

const NewRoutinePage = () => {
  const router = useRouter();
  const { artistId } = useArtist();
  const [selectedDays, setSelectedDays] = useState<number[]>([]);
  const createRoutine = useCreateRoutine(artistId);

  const {
    register,
    handleSubmit,
    setValue,
    formState: { errors, isSubmitting },
  } = useForm<RoutineFormData>({
    resolver: zodResolver(routineSchema),
    defaultValues: {
      routine_type: RoutineType.CUSTOM,
      priority: 5,
      days_of_week: [],
      is_required: true,
    },
  });

  const handleDayChange = (day: number) => {
    const newDays = selectedDays.includes(day)
      ? selectedDays.filter((d) => d !== day)
      : [...selectedDays, day];
    setSelectedDays(newDays);
    setValue('days_of_week', newDays, { shouldValidate: true });
  };

  const onSubmit = async (data: RoutineFormData) => {
    try {
      await createRoutine.mutateAsync({
        ...data,
        days_of_week: selectedDays,
      });
      toast.success('Rutina creada exitosamente');
      router.push('/routines');
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Error al crear la rutina');
    }
  };

  return (
    <FormLayout title="Nueva Rutina" backHref="/routines" backLabel="Volver a Rutinas">
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
              <Input
                label="Título"
                {...register('title')}
                error={errors.title?.message}
                placeholder="Ej: Ejercicio matutino"
                required
              />

              <Textarea
                label="Descripción"
                {...register('description')}
                error={errors.description?.message}
                placeholder="Descripción de la rutina"
                rows={4}
                required
              />

              <Select
                label="Tipo de Rutina"
                {...register('routine_type')}
                error={errors.routine_type?.message}
                options={ROUTINE_TYPE_OPTIONS}
                required
              />

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Input
                  label="Hora Programada"
                  type="time"
                  {...register('scheduled_time')}
                  error={errors.scheduled_time?.message}
                  required
                />

                <Input
                  label="Duración (minutos)"
                  type="number"
                  {...register('duration_minutes', { valueAsNumber: true })}
                  error={errors.duration_minutes?.message}
                  min={1}
                  max={1440}
                  required
                />
              </div>

              <Input
                label="Prioridad (1-10)"
                type="number"
                {...register('priority', { valueAsNumber: true })}
                error={errors.priority?.message}
                min={1}
                max={10}
                required
              />

              <MultiSelectButtons
                options={DAY_OPTIONS}
                selected={selectedDays}
                onChange={handleDayChange}
                label="Días de la Semana"
                error={errors.days_of_week?.message}
                required
              />

              <Textarea
                label="Notas"
                {...register('notes')}
                error={errors.notes?.message}
                placeholder="Notas adicionales"
                rows={3}
              />

              <FormActions
                submitLabel="Crear Rutina"
                cancelHref="/routines"
                isSubmitting={isSubmitting}
              />
            </form>
    </FormLayout>
  );
};

export default NewRoutinePage;

