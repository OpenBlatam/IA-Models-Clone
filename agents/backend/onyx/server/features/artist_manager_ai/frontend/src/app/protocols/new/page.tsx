'use client';

import { useRouter } from 'next/navigation';
import { useForm, useFieldArray } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { useCreateProtocol } from '@/hooks/use-protocols';
import { useArtist } from '@/hooks/use-artist';
import { protocolSchema, type ProtocolFormData } from '@/lib/validations';
import { PROTOCOL_CATEGORY_OPTIONS, PROTOCOL_PRIORITY_OPTIONS } from '@/lib/constants';
import { FormLayout } from '@/components/forms/form-layout';
import { FormActions } from '@/components/forms/form-actions';
import { FieldArrayInput } from '@/components/forms/field-array-input';
import { Input, Textarea, Select } from '@/components/ui';
import { toast } from '@/lib/toast';

const NewProtocolPage = () => {
  const router = useRouter();
  const { artistId } = useArtist();
  const createProtocol = useCreateProtocol(artistId);

  const {
    register,
    handleSubmit,
    control,
    formState: { errors, isSubmitting },
  } = useForm<ProtocolFormData>({
    resolver: zodResolver(protocolSchema),
    defaultValues: {
      rules: [''],
      do_s: [],
      dont_s: [],
    },
  });

  const { fields: ruleFields, append: appendRule, remove: removeRule } = useFieldArray({
    control,
    name: 'rules',
  });

  const { fields: doFields, append: appendDo, remove: removeDo } = useFieldArray({
    control,
    name: 'do_s',
  });

  const { fields: dontFields, append: appendDont, remove: removeDont } = useFieldArray({
    control,
    name: 'dont_s',
  });

  const onSubmit = async (data: ProtocolFormData) => {
    try {
      await createProtocol.mutateAsync({
        ...data,
        rules: data.rules.filter((rule) => rule.trim() !== ''),
      });
      toast.success('Protocolo creado exitosamente');
      router.push('/protocols');
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Error al crear el protocolo');
    }
  };

  return (
    <FormLayout title="Nuevo Protocolo" backHref="/protocols" backLabel="Volver a Protocolos">
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
        <Input
          label="Título"
          {...register('title')}
          error={errors.title?.message}
          placeholder="Ej: Protocolo de Redes Sociales"
          required
        />

        <Textarea
          label="Descripción"
          {...register('description')}
          error={errors.description?.message}
          placeholder="Descripción del protocolo"
          rows={4}
          required
        />

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Select
            label="Categoría"
            {...register('category')}
            error={errors.category?.message}
            options={PROTOCOL_CATEGORY_OPTIONS}
            required
          />

          <Select
            label="Prioridad"
            {...register('priority')}
            error={errors.priority?.message}
            options={PROTOCOL_PRIORITY_OPTIONS}
            required
          />
        </div>

        <FieldArrayInput
          fields={ruleFields}
          register={(index) => register(`rules.${index}` as const)}
          onAppend={() => appendRule('')}
          onRemove={removeRule}
          label="Reglas"
          placeholder="Regla del protocolo"
          error={
            errors.rules && typeof errors.rules === 'object' && !Array.isArray(errors.rules)
              ? errors.rules.message
              : undefined
          }
          required
          minFields={1}
        />

        <FieldArrayInput
          fields={doFields}
          register={(index) => register(`do_s.${index}` as const)}
          onAppend={() => appendDo('')}
          onRemove={removeDo}
          label="Cosas a Hacer"
          placeholder="Cosa a hacer"
          minFields={0}
        />

        <FieldArrayInput
          fields={dontFields}
          register={(index) => register(`dont_s.${index}` as const)}
          onAppend={() => appendDont('')}
          onRemove={removeDont}
          label="Cosas a Evitar"
          placeholder="Cosa a evitar"
          minFields={0}
        />

        <Textarea
          label="Contexto"
          {...register('context')}
          error={errors.context?.message}
          placeholder="Contexto adicional del protocolo"
          rows={3}
        />

        <Textarea
          label="Notas"
          {...register('notes')}
          error={errors.notes?.message}
          placeholder="Notas adicionales"
          rows={3}
        />

        <FormActions
          submitLabel="Crear Protocolo"
          cancelHref="/protocols"
          isSubmitting={isSubmitting}
        />
      </form>
    </FormLayout>
  );
};

export default NewProtocolPage;

