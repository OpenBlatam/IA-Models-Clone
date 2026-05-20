'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { useCreateWardrobeItem } from '@/hooks/use-wardrobe';
import { useArtist } from '@/hooks/use-artist';
import { wardrobeItemSchema, type WardrobeItemFormData } from '@/lib/validations';
import { Season, DressCode } from '@/types';
import { SEASON_OPTIONS, DRESS_CODE_OPTIONS } from '@/lib/constants';
import { FormLayout } from '@/components/forms/form-layout';
import { FormActions } from '@/components/forms/form-actions';
import { MultiSelectButtons } from '@/components/forms/multi-select-buttons';
import { Input, Textarea, Select } from '@/components/ui';
import { toast } from '@/lib/toast';

const NewWardrobeItemPage = () => {
  const router = useRouter();
  const { artistId } = useArtist();
  const [selectedDressCodes, setSelectedDressCodes] = useState<DressCode[]>([]);
  const createItem = useCreateWardrobeItem(artistId);

  const {
    register,
    handleSubmit,
    setValue,
    formState: { errors, isSubmitting },
  } = useForm<WardrobeItemFormData>({
    resolver: zodResolver(wardrobeItemSchema),
    defaultValues: {
      season: Season.ALL_SEASON,
      dress_codes: [],
    },
  });

  const handleDressCodeChange = (value: string | number) => {
    const dressCode = value as DressCode;
    const newCodes = selectedDressCodes.includes(dressCode)
      ? selectedDressCodes.filter((dc) => dc !== dressCode)
      : [...selectedDressCodes, dressCode];
    setSelectedDressCodes(newCodes);
    setValue('dress_codes', newCodes, { shouldValidate: true });
  };

  const onSubmit = async (data: WardrobeItemFormData) => {
    try {
      await createItem.mutateAsync({
        ...data,
        dress_codes: selectedDressCodes,
      });
      toast.success('Item creado exitosamente');
      router.push('/wardrobe');
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Error al crear el item');
    }
  };

  return (
    <FormLayout title="Nuevo Item de Guardarropa" backHref="/wardrobe" backLabel="Volver al Guardarropa">
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
              <Input
                label="Nombre"
                {...register('name')}
                error={errors.name?.message}
                placeholder="Ej: Camisa negra elegante"
                required
              />

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Input
                  label="Categoría"
                  {...register('category')}
                  error={errors.category?.message}
                  placeholder="Ej: shirt, pants, shoes"
                  required
                />

                <Input
                  label="Color"
                  {...register('color')}
                  error={errors.color?.message}
                  placeholder="Ej: black, white"
                  required
                />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Input
                  label="Marca"
                  {...register('brand')}
                  error={errors.brand?.message}
                  placeholder="Ej: Designer Brand"
                />

                <Input
                  label="Talla"
                  {...register('size')}
                  error={errors.size?.message}
                  placeholder="Ej: M, L, XL"
                />
              </div>

              <Select
                label="Estación"
                {...register('season')}
                error={errors.season?.message}
                options={SEASON_OPTIONS}
                required
              />

              <MultiSelectButtons
                options={DRESS_CODE_OPTIONS}
                selected={selectedDressCodes}
                onChange={handleDressCodeChange}
                label="Códigos de Vestimenta"
                error={errors.dress_codes?.message}
                required
              />

              <Input
                label="URL de Imagen"
                type="url"
                {...register('image_url')}
                error={errors.image_url?.message}
                placeholder="https://example.com/image.jpg"
              />

              <Textarea
                label="Notas"
                {...register('notes')}
                error={errors.notes?.message}
                placeholder="Notas adicionales"
                rows={3}
              />

              <FormActions
                submitLabel="Crear Item"
                cancelHref="/wardrobe"
                isSubmitting={isSubmitting}
              />
            </form>
    </FormLayout>
  );
};

export default NewWardrobeItemPage;

