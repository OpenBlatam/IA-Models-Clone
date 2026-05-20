/**
 * Meme Form Component
 * Reusable form component for creating memes
 */

'use client';

import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Textarea } from '@/components/ui/Textarea';
import { Loading } from '@/components/ui/Loading';
import { useState } from 'react';
import { memeSchema, type MemeFormData } from '@/lib/zod-schemas';
import { Meme } from '@/types';

interface MemeFormProps {
  meme?: Meme | null;
  onSubmit: (data: MemeFormData & { file: FileList }) => Promise<void>;
  onCancel: () => void;
  isLoading?: boolean;
}

/**
 * Meme form component with validation
 */
export const MemeForm = ({ meme, onSubmit, onCancel, isLoading }: MemeFormProps) => {
  const {
    register,
    handleSubmit,
    formState: { errors },
    watch,
  } = useForm<MemeFormData & { file: FileList; tags?: string }>({
    resolver: zodResolver(memeSchema),
    defaultValues: {
      caption: meme?.caption || '',
      category: meme?.category || '',
      tags: meme?.tags?.join(', ') || '',
    },
  });

  const selectedFile = watch('file');
  const [preview, setPreview] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setPreview(URL.createObjectURL(file));
    }
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
      <div>
        <label htmlFor="file" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Imagen <span className="text-red-500">*</span>
        </label>
        {!meme && (
          <>
            <input
              id="file"
              type="file"
              accept="image/*"
              {...register('file', { 
                required: 'La imagen es requerida',
                onChange: handleFileChange,
              })}
              className="w-full rounded-lg border border-gray-300 dark:border-gray-700 px-3 py-2 text-sm focus:border-primary-500 focus:outline-none focus:ring-2 focus:ring-primary-500"
              aria-invalid={errors.file ? 'true' : 'false'}
              aria-describedby={errors.file ? 'file-error' : undefined}
            />
            {errors.file && (
              <p id="file-error" className="mt-1 text-sm text-red-600 dark:text-red-400" role="alert">
                {errors.file.message}
              </p>
            )}
            {preview && (
              <div className="mt-4 rounded-lg border border-gray-300 dark:border-gray-700 p-4">
                <img
                  src={preview}
                  alt="Preview"
                  className="max-h-48 mx-auto rounded"
                />
              </div>
            )}
          </>
        )}
        {meme && (
          <div className="rounded-lg border border-gray-300 dark:border-gray-700 p-4">
            <img
              src={meme.image_path}
              alt={meme.caption || 'Meme'}
              className="max-h-48 mx-auto rounded"
            />
            <p className="mt-2 text-sm text-gray-500 dark:text-gray-400 text-center">
              Imagen actual (no se puede cambiar)
            </p>
          </div>
        )}
      </div>

      <Textarea
        label="Caption (opcional)"
        {...register('caption')}
        rows={3}
        error={errors.caption?.message}
        placeholder="Añade un caption para tu meme..."
      />

      <Input
        label="Categoría (opcional)"
        type="text"
        {...register('category')}
        error={errors.category?.message}
        placeholder="Ej: funny, tech, viral"
      />

      <Input
        label="Tags (separados por comas, opcional)"
        type="text"
        {...register('tags')}
        error={errors.tags?.message}
        placeholder="funny, tech, viral"
      />

      <div className="flex justify-end gap-2 pt-4 border-t border-gray-200 dark:border-gray-700">
        <Button
          type="button"
          variant="secondary"
          onClick={onCancel}
          disabled={isLoading}
        >
          Cancelar
        </Button>
        <Button
          type="submit"
          variant="primary"
          disabled={isLoading || (!meme && !selectedFile?.[0])}
        >
          {isLoading ? (
            <>
              <Loading size="sm" />
              <span className="ml-2">Subiendo...</span>
            </>
          ) : (
            meme ? 'Guardar cambios' : 'Subir meme'
          )}
        </Button>
      </div>
    </form>
  );
};

