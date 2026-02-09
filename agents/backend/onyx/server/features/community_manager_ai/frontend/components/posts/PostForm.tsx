/**
 * Post Form Component
 * Reusable form component for creating and editing posts
 */

'use client';

import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Textarea } from '@/components/ui/Textarea';
import { Loading } from '@/components/ui/Loading';
import { postSchema, type PostFormData } from '@/lib/zod-schemas';
import { Post } from '@/types';
import { PLATFORMS } from '@/lib/config/constants';
import { getPlatformIcon } from '@/lib/utils';
import { Checkbox } from '@/components/ui/Checkbox';

interface PostFormProps {
  post?: Post | null;
  onSubmit: (data: PostFormData) => Promise<void>;
  onCancel: () => void;
  isLoading?: boolean;
}

/**
 * Post form component with validation
 */
export const PostForm = ({ post, onSubmit, onCancel, isLoading }: PostFormProps) => {
  const {
    register,
    handleSubmit,
    formState: { errors },
    watch,
    setValue,
  } = useForm<PostFormData & { platforms: string[] }>({
    resolver: zodResolver(postSchema),
    defaultValues: {
      content: post?.content || '',
      platforms: post?.platforms || [],
      scheduled_time: post?.scheduled_time
        ? new Date(post.scheduled_time).toISOString().slice(0, 16)
        : undefined,
      tags: post?.tags?.join(', ') || undefined,
    },
  });

  const selectedPlatforms = watch('platforms') || [];

  const handlePlatformToggle = (platform: string) => {
    const current = selectedPlatforms;
    const updated = current.includes(platform)
      ? current.filter((p) => p !== platform)
      : [...current, platform];
    setValue('platforms', updated, { shouldValidate: true });
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
      <Textarea
        label="Contenido"
        {...register('content')}
        rows={4}
        error={errors.content?.message}
        required
        placeholder="Escribe el contenido de tu post..."
      />

      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Plataformas <span className="text-red-500">*</span>
        </label>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
          {Object.values(PLATFORMS).map((platform) => (
            <label
              key={platform}
              className="flex items-center gap-2 cursor-pointer p-2 rounded-lg border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
            >
              <Checkbox
                checked={selectedPlatforms.includes(platform)}
                onCheckedChange={() => handlePlatformToggle(platform)}
                aria-label={`Seleccionar ${platform}`}
              />
              <span className="text-sm">
                {getPlatformIcon(platform)} {platform}
              </span>
            </label>
          ))}
        </div>
        {errors.platforms && (
          <p className="mt-1 text-sm text-red-600 dark:text-red-400" role="alert">
            {errors.platforms.message}
          </p>
        )}
        {selectedPlatforms.length === 0 && (
          <p className="mt-1 text-sm text-yellow-600 dark:text-yellow-400">
            Selecciona al menos una plataforma
          </p>
        )}
      </div>

      <Input
        label="Fecha Programada (opcional)"
        type="datetime-local"
        {...register('scheduled_time')}
        error={errors.scheduled_time?.message}
      />

      <Input
        label="Tags (separados por comas)"
        type="text"
        {...register('tags')}
        placeholder="tag1, tag2, tag3"
        error={errors.tags?.message}
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
          disabled={selectedPlatforms.length === 0 || isLoading}
        >
          {isLoading ? (
            <>
              <Loading size="sm" />
              <span className="ml-2">Guardando...</span>
            </>
          ) : (
            post ? 'Guardar cambios' : 'Crear post'
          )}
        </Button>
      </div>
    </form>
  );
};

