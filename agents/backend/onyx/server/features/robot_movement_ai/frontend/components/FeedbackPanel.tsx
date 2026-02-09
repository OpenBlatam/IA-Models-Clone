'use client';

import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { MessageSquare, Send, Star, AlertCircle } from 'lucide-react';
import { toast } from '@/lib/utils/toast';
import { motion } from 'framer-motion';

const feedbackSchema = z.object({
  type: z.enum(['bug', 'feature', 'improvement', 'other']),
  title: z.string().min(3, 'El título debe tener al menos 3 caracteres').max(100, 'El título es demasiado largo'),
  description: z.string().min(10, 'La descripción debe tener al menos 10 caracteres').max(1000, 'La descripción es demasiado larga'),
  rating: z.number().min(0).max(5),
});

type FeedbackFormData = z.infer<typeof feedbackSchema>;

export default function FeedbackPanel() {
  const {
    register,
    handleSubmit,
    setValue,
    watch,
    formState: { errors, isSubmitting },
    reset,
  } = useForm<FeedbackFormData>({
    resolver: zodResolver(feedbackSchema),
    defaultValues: {
      type: 'feature',
      title: '',
      description: '',
      rating: 0,
    },
  });

  const feedbackType = watch('type');
  const rating = watch('rating');

  const onSubmit = async (data: FeedbackFormData) => {
    try {
      // Simulate feedback submission
      await new Promise((resolve) => setTimeout(resolve, 2000));
      toast.success('¡Gracias por tu feedback!');
      reset();
    } catch (error: any) {
      toast.error(`Error: ${error.message || 'Failed to submit feedback'}`);
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg p-6 border border-gray-200 shadow-sm">
        <div className="flex items-center gap-2 mb-6">
          <MessageSquare className="w-5 h-5 text-tesla-blue" />
          <h3 className="text-lg font-semibold text-tesla-black">Enviar Feedback</h3>
        </div>

        <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
          {/* Type Selection */}
          <div>
            <label className="block text-sm font-medium text-tesla-black mb-3">
              Tipo de Feedback
            </label>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {(['bug', 'feature', 'improvement', 'other'] as const).map((type) => (
                <motion.button
                  key={type}
                  type="button"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => setValue('type', type)}
                  className={`p-4 rounded-md border-2 transition-all min-h-[44px] ${
                    feedbackType === type
                      ? 'border-tesla-blue bg-blue-50 text-tesla-blue'
                      : 'border-gray-300 bg-white text-tesla-black hover:border-gray-400'
                  }`}
                  aria-label={`Seleccionar tipo ${type}`}
                >
                  <p className="text-sm font-medium capitalize">{type}</p>
                </motion.button>
              ))}
            </div>
            {errors.type && (
              <p className="mt-1 text-sm text-red-600">{errors.type.message}</p>
            )}
          </div>

          {/* Rating */}
          <div>
            <label className="block text-sm font-medium text-tesla-black mb-3">
              Calificación
            </label>
            <div className="flex gap-2">
              {[1, 2, 3, 4, 5].map((star) => (
                <motion.button
                  key={star}
                  type="button"
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => setValue('rating', star)}
                  className="p-2 transition-transform min-h-[44px] min-w-[44px] flex items-center justify-center"
                  aria-label={`Calificar con ${star} estrella${star > 1 ? 's' : ''}`}
                >
                  <Star
                    className={`w-6 h-6 transition-colors ${
                      star <= rating ? 'text-yellow-500 fill-yellow-500' : 'text-tesla-gray-light'
                    }`}
                  />
                </motion.button>
              ))}
            </div>
          </div>

          {/* Title */}
          <div>
            <label htmlFor="title" className="block text-sm font-medium text-tesla-black mb-2">
              Título
            </label>
            <input
              id="title"
              type="text"
              {...register('title')}
              placeholder="Resumen breve..."
              className={`w-full px-4 py-3 bg-white border rounded-md text-tesla-black focus:outline-none focus:ring-2 focus:ring-tesla-blue focus:border-transparent transition-all ${
                errors.title ? 'border-red-300' : 'border-gray-300'
              }`}
              aria-label="Título del feedback"
              aria-invalid={errors.title ? 'true' : 'false'}
            />
            {errors.title && (
              <p className="mt-1 text-sm text-red-600">{errors.title.message}</p>
            )}
          </div>

          {/* Description */}
          <div>
            <label htmlFor="description" className="block text-sm font-medium text-tesla-black mb-2">
              Descripción
            </label>
            <textarea
              id="description"
              {...register('description')}
              placeholder="Describe tu feedback en detalle..."
              rows={6}
              className={`w-full px-4 py-3 bg-white border rounded-md text-tesla-black focus:outline-none focus:ring-2 focus:ring-tesla-blue focus:border-transparent resize-none transition-all ${
                errors.description ? 'border-red-300' : 'border-gray-300'
              }`}
              aria-label="Descripción del feedback"
              aria-invalid={errors.description ? 'true' : 'false'}
            />
            {errors.description && (
              <p className="mt-1 text-sm text-red-600">{errors.description.message}</p>
            )}
          </div>

          {/* Submit */}
          <motion.button
            type="submit"
            disabled={isSubmitting}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-tesla-blue hover:bg-opacity-90 text-white font-medium rounded-md transition-all disabled:opacity-50 disabled:cursor-not-allowed min-h-[44px]"
            aria-label="Enviar feedback"
          >
            <Send className="w-5 h-5" />
            {isSubmitting ? 'Enviando...' : 'Enviar Feedback'}
          </motion.button>
        </form>

        {/* Info */}
        <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-md">
          <div className="flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-tesla-blue mt-0.5 flex-shrink-0" />
            <div>
              <p className="text-sm text-tesla-blue font-semibold mb-1">Información</p>
              <p className="text-xs text-tesla-gray-dark">
                Tu feedback es muy valioso para nosotros. Revisaremos cada sugerencia y la
                consideraremos para futuras actualizaciones.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}


