'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { apiClient } from '@/lib/api-client';
import { SupportedPlatform, TranscriptionStatus } from '@/types/api';
import { validateVideoUrl } from '@/lib/utils';
import { Video, Loader2, CheckCircle, XCircle } from 'lucide-react';
import toast from 'react-hot-toast';

const transcriptionSchema = z.object({
  video_url: z.string().min(1, 'URL requerida').refine(validateVideoUrl, 'URL inválida'),
  platform: z.nativeEnum(SupportedPlatform).default(SupportedPlatform.AUTO),
  include_timestamps: z.boolean().default(true),
  include_analysis: z.boolean().default(true),
  language: z.string().optional(),
  webhook_url: z.string().url().optional().or(z.literal('')),
});

type TranscriptionFormData = z.infer<typeof transcriptionSchema>;

export default function TranscribePage() {
  const router = useRouter();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<TranscriptionStatus | null>(null);

  const {
    register,
    handleSubmit,
    formState: { errors },
    watch,
  } = useForm<TranscriptionFormData>({
    resolver: zodResolver(transcriptionSchema),
    defaultValues: {
      platform: SupportedPlatform.AUTO,
      include_timestamps: true,
      include_analysis: true,
    },
  });

  const onSubmit = async (data: TranscriptionFormData) => {
    setIsSubmitting(true);
    try {
      const response = await apiClient.transcribeVideo({
        ...data,
        webhook_url: data.webhook_url || undefined,
        language: data.language || undefined,
      });

      setJobId(response.job_id);
      setStatus(response.status);
      toast.success('Transcripción iniciada correctamente');

      // Redirect to job detail page
      router.push(`/jobs/${response.job_id}`);
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Error al iniciar transcripción');
      console.error('Transcription error:', error);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 py-12">
      <div className="container mx-auto px-4 max-w-4xl">
        <div className="bg-white rounded-xl shadow-lg p-8">
          <div className="flex items-center gap-3 mb-8">
            <Video className="w-8 h-8 text-blue-600" />
            <h1 className="text-3xl font-bold">Nueva Transcripción</h1>
          </div>

          <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
            {/* Video URL */}
            <div>
              <label htmlFor="video_url" className="block text-sm font-medium text-gray-700 mb-2">
                URL del Video
              </label>
              <input
                {...register('video_url')}
                type="url"
                id="video_url"
                placeholder="https://www.youtube.com/watch?v=..."
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              {errors.video_url && (
                <p className="mt-1 text-sm text-red-600">{errors.video_url.message}</p>
              )}
              <p className="mt-1 text-sm text-gray-500">
                Soporta YouTube, TikTok e Instagram
              </p>
            </div>

            {/* Platform Selection */}
            <div>
              <label htmlFor="platform" className="block text-sm font-medium text-gray-700 mb-2">
                Plataforma
              </label>
              <select
                {...register('platform')}
                id="platform"
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value={SupportedPlatform.AUTO}>Auto-detectar</option>
                <option value={SupportedPlatform.YOUTUBE}>YouTube</option>
                <option value={SupportedPlatform.TIKTOK}>TikTok</option>
                <option value={SupportedPlatform.INSTAGRAM}>Instagram</option>
              </select>
            </div>

            {/* Options */}
            <div className="space-y-4">
              <div className="flex items-center">
                <input
                  {...register('include_timestamps')}
                  type="checkbox"
                  id="include_timestamps"
                  className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                />
                <label htmlFor="include_timestamps" className="ml-2 text-sm text-gray-700">
                  Incluir timestamps
                </label>
              </div>

              <div className="flex items-center">
                <input
                  {...register('include_analysis')}
                  type="checkbox"
                  id="include_analysis"
                  className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                />
                <label htmlFor="include_analysis" className="ml-2 text-sm text-gray-700">
                  Incluir análisis con IA (framework, estructura, etc.)
                </label>
              </div>
            </div>

            {/* Language */}
            <div>
              <label htmlFor="language" className="block text-sm font-medium text-gray-700 mb-2">
                Idioma (opcional, auto-detecta si no se especifica)
              </label>
              <input
                {...register('language')}
                type="text"
                id="language"
                placeholder="es, en, pt, etc."
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            {/* Webhook URL */}
            <div>
              <label htmlFor="webhook_url" className="block text-sm font-medium text-gray-700 mb-2">
                Webhook URL (opcional)
              </label>
              <input
                {...register('webhook_url')}
                type="url"
                id="webhook_url"
                placeholder="https://tu-servidor.com/webhook"
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={isSubmitting}
              className="w-full bg-blue-600 text-white py-3 rounded-lg font-semibold hover:bg-blue-700 transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {isSubmitting ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Iniciando transcripción...
                </>
              ) : (
                <>
                  <Video className="w-5 h-5" />
                  Iniciar Transcripción
                </>
              )}
            </button>
          </form>

          {/* Status Display */}
          {jobId && status && (
            <div className="mt-6 p-4 bg-blue-50 rounded-lg">
              <div className="flex items-center gap-2">
                {status === TranscriptionStatus.COMPLETED ? (
                  <CheckCircle className="w-5 h-5 text-green-600" />
                ) : status === TranscriptionStatus.FAILED ? (
                  <XCircle className="w-5 h-5 text-red-600" />
                ) : (
                  <Loader2 className="w-5 h-5 text-blue-600 animate-spin" />
                )}
                <span className="font-medium">Job ID: {jobId}</span>
                <span className="text-sm text-gray-600">({status})</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}




