'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useForm } from 'react-hook-form';
import { apiClient } from '@/lib/api-client';
import { Loader2, Plus, X, CheckCircle, AlertCircle } from 'lucide-react';
import toast from 'react-hot-toast';
import { useQuery } from '@tanstack/react-query';

export default function BatchPage() {
  const router = useRouter();
  const [urls, setUrls] = useState<string[]>(['']);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [batchId, setBatchId] = useState<string | null>(null);

  const { data: batchStatus } = useQuery({
    queryKey: ['batch', batchId],
    queryFn: () => apiClient.getBatchStatus(batchId!),
    enabled: !!batchId,
    refetchInterval: (data) => {
      if (
        data?.status === 'completed' ||
        data?.status === 'failed'
      ) {
        return false;
      }
      return 3000;
    },
  });

  const addUrl = () => {
    setUrls([...urls, '']);
  };

  const removeUrl = (index: number) => {
    setUrls(urls.filter((_, i) => i !== index));
  };

  const updateUrl = (index: number, value: string) => {
    const newUrls = [...urls];
    newUrls[index] = value;
    setUrls(newUrls);
  };

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const validUrls = urls.filter((url) => url.trim() !== '');
    
    if (validUrls.length === 0) {
      toast.error('Agrega al menos una URL');
      return;
    }

    setIsSubmitting(true);
    try {
      const batch = await apiClient.createBatchJob(validUrls, true, true);
      setBatchId(batch.batch_id);
      toast.success('Procesamiento batch iniciado');
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Error al iniciar batch');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 py-12">
      <div className="container mx-auto px-4 max-w-4xl">
        <div className="bg-white rounded-xl shadow-lg p-8">
          <h1 className="text-3xl font-bold mb-8">Procesamiento Batch</h1>

          <form onSubmit={onSubmit} className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                URLs de Videos
              </label>
              <div className="space-y-2">
                {urls.map((url, index) => (
                  <div key={index} className="flex gap-2">
                    <input
                      type="url"
                      value={url}
                      onChange={(e) => updateUrl(index, e.target.value)}
                      placeholder="https://www.youtube.com/watch?v=..."
                      className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                    {urls.length > 1 && (
                      <button
                        type="button"
                        onClick={() => removeUrl(index)}
                        className="px-3 py-2 bg-red-100 text-red-600 rounded-lg hover:bg-red-200"
                      >
                        <X className="w-5 h-5" />
                      </button>
                    )}
                  </div>
                ))}
              </div>
              <button
                type="button"
                onClick={addUrl}
                className="mt-2 flex items-center gap-2 text-blue-600 hover:text-blue-800"
              >
                <Plus className="w-4 h-4" />
                Agregar otra URL
              </button>
            </div>

            <button
              type="submit"
              disabled={isSubmitting}
              className="w-full bg-blue-600 text-white py-3 rounded-lg font-semibold hover:bg-blue-700 transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {isSubmitting ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Iniciando batch...
                </>
              ) : (
                'Iniciar Procesamiento Batch'
              )}
            </button>
          </form>

          {/* Batch Status */}
          {batchStatus && (
            <div className="mt-8 bg-gray-50 rounded-lg p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold">Estado del Batch</h2>
                <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-semibold">
                  {batchStatus.status}
                </span>
              </div>
              <div className="grid grid-cols-3 gap-4 mb-4">
                <div>
                  <p className="text-sm text-gray-600">Total</p>
                  <p className="text-2xl font-bold">{batchStatus.total}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Completados</p>
                  <p className="text-2xl font-bold text-green-600">
                    {batchStatus.completed}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Fallidos</p>
                  <p className="text-2xl font-bold text-red-600">
                    {batchStatus.failed}
                  </p>
                </div>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-blue-600 h-2 rounded-full transition-all"
                  style={{
                    width: `${(batchStatus.completed / batchStatus.total) * 100}%`,
                  }}
                />
              </div>
              {batchStatus.status === 'completed' && (
                <button
                  onClick={() => router.push(`/batch/${batchStatus.batch_id}`)}
                  className="mt-4 w-full bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700"
                >
                  Ver Resultados
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}




