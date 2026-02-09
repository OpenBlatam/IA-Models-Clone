'use client';

import { useQuery } from '@tanstack/react-query';
import { useParams } from 'next/navigation';
import { apiClient } from '@/lib/api-client';
import {
  formatDuration,
  formatDate,
  getStatusColor,
  getFrameworkName,
  copyToClipboard,
  downloadFile,
} from '@/lib/utils';
import { TranscriptionStatus } from '@/types/api';
import {
  Loader2,
  Copy,
  Download,
  RefreshCw,
  FileText,
  Sparkles,
  Clock,
  User,
  Video as VideoIcon,
} from 'lucide-react';
import { useState } from 'react';
import toast from 'react-hot-toast';
import Link from 'next/link';

export default function JobDetailPage() {
  const params = useParams();
  const jobId = params.jobId as string;
  const [selectedFormat, setSelectedFormat] = useState<'text' | 'srt' | 'vtt'>('text');

  const { data: job, isLoading, refetch } = useQuery({
    queryKey: ['job', jobId],
    queryFn: () => apiClient.getTranscriptionStatus(jobId),
    refetchInterval: (data) => {
      // Stop polling if completed or failed
      if (
        data?.status === TranscriptionStatus.COMPLETED ||
        data?.status === TranscriptionStatus.FAILED
      ) {
        return false;
      }
      return 3000; // Poll every 3 seconds if still processing
    },
  });

  const { data: transcriptionText } = useQuery({
    queryKey: ['transcription-text', jobId, selectedFormat],
    queryFn: () => apiClient.getTranscriptionText(jobId, true, selectedFormat),
    enabled: job?.status === TranscriptionStatus.COMPLETED,
  });

  const handleCopy = async () => {
    if (transcriptionText?.content) {
      await copyToClipboard(transcriptionText.content);
      toast.success('Copiado al portapapeles');
    }
  };

  const handleDownload = () => {
    if (transcriptionText?.content) {
      const extension = selectedFormat === 'srt' ? 'srt' : selectedFormat === 'vtt' ? 'vtt' : 'txt';
      const mimeType =
        selectedFormat === 'srt' || selectedFormat === 'vtt'
          ? 'text/plain'
          : 'text/plain';
      downloadFile(
        transcriptionText.content,
        `transcription-${jobId.substring(0, 8)}.${extension}`,
        mimeType
      );
      toast.success('Descarga iniciada');
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
      </div>
    );
  }

  if (!job) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <p className="text-gray-600 mb-4">Job no encontrado</p>
          <Link href="/jobs" className="text-blue-600 hover:underline">
            Volver a trabajos
          </Link>
        </div>
      </div>
    );
  }

  const isProcessing =
    job.status !== TranscriptionStatus.COMPLETED &&
    job.status !== TranscriptionStatus.FAILED;

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 py-12">
      <div className="container mx-auto px-4 max-w-6xl">
        {/* Header */}
        <div className="mb-6">
          <Link
            href="/jobs"
            className="text-blue-600 hover:text-blue-800 mb-4 inline-block"
          >
            ← Volver a trabajos
          </Link>
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold mb-2">
                {job.video_title || 'Transcripción'}
              </h1>
              <p className="text-gray-600 font-mono text-sm">{job.job_id}</p>
            </div>
            <button
              onClick={() => refetch()}
              className="flex items-center gap-2 px-4 py-2 bg-white border border-gray-300 rounded-lg hover:bg-gray-50"
            >
              <RefreshCw className="w-4 h-4" />
              Actualizar
            </button>
          </div>
        </div>

        {/* Status Card */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              {isProcessing ? (
                <Loader2 className="w-6 h-6 animate-spin text-blue-600" />
              ) : job.status === TranscriptionStatus.COMPLETED ? (
                <div className="w-6 h-6 rounded-full bg-green-500"></div>
              ) : (
                <div className="w-6 h-6 rounded-full bg-red-500"></div>
              )}
              <span
                className={`px-3 py-1 rounded-full text-sm font-semibold ${getStatusColor(
                  job.status
                )}`}
              >
                {job.status}
              </span>
            </div>
            {job.processing_time && (
              <span className="text-sm text-gray-600">
                Tiempo: {job.processing_time.toFixed(2)}s
              </span>
            )}
          </div>

          {/* Video Info */}
          <div className="grid md:grid-cols-3 gap-4 mt-4">
            {job.video_author && (
              <div className="flex items-center gap-2">
                <User className="w-5 h-5 text-gray-400" />
                <div>
                  <p className="text-sm text-gray-600">Autor</p>
                  <p className="font-medium">{job.video_author}</p>
                </div>
              </div>
            )}
            {job.video_duration && (
              <div className="flex items-center gap-2">
                <Clock className="w-5 h-5 text-gray-400" />
                <div>
                  <p className="text-sm text-gray-600">Duración</p>
                  <p className="font-medium">{formatDuration(job.video_duration)}</p>
                </div>
              </div>
            )}
            {job.platform_detected && (
              <div className="flex items-center gap-2">
                <VideoIcon className="w-5 h-5 text-gray-400" />
                <div>
                  <p className="text-sm text-gray-600">Plataforma</p>
                  <p className="font-medium capitalize">{job.platform_detected}</p>
                </div>
              </div>
            )}
          </div>

          {/* Metadata */}
          <div className="mt-4 pt-4 border-t text-sm text-gray-600">
            <p>Creado: {formatDate(job.created_at)}</p>
            {job.completed_at && <p>Completado: {formatDate(job.completed_at)}</p>}
          </div>
        </div>

        {/* Error Display */}
        {job.error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
            <p className="text-red-800 font-semibold">Error:</p>
            <p className="text-red-600">{job.error}</p>
          </div>
        )}

        {/* Analysis */}
        {job.analysis && (
          <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
            <div className="flex items-center gap-2 mb-4">
              <Sparkles className="w-6 h-6 text-purple-600" />
              <h2 className="text-2xl font-bold">Análisis con IA</h2>
            </div>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <p className="text-sm text-gray-600 mb-1">Framework</p>
                <p className="font-semibold text-lg">
                  {getFrameworkName(job.analysis.framework)}
                </p>
                <p className="text-sm text-gray-500">
                  Confianza: {(job.analysis.framework_confidence * 100).toFixed(1)}%
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600 mb-1">Tono</p>
                <p className="font-semibold capitalize">{job.analysis.tone}</p>
              </div>
              {job.analysis.key_points && job.analysis.key_points.length > 0 && (
                <div className="md:col-span-2">
                  <p className="text-sm text-gray-600 mb-2">Puntos Clave</p>
                  <ul className="list-disc list-inside space-y-1">
                    {job.analysis.key_points.map((point, idx) => (
                      <li key={idx} className="text-gray-700">
                        {point}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {job.analysis.hashtags_suggested &&
                job.analysis.hashtags_suggested.length > 0 && (
                  <div className="md:col-span-2">
                    <p className="text-sm text-gray-600 mb-2">Hashtags Sugeridos</p>
                    <div className="flex flex-wrap gap-2">
                      {job.analysis.hashtags_suggested.map((tag, idx) => (
                        <span
                          key={idx}
                          className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-sm"
                        >
                          #{tag}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
            </div>
          </div>
        )}

        {/* Transcription Text */}
        {job.status === TranscriptionStatus.COMPLETED && (
          <div className="bg-white rounded-xl shadow-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <FileText className="w-6 h-6 text-blue-600" />
                <h2 className="text-2xl font-bold">Transcripción</h2>
              </div>
              <div className="flex gap-2">
                <select
                  value={selectedFormat}
                  onChange={(e) =>
                    setSelectedFormat(e.target.value as 'text' | 'srt' | 'vtt')
                  }
                  className="px-3 py-1 border border-gray-300 rounded-lg text-sm"
                >
                  <option value="text">Texto</option>
                  <option value="srt">SRT</option>
                  <option value="vtt">VTT</option>
                </select>
                <button
                  onClick={handleCopy}
                  className="px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg flex items-center gap-2"
                >
                  <Copy className="w-4 h-4" />
                  Copiar
                </button>
                <button
                  onClick={handleDownload}
                  className="px-4 py-2 bg-blue-600 text-white hover:bg-blue-700 rounded-lg flex items-center gap-2"
                >
                  <Download className="w-4 h-4" />
                  Descargar
                </button>
              </div>
            </div>

            {transcriptionText ? (
              <div className="bg-gray-50 rounded-lg p-4 max-h-96 overflow-y-auto">
                <pre className="whitespace-pre-wrap text-sm font-mono">
                  {transcriptionText.content}
                </pre>
              </div>
            ) : (
              <div className="flex justify-center py-8">
                <Loader2 className="w-6 h-6 animate-spin text-blue-600" />
              </div>
            )}

            {/* Segments */}
            {job.segments && job.segments.length > 0 && (
              <div className="mt-6">
                <h3 className="text-lg font-semibold mb-4">Segmentos</h3>
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {job.segments.map((segment) => (
                    <div
                      key={segment.id}
                      className="bg-gray-50 rounded p-3 text-sm"
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-gray-600 font-mono text-xs">
                          {segment.formatted_timestamp || `${segment.start_time}s - ${segment.end_time}s`}
                        </span>
                        {segment.confidence && (
                          <span className="text-gray-500">
                            {(segment.confidence * 100).toFixed(0)}%
                          </span>
                        )}
                      </div>
                      <p className="text-gray-800">{segment.text}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}




