'use client';

import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/lib/api-client';
import { formatDate, getStatusColor } from '@/lib/utils';
import { TranscriptionStatus } from '@/types/api';
import { Loader2, Search, Filter } from 'lucide-react';
import Link from 'next/link';
import { useState } from 'react';

export default function JobsPage() {
  const [statusFilter, setStatusFilter] = useState<string>('');
  const [searchQuery, setSearchQuery] = useState('');

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['jobs', statusFilter],
    queryFn: () => apiClient.listJobs(statusFilter || undefined),
    refetchInterval: 5000, // Refetch every 5 seconds
  });

  const filteredJobs = data?.jobs.filter((job) => {
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      return (
        job.job_id.toLowerCase().includes(query) ||
        job.video_title?.toLowerCase().includes(query) ||
        job.platform?.toLowerCase().includes(query)
      );
    }
    return true;
  });

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 py-12">
      <div className="container mx-auto px-4">
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-4">Trabajos de Transcripción</h1>

          {/* Filters */}
          <div className="flex gap-4 mb-6">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
              <input
                type="text"
                placeholder="Buscar por ID, título o plataforma..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
            <div className="relative">
              <Filter className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
              <select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                className="pl-10 pr-8 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent appearance-none bg-white"
              >
                <option value="">Todos los estados</option>
                {Object.values(TranscriptionStatus).map((status) => (
                  <option key={status} value={status}>
                    {status}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Stats */}
          {data && (
            <div className="bg-white rounded-lg p-4 mb-6 shadow-md">
              <div className="flex gap-6">
                <div>
                  <span className="text-sm text-gray-600">Total:</span>
                  <span className="ml-2 font-semibold">{data.total}</span>
                </div>
                <div>
                  <span className="text-sm text-gray-600">Completados:</span>
                  <span className="ml-2 font-semibold text-green-600">
                    {data.jobs.filter((j) => j.status === TranscriptionStatus.COMPLETED).length}
                  </span>
                </div>
                <div>
                  <span className="text-sm text-gray-600">En proceso:</span>
                  <span className="ml-2 font-semibold text-blue-600">
                    {data.jobs.filter(
                      (j) =>
                        j.status !== TranscriptionStatus.COMPLETED &&
                        j.status !== TranscriptionStatus.FAILED
                    ).length}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Jobs List */}
        {isLoading ? (
          <div className="flex justify-center items-center py-12">
            <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
          </div>
        ) : error ? (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-800">
            Error al cargar trabajos. Por favor, intenta de nuevo.
          </div>
        ) : filteredJobs && filteredJobs.length > 0 ? (
          <div className="bg-white rounded-xl shadow-lg overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Job ID
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Título
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Plataforma
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Estado
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Creado
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Acciones
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {filteredJobs.map((job) => (
                    <tr key={job.job_id} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-gray-900">
                        {job.job_id.substring(0, 8)}...
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-900">
                        {job.video_title || 'Sin título'}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {job.platform || '-'}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span
                          className={`px-2 py-1 text-xs font-semibold rounded-full ${getStatusColor(
                            job.status
                          )}`}
                        >
                          {job.status}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {formatDate(job.created_at)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">
                        <Link
                          href={`/jobs/${job.job_id}`}
                          className="text-blue-600 hover:text-blue-800 font-medium"
                        >
                          Ver detalles
                        </Link>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : (
          <div className="bg-white rounded-xl shadow-lg p-12 text-center">
            <p className="text-gray-500">No se encontraron trabajos</p>
          </div>
        )}
      </div>
    </div>
  );
}




