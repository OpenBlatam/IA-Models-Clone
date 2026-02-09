'use client';

import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { Download, FileText, FileJson, FileCode } from 'lucide-react';
import toast from 'react-hot-toast';

interface ExportAnalysisProps {
  trackId: string;
  trackName: string;
}

export function ExportAnalysis({ trackId, trackName }: ExportAnalysisProps) {
  const [format, setFormat] = useState<'json' | 'text' | 'markdown'>('json');
  const [includeCoaching, setIncludeCoaching] = useState(true);

  const exportMutation = useMutation({
    mutationFn: () => musicApiService.exportAnalysis(trackId, format, includeCoaching),
    onSuccess: (data) => {
      // Crear blob y descargar
      const blob = new Blob([typeof data === 'string' ? data : JSON.stringify(data, null, 2)], {
        type: format === 'json' ? 'application/json' : 'text/plain',
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${trackName.replace(/[^a-z0-9]/gi, '_')}_analysis.${format === 'json' ? 'json' : format === 'markdown' ? 'md' : 'txt'}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      toast.success('Análisis exportado');
    },
    onError: () => {
      toast.error('Error al exportar análisis');
    },
  });

  const formatIcons = {
    json: FileJson,
    text: FileText,
    markdown: FileCode,
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <Download className="w-6 h-6 text-purple-300" />
        <h2 className="text-2xl font-semibold text-white">Exportar Análisis</h2>
      </div>

      <div className="space-y-4">
        {/* Format Selection */}
        <div>
          <p className="text-sm text-gray-400 mb-2">Formato</p>
          <div className="flex gap-2">
            {(['json', 'text', 'markdown'] as const).map((fmt) => {
              const Icon = formatIcons[fmt];
              return (
                <button
                  key={fmt}
                  onClick={() => setFormat(fmt)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                    format === fmt
                      ? 'bg-purple-600 text-white'
                      : 'bg-white/10 text-gray-300 hover:bg-white/20'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span className="capitalize">{fmt}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Options */}
        <div>
          <label className="flex items-center gap-2 text-gray-300">
            <input
              type="checkbox"
              checked={includeCoaching}
              onChange={(e) => setIncludeCoaching(e.target.checked)}
              className="w-4 h-4"
            />
            <span>Incluir coaching</span>
          </label>
        </div>

        {/* Export Button */}
        <button
          onClick={() => exportMutation.mutate()}
          disabled={exportMutation.isPending}
          className="w-full px-6 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
        >
          <Download className="w-5 h-5" />
          {exportMutation.isPending ? 'Exportando...' : 'Exportar Análisis'}
        </button>
      </div>
    </div>
  );
}

