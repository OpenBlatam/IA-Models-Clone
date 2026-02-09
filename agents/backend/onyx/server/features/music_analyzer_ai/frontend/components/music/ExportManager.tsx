'use client';

import { useState } from 'react';
import { Download, FileText, FileJson, FileCode, Package } from 'lucide-react';
import toast from 'react-hot-toast';

interface ExportManagerProps {
  analyses: any[];
}

export function ExportManager({ analyses }: ExportManagerProps) {
  const [format, setFormat] = useState<'json' | 'csv' | 'markdown'>('json');

  const handleExport = () => {
    if (analyses.length === 0) {
      toast.error('No hay análisis para exportar');
      return;
    }

    let content = '';
    let filename = '';
    let mimeType = '';

    switch (format) {
      case 'json':
        content = JSON.stringify(analyses, null, 2);
        filename = 'analyses.json';
        mimeType = 'application/json';
        break;
      case 'csv':
        // Convert to CSV
        const headers = ['Track', 'Artists', 'Key', 'Tempo', 'Energy', 'Danceability'];
        const rows = analyses.map((a: any) => [
          a.track_name || '',
          Array.isArray(a.artists) ? a.artists.join('; ') : a.artists || '',
          a.key_signature || '',
          a.tempo || '',
          a.energy || '',
          a.danceability || '',
        ]);
        content = [headers.join(','), ...rows.map((r: any[]) => r.join(','))].join('\n');
        filename = 'analyses.csv';
        mimeType = 'text/csv';
        break;
      case 'markdown':
        content = analyses.map((a: any, idx: number) => `
## ${idx + 1}. ${a.track_name || 'Unknown'}

**Artists:** ${Array.isArray(a.artists) ? a.artists.join(', ') : a.artists || 'Unknown'}
**Key:** ${a.key_signature || 'N/A'}
**Tempo:** ${a.tempo || 'N/A'} BPM
**Energy:** ${a.energy || 'N/A'}
**Danceability:** ${a.danceability || 'N/A'}

---
        `).join('\n');
        filename = 'analyses.md';
        mimeType = 'text/markdown';
        break;
    }

    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    toast.success(`Exportado: ${analyses.length} análisis`);
  };

  const formatIcons = {
    json: FileJson,
    csv: FileText,
    markdown: FileCode,
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <Package className="w-6 h-6 text-purple-300" />
        <h2 className="text-2xl font-semibold text-white">Exportar Múltiples</h2>
      </div>

      <div className="space-y-4">
        <div>
          <p className="text-sm text-gray-400 mb-2">Formato</p>
          <div className="flex gap-2">
            {(['json', 'csv', 'markdown'] as const).map((fmt) => {
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
                  <span className="uppercase">{fmt}</span>
                </button>
              );
            })}
          </div>
        </div>

        <div className="bg-white/5 rounded-lg p-3">
          <p className="text-sm text-gray-400 mb-1">Análisis a exportar</p>
          <p className="text-lg font-semibold text-white">{analyses.length}</p>
        </div>

        <button
          onClick={handleExport}
          disabled={analyses.length === 0}
          className="w-full px-6 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
        >
          <Download className="w-5 h-5" />
          Exportar {analyses.length} Análisis
        </button>
      </div>
    </div>
  );
}


