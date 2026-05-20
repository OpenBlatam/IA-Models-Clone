'use client';

import { useState } from 'react';
import { Download, FileJson, FileText, FileCode, Loader2 } from 'lucide-react';
import toast from 'react-hot-toast';

interface MusicExportProps {
  data: any;
  filename?: string;
}

export function MusicExport({ data, filename = 'music-analysis' }: MusicExportProps) {
  const [exporting, setExporting] = useState(false);
  const [format, setFormat] = useState<'json' | 'csv' | 'markdown'>('json');

  const handleExport = async () => {
    if (!data) {
      toast.error('No hay datos para exportar');
      return;
    }

    setExporting(true);

    try {
      let content = '';
      let mimeType = '';
      let extension = '';

      switch (format) {
        case 'json':
          content = JSON.stringify(data, null, 2);
          mimeType = 'application/json';
          extension = 'json';
          break;
        case 'csv':
          // Convertir a CSV básico
          const headers = Object.keys(data).join(',');
          const values = Object.values(data).join(',');
          content = `${headers}\n${values}`;
          mimeType = 'text/csv';
          extension = 'csv';
          break;
        case 'markdown':
          content = `# Análisis Musical\n\n\`\`\`json\n${JSON.stringify(data, null, 2)}\n\`\`\``;
          mimeType = 'text/markdown';
          extension = 'md';
          break;
      }

      const blob = new Blob([content], { type: mimeType });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${filename}.${extension}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      toast.success(`Exportado como ${format.toUpperCase()}`);
    } catch (error) {
      toast.error('Error al exportar');
    } finally {
      setExporting(false);
    }
  };

  const formatIcons = {
    json: FileJson,
    csv: FileText,
    markdown: FileCode,
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <Download className="w-5 h-5 text-purple-300" />
        <h3 className="text-lg font-semibold text-white">Exportar</h3>
      </div>

      <div className="space-y-4">
        <div>
          <label className="block text-sm text-gray-400 mb-2">Formato</label>
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

        <button
          onClick={handleExport}
          disabled={exporting || !data}
          className="w-full px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
        >
          {exporting ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Exportando...
            </>
          ) : (
            <>
              <Download className="w-5 h-5" />
              Exportar como {format.toUpperCase()}
            </>
          )}
        </button>
      </div>
    </div>
  );
}


