'use client';

import { useState } from 'react';
import { Download, FileJson, FileText, FileCode, FileSpreadsheet, Image } from 'lucide-react';
import toast from 'react-hot-toast';

interface MusicExportAdvancedProps {
  data: any;
  filename?: string;
}

export function MusicExportAdvanced({ data, filename = 'music-data' }: MusicExportAdvancedProps) {
  const [format, setFormat] = useState<'json' | 'csv' | 'markdown' | 'xlsx' | 'png'>('json');
  const [includeMetadata, setIncludeMetadata] = useState(true);

  const handleExport = () => {
    if (!data) {
      toast.error('No hay datos para exportar');
      return;
    }

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
        // Convertir a CSV
        const headers = Object.keys(data).join(',');
        const values = Object.values(data).join(',');
        content = `${headers}\n${values}`;
        mimeType = 'text/csv';
        extension = 'csv';
        break;
      case 'markdown':
        content = `# ${filename}\n\n\`\`\`json\n${JSON.stringify(data, null, 2)}\n\`\`\``;
        mimeType = 'text/markdown';
        extension = 'md';
        break;
      case 'xlsx':
        // Simular exportación XLSX (en producción usaría una librería)
        toast.info('Exportación XLSX próximamente');
        return;
      case 'png':
        // Simular exportación PNG (en producción usaría canvas)
        toast.info('Exportación PNG próximamente');
        return;
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
  };

  const formatIcons = {
    json: FileJson,
    csv: FileText,
    markdown: FileCode,
    xlsx: FileSpreadsheet,
    png: Image,
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <Download className="w-5 h-5 text-purple-300" />
        <h3 className="text-lg font-semibold text-white">Exportación Avanzada</h3>
      </div>

      <div className="space-y-4">
        <div>
          <label className="block text-sm text-gray-400 mb-2">Formato</label>
          <div className="grid grid-cols-5 gap-2">
            {(['json', 'csv', 'markdown', 'xlsx', 'png'] as const).map((fmt) => {
              const Icon = formatIcons[fmt];
              return (
                <button
                  key={fmt}
                  onClick={() => setFormat(fmt)}
                  className={`flex flex-col items-center gap-2 px-3 py-3 rounded-lg transition-colors ${
                    format === fmt
                      ? 'bg-purple-600 text-white'
                      : 'bg-white/10 text-gray-300 hover:bg-white/20'
                  }`}
                >
                  <Icon className="w-5 h-5" />
                  <span className="text-xs uppercase">{fmt}</span>
                </button>
              );
            })}
          </div>
        </div>

        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            id="include-metadata"
            checked={includeMetadata}
            onChange={(e) => setIncludeMetadata(e.target.checked)}
            className="w-4 h-4 rounded accent-purple-500"
          />
          <label htmlFor="include-metadata" className="text-sm text-gray-300">
            Incluir metadatos
          </label>
        </div>

        <button
          onClick={handleExport}
          disabled={!data}
          className="w-full px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
        >
          <Download className="w-5 h-5" />
          Exportar como {format.toUpperCase()}
        </button>
      </div>
    </div>
  );
}


