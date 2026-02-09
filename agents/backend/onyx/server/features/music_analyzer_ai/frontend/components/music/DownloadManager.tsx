'use client';

import { useState } from 'react';
import { Download, CheckCircle, Loader2 } from 'lucide-react';
import toast from 'react-hot-toast';

interface DownloadItem {
  id: string;
  name: string;
  type: 'analysis' | 'playlist' | 'export';
  status: 'pending' | 'downloading' | 'completed' | 'error';
  progress?: number;
}

export function DownloadManager() {
  const [downloads, setDownloads] = useState<DownloadItem[]>([]);

  const addDownload = (item: DownloadItem) => {
    setDownloads((prev) => [...prev, item]);

    // Simular descarga
    setTimeout(() => {
      setDownloads((prev) =>
        prev.map((d) =>
          d.id === item.id
            ? { ...d, status: 'downloading', progress: 50 }
            : d
        )
      );

      setTimeout(() => {
        setDownloads((prev) =>
          prev.map((d) =>
            d.id === item.id
              ? { ...d, status: 'completed', progress: 100 }
              : d
          )
        );
        toast.success(`Descarga completada: ${item.name}`);
      }, 2000);
    }, 500);
  };

  const removeDownload = (id: string) => {
    setDownloads((prev) => prev.filter((d) => d.id !== id));
  };

  if (downloads.length === 0) return null;

  return (
    <div className="fixed bottom-4 right-4 w-80 bg-gray-800 rounded-lg shadow-lg border border-white/20 z-50">
      <div className="p-4 border-b border-white/10">
        <h3 className="text-sm font-semibold text-white">Descargas</h3>
      </div>
      <div className="max-h-64 overflow-y-auto">
        {downloads.map((download) => (
          <div
            key={download.id}
            className="p-3 border-b border-white/10 last:border-0"
          >
            <div className="flex items-center gap-2 mb-2">
              {download.status === 'completed' ? (
                <CheckCircle className="w-4 h-4 text-green-400" />
              ) : download.status === 'downloading' ? (
                <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />
              ) : (
                <Download className="w-4 h-4 text-gray-400" />
              )}
              <div className="flex-1 min-w-0">
                <p className="text-sm text-white truncate">{download.name}</p>
                <p className="text-xs text-gray-400">{download.type}</p>
              </div>
              <button
                onClick={() => removeDownload(download.id)}
                className="text-gray-400 hover:text-white"
              >
                ×
              </button>
            </div>
            {download.status === 'downloading' && download.progress !== undefined && (
              <div className="w-full bg-gray-700 rounded-full h-1">
                <div
                  className="bg-blue-400 h-1 rounded-full transition-all"
                  style={{ width: `${download.progress}%` }}
                />
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}


