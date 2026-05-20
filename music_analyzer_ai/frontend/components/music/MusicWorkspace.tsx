'use client';

import { useState } from 'react';
import { Layout, Save, Folder, FileText } from 'lucide-react';
import toast from 'react-hot-toast';

export function MusicWorkspace() {
  const [workspaces, setWorkspaces] = useState<Array<{ id: string; name: string; tracks: number }>>([
    { id: '1', name: 'Mi Workspace', tracks: 15 },
    { id: '2', name: 'Análisis 2024', tracks: 8 },
  ]);

  const [currentWorkspace, setCurrentWorkspace] = useState('1');

  const handleSaveWorkspace = () => {
    toast.success('Workspace guardado');
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Layout className="w-5 h-5 text-purple-300" />
          <h3 className="text-lg font-semibold text-white">Workspace</h3>
        </div>
        <button
          onClick={handleSaveWorkspace}
          className="px-3 py-1 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors flex items-center gap-2 text-sm"
        >
          <Save className="w-4 h-4" />
          Guardar
        </button>
      </div>

      <div className="space-y-3">
        <div>
          <label className="block text-sm text-gray-400 mb-2">Workspace Actual</label>
          <select
            value={currentWorkspace}
            onChange={(e) => setCurrentWorkspace(e.target.value)}
            className="w-full px-4 py-2 bg-white/20 border border-white/30 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-400"
          >
            {workspaces.map((ws) => (
              <option key={ws.id} value={ws.id}>
                {ws.name} ({ws.tracks} tracks)
              </option>
            ))}
          </select>
        </div>

        <div className="grid grid-cols-3 gap-2">
          {workspaces.map((ws) => (
            <button
              key={ws.id}
              onClick={() => setCurrentWorkspace(ws.id)}
              className={`p-3 rounded-lg border transition-colors text-left ${
                currentWorkspace === ws.id
                  ? 'bg-purple-600/30 border-purple-500'
                  : 'bg-white/5 border-white/10 hover:bg-white/10'
              }`}
            >
              <div className="flex items-center gap-2 mb-1">
                <Folder className="w-4 h-4 text-purple-300" />
                <span className="text-white font-medium text-sm">{ws.name}</span>
              </div>
              <div className="flex items-center gap-1 text-xs text-gray-400">
                <FileText className="w-3 h-3" />
                {ws.tracks} tracks
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}


