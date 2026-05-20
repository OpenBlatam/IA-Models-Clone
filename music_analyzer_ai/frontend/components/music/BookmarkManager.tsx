'use client';

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { Bookmark, BookmarkCheck, Plus, Folder } from 'lucide-react';
import toast from 'react-hot-toast';

export function BookmarkManager() {
  const [userId] = useState('user123');
  const [selectedFolder, setSelectedFolder] = useState<string | null>(null);

  const { data: bookmarks } = useQuery({
    queryKey: ['bookmarks', userId],
    queryFn: () => musicApiService.getBookmarks?.(userId) || Promise.resolve({ bookmarks: [] }),
  });

  const { data: folders } = useQuery({
    queryKey: ['bookmark-folders', userId],
    queryFn: () => musicApiService.getBookmarkFolders?.(userId) || Promise.resolve({ folders: [] }),
  });

  const queryClient = useQueryClient();

  const addBookmarkMutation = useMutation({
    mutationFn: ({ trackId, folderId }: { trackId: string; folderId?: string }) =>
      musicApiService.addBookmark?.(userId, trackId, folderId) || Promise.resolve({}),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['bookmarks', userId] });
      toast.success('Marcador agregado');
    },
    onError: () => {
      toast.error('Error al agregar marcador');
    },
  });

  const removeBookmarkMutation = useMutation({
    mutationFn: (bookmarkId: string) =>
      musicApiService.removeBookmark?.(bookmarkId) || Promise.resolve({}),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['bookmarks', userId] });
      toast.success('Marcador eliminado');
    },
    onError: () => {
      toast.error('Error al eliminar marcador');
    },
  });

  const bookmarkList = bookmarks?.bookmarks || [];
  const folderList = folders?.folders || [];
  const filteredBookmarks = selectedFolder
    ? bookmarkList.filter((b: any) => b.folder_id === selectedFolder)
    : bookmarkList;

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-6 border border-white/20">
      <div className="flex items-center gap-2 mb-4">
        <Bookmark className="w-6 h-6 text-purple-300" />
        <h2 className="text-2xl font-semibold text-white">Marcadores</h2>
      </div>

      {folderList.length > 0 && (
        <div className="mb-4 flex gap-2 flex-wrap">
          <button
            onClick={() => setSelectedFolder(null)}
            className={`px-3 py-1 rounded-lg transition-colors ${
              selectedFolder === null
                ? 'bg-purple-600 text-white'
                : 'bg-white/10 text-gray-300 hover:bg-white/20'
            }`}
          >
            Todos
          </button>
          {folderList.map((folder: any) => (
            <button
              key={folder.id}
              onClick={() => setSelectedFolder(folder.id)}
              className={`px-3 py-1 rounded-lg transition-colors flex items-center gap-2 ${
                selectedFolder === folder.id
                  ? 'bg-purple-600 text-white'
                  : 'bg-white/10 text-gray-300 hover:bg-white/20'
              }`}
            >
              <Folder className="w-4 h-4" />
              {folder.name}
            </button>
          ))}
        </div>
      )}

      <div className="space-y-2 max-h-96 overflow-y-auto">
        {filteredBookmarks.length === 0 ? (
          <div className="text-center py-12">
            <Bookmark className="w-16 h-16 text-gray-500 mx-auto mb-4" />
            <p className="text-gray-400">No hay marcadores</p>
          </div>
        ) : (
          filteredBookmarks.map((bookmark: any) => (
            <div
              key={bookmark.id}
              className="flex items-center gap-3 p-3 bg-white/5 hover:bg-white/10 rounded-lg transition-colors"
            >
              <BookmarkCheck className="w-5 h-5 text-purple-300 flex-shrink-0" />
              <div className="flex-1 min-w-0">
                <p className="text-white font-medium truncate">
                  {bookmark.track_name || 'Canción desconocida'}
                </p>
                <p className="text-sm text-gray-300 truncate">
                  {bookmark.artists || 'Artista desconocido'}
                </p>
              </div>
              <button
                onClick={() => removeBookmarkMutation.mutate(bookmark.id)}
                className="p-2 text-red-400 hover:text-red-300 hover:bg-red-500/20 rounded-lg transition-colors"
              >
                ×
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  );
}


