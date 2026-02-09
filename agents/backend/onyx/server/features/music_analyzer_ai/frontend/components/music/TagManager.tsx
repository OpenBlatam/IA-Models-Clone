'use client';

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { Tag, Plus, X } from 'lucide-react';
import toast from 'react-hot-toast';

interface TagManagerProps {
  resourceId: string;
  resourceType: 'track' | 'analysis' | 'playlist';
}

export function TagManager({ resourceId, resourceType }: TagManagerProps) {
  const [newTag, setNewTag] = useState('');
  const queryClient = useQueryClient();

  const { data: tagsData } = useQuery({
    queryKey: ['tags', resourceId, resourceType],
    queryFn: () => musicApiService.getTags?.(resourceId, resourceType) || Promise.resolve({ tags: [] }),
  });

  const addTagMutation = useMutation({
    mutationFn: (tag: string) =>
      musicApiService.addTag?.(resourceId, resourceType, [tag]) || Promise.resolve({}),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tags', resourceId, resourceType] });
      toast.success('Tag agregado');
      setNewTag('');
    },
    onError: () => {
      toast.error('Error al agregar tag');
    },
  });

  const removeTagMutation = useMutation({
    mutationFn: (tag: string) =>
      musicApiService.removeTag?.(resourceId, resourceType, [tag]) || Promise.resolve({}),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tags', resourceId, resourceType] });
      toast.success('Tag eliminado');
    },
    onError: () => {
      toast.error('Error al eliminar tag');
    },
  });

  const tags = tagsData?.tags || [];

  const handleAddTag = () => {
    if (!newTag.trim()) {
      toast.error('Ingresa un tag');
      return;
    }
    addTagMutation.mutate(newTag.trim());
  };

  return (
    <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
      <div className="flex items-center gap-2 mb-3">
        <Tag className="w-5 h-5 text-purple-300" />
        <h3 className="text-sm font-semibold text-white">Tags</h3>
      </div>

      <div className="flex flex-wrap gap-2 mb-3">
        {tags.map((tag: string, idx: number) => (
          <div
            key={idx}
            className="flex items-center gap-1 px-3 py-1 bg-purple-500/30 rounded-full"
          >
            <span className="text-sm text-white">{tag}</span>
            <button
              onClick={() => removeTagMutation.mutate(tag)}
              className="text-purple-200 hover:text-white"
            >
              <X className="w-3 h-3" />
            </button>
          </div>
        ))}
      </div>

      <div className="flex gap-2">
        <input
          type="text"
          value={newTag}
          onChange={(e) => setNewTag(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleAddTag()}
          placeholder="Agregar tag..."
          className="flex-1 px-3 py-2 bg-white/20 border border-white/30 rounded-lg text-white text-sm placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-purple-400"
        />
        <button
          onClick={handleAddTag}
          disabled={addTagMutation.isPending}
          className="px-3 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors disabled:opacity-50"
        >
          <Plus className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}


