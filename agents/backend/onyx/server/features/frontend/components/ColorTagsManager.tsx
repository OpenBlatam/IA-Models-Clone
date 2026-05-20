'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { FiTag, FiPlus, FiX, FiEdit2 } from 'react-icons/fi';

interface ColorTag {
  id: string;
  name: string;
  color: string;
  count?: number;
}

const defaultColors = [
  { name: 'Rojo', value: '#ef4444' },
  { name: 'Naranja', value: '#f97316' },
  { name: 'Amarillo', value: '#eab308' },
  { name: 'Verde', value: '#22c55e' },
  { name: 'Azul', value: '#3b82f6' },
  { name: 'Índigo', value: '#6366f1' },
  { name: 'Púrpura', value: '#a855f7' },
  { name: 'Rosa', value: '#ec4899' },
  { name: 'Gris', value: '#6b7280' },
];

interface ColorTagsManagerProps {
  selectedTags: string[];
  onChange: (tags: string[]) => void;
  taskId?: string;
}

export default function ColorTagsManager({
  selectedTags,
  onChange,
  taskId,
}: ColorTagsManagerProps) {
  const [tags, setTags] = useState<ColorTag[]>([]);
  const [isCreating, setIsCreating] = useState(false);
  const [editingTag, setEditingTag] = useState<ColorTag | null>(null);
  const [newTagName, setNewTagName] = useState('');
  const [newTagColor, setNewTagColor] = useState(defaultColors[0].value);

  useEffect(() => {
    loadTags();
  }, []);

  const loadTags = () => {
    const stored = localStorage.getItem('bul_color_tags');
    if (stored) {
      setTags(JSON.parse(stored));
    } else {
      // Create default tags
      const defaults: ColorTag[] = defaultColors.slice(0, 5).map((color, index) => ({
        id: `tag_${index}`,
        name: color.name,
        color: color.value,
      }));
      setTags(defaults);
      localStorage.setItem('bul_color_tags', JSON.stringify(defaults));
    }
  };

  const saveTags = (updatedTags: ColorTag[]) => {
    setTags(updatedTags);
    localStorage.setItem('bul_color_tags', JSON.stringify(updatedTags));
  };

  const createTag = () => {
    if (!newTagName.trim()) return;

    const newTag: ColorTag = {
      id: `tag_${Date.now()}`,
      name: newTagName.trim(),
      color: newTagColor,
    };

    const updated = [...tags, newTag];
    saveTags(updated);
    setNewTagName('');
    setNewTagColor(defaultColors[0].value);
    setIsCreating(false);
  };

  const updateTag = (tagId: string, updates: Partial<ColorTag>) => {
    const updated = tags.map((tag) =>
      tag.id === tagId ? { ...tag, ...updates } : tag
    );
    saveTags(updated);
    setEditingTag(null);
  };

  const deleteTag = (tagId: string) => {
    const updated = tags.filter((tag) => tag.id !== tagId);
    saveTags(updated);
    // Remove from selected tags if present
    onChange(selectedTags.filter((id) => id !== tagId));
  };

  const toggleTag = (tagId: string) => {
    if (selectedTags.includes(tagId)) {
      onChange(selectedTags.filter((id) => id !== tagId));
    } else {
      onChange([...selectedTags, tagId]);
    }
  };

  const getTagById = (tagId: string) => tags.find((t) => t.id === tagId);

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
          Etiquetas con Color
        </label>
        <button
          onClick={() => setIsCreating(true)}
          className="btn btn-secondary text-xs"
        >
          <FiPlus size={14} className="mr-1" />
          Nueva
        </button>
      </div>

      {/* Selected Tags */}
      {selectedTags.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {selectedTags.map((tagId) => {
            const tag = getTagById(tagId);
            if (!tag) return null;
            return (
              <motion.button
                key={tagId}
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                onClick={() => toggleTag(tagId)}
                className="flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium text-white transition-all hover:opacity-80"
                style={{ backgroundColor: tag.color }}
              >
                <FiTag size={14} />
                {tag.name}
                <FiX size={12} />
              </motion.button>
            );
          })}
        </div>
      )}

      {/* Available Tags */}
      <div className="flex flex-wrap gap-2">
        {tags.map((tag) => {
          const isSelected = selectedTags.includes(tag.id);
          return (
            <motion.button
              key={tag.id}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => toggleTag(tag.id)}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium transition-all ${
                isSelected
                  ? 'text-white opacity-100'
                  : 'text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 opacity-60 hover:opacity-100'
              }`}
              style={isSelected ? { backgroundColor: tag.color } : {}}
            >
              <FiTag size={14} />
              {tag.name}
            </motion.button>
          );
        })}
      </div>

      {/* Create Tag Modal */}
      {isCreating && (
        <div className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg space-y-3">
          <input
            type="text"
            value={newTagName}
            onChange={(e) => setNewTagName(e.target.value)}
            placeholder="Nombre de la etiqueta"
            className="input w-full"
            autoFocus
            onKeyDown={(e) => {
              if (e.key === 'Enter') createTag();
              if (e.key === 'Escape') setIsCreating(false);
            }}
          />
          <div className="flex flex-wrap gap-2">
            {defaultColors.map((color) => (
              <button
                key={color.value}
                onClick={() => setNewTagColor(color.value)}
                className={`w-8 h-8 rounded-full border-2 transition-all ${
                  newTagColor === color.value
                    ? 'border-gray-900 dark:border-white scale-110'
                    : 'border-gray-300 dark:border-gray-600'
                }`}
                style={{ backgroundColor: color.value }}
                title={color.name}
              />
            ))}
          </div>
          <div className="flex gap-2">
            <button onClick={createTag} className="btn btn-primary flex-1 text-sm">
              Crear
            </button>
            <button
              onClick={() => {
                setIsCreating(false);
                setNewTagName('');
              }}
              className="btn btn-secondary text-sm"
            >
              Cancelar
            </button>
          </div>
        </div>
      )}

      {/* Edit Tag (inline) */}
      {editingTag && (
        <div className="p-4 bg-gray-50 dark:bg-gray-700 rounded-lg space-y-3">
          <input
            type="text"
            value={editingTag.name}
            onChange={(e) => setEditingTag({ ...editingTag, name: e.target.value })}
            className="input w-full"
            autoFocus
          />
          <div className="flex flex-wrap gap-2">
            {defaultColors.map((color) => (
              <button
                key={color.value}
                onClick={() => setEditingTag({ ...editingTag, color: color.value })}
                className={`w-8 h-8 rounded-full border-2 transition-all ${
                  editingTag.color === color.value
                    ? 'border-gray-900 dark:border-white scale-110'
                    : 'border-gray-300 dark:border-gray-600'
                }`}
                style={{ backgroundColor: color.value }}
                title={color.name}
              />
            ))}
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => {
                updateTag(editingTag.id, editingTag);
              }}
              className="btn btn-primary flex-1 text-sm"
            >
              Guardar
            </button>
            <button
              onClick={() => setEditingTag(null)}
              className="btn btn-secondary text-sm"
            >
              Cancelar
            </button>
            <button
              onClick={() => {
                deleteTag(editingTag.id);
                setEditingTag(null);
              }}
              className="btn btn-danger text-sm"
            >
              Eliminar
            </button>
          </div>
        </div>
      )}

      {/* Tag Management (for editing) */}
      {tags.length > 0 && (
        <div className="pt-2 border-t border-gray-200 dark:border-gray-700">
          <button
            onClick={() => setEditingTag(tags[0])}
            className="text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300"
          >
            Gestionar etiquetas
          </button>
        </div>
      )}
    </div>
  );
}


