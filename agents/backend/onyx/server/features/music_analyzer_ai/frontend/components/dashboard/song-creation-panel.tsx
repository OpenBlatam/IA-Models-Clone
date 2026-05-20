/**
 * Song Creation Panel component.
 * Center panel for creating new songs with description, audio, lyrics inputs.
 */

'use client';

import { useState } from 'react';
import { Music, Plus, Grid } from 'lucide-react';
import { Button, Textarea } from '@/components/ui';

/**
 * Song Creation Panel component.
 * Provides interface for creating new songs.
 *
 * @returns Song Creation Panel component
 */
export function SongCreationPanel() {
  const [songDescription, setSongDescription] = useState('Hip-hop, R&B, upbeat');
  const [isInstrumental, setIsInstrumental] = useState(true);
  const [inspirationTags, setInspirationTags] = useState<string[]>([]);

  const availableTags = [
    'steady',
    'pop',
    'redneck',
    'reggae',
    'rock',
    'jazz',
    'electronic',
    'classical',
  ];

  const handleTagClick = (tag: string) => {
    if (inspirationTags.includes(tag)) {
      setInspirationTags(inspirationTags.filter((t) => t !== tag));
    } else {
      setInspirationTags([...inspirationTags, tag]);
    }
  };

  const handleCreate = () => {
    // TODO: Implement song creation logic
    console.log('Creating song with:', {
      description: songDescription,
      instrumental: isInstrumental,
      inspiration: inspirationTags,
    });
  };

  return (
    <div className="h-full p-6 space-y-5 bg-black">
      {/* Song Description Section */}
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Song Description
        </label>
        <div className="relative">
          <Textarea
            value={songDescription}
            onChange={(e) => setSongDescription(e.target.value)}
            placeholder="Describe your song..."
            className="w-full min-h-[100px] bg-white/5 border border-white/20 text-white placeholder-gray-500 resize-none pr-10"
          />
          <button className="absolute top-3 right-3 p-1.5 hover:bg-white/10 rounded transition-colors">
            <Grid className="w-4 h-4 text-gray-400" />
          </button>
        </div>
      </div>

      {/* Input Type Section */}
      <div>
        <div className="flex items-center gap-3">
          <button className="flex items-center gap-2 px-3 py-2 text-sm font-medium bg-white/5 border border-white/10 rounded-lg text-gray-300 hover:bg-white/10 hover:text-white transition-colors">
            <Plus className="w-4 h-4" />
            Audio
          </button>
          <button className="flex items-center gap-2 px-3 py-2 text-sm font-medium bg-white/5 border border-white/10 rounded-lg text-gray-300 hover:bg-white/10 hover:text-white transition-colors">
            <Plus className="w-4 h-4" />
            Lyrics
          </button>
          
          {/* Instrumental Toggle - inline with buttons */}
          <button
            onClick={() => setIsInstrumental(!isInstrumental)}
            className={`
              ml-auto px-4 py-2 rounded-lg transition-colors text-sm font-medium
              ${
                isInstrumental
                  ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30'
                  : 'bg-white/5 text-gray-400 border border-white/10 hover:bg-white/10'
              }
            `}
          >
            Instrumental
          </button>
        </div>
      </div>

      {/* Inspiration Section */}
      <div>
        <label className="block text-sm font-medium text-gray-300 mb-3">
          Inspiration
        </label>
        <div className="flex gap-2 overflow-x-auto pb-2 scrollbar-hide">
          {availableTags.map((tag) => (
            <button
              key={tag}
              onClick={() => handleTagClick(tag)}
              className={`
                px-3 py-1.5 rounded-lg text-sm font-medium transition-colors whitespace-nowrap flex-shrink-0
                ${
                  inspirationTags.includes(tag)
                    ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30'
                    : 'bg-white/5 text-gray-400 border border-white/10 hover:bg-white/10 hover:text-white'
                }
              `}
            >
              + {tag}
            </button>
          ))}
        </div>
      </div>

      {/* Create Button */}
      <Button
        variant="primary"
        size="lg"
        onClick={handleCreate}
        className="w-full bg-gradient-to-r from-orange-500 to-orange-600 hover:from-orange-600 hover:to-orange-700 flex items-center justify-center gap-2"
      >
        <Music className="w-5 h-5" />
        Create
      </Button>
    </div>
  );
}

