'use client';

import { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { musicApiService } from '@/lib/api/music-api';
import { Star } from 'lucide-react';
import toast from 'react-hot-toast';

interface TrackRatingProps {
  trackId: string;
  initialRating?: number;
}

export function TrackRating({ trackId, initialRating = 0 }: TrackRatingProps) {
  const [rating, setRating] = useState(initialRating);
  const [hoverRating, setHoverRating] = useState(0);
  const queryClient = useQueryClient();

  const rateMutation = useMutation({
    mutationFn: (newRating: number) =>
      musicApiService.rateTrack?.(trackId, newRating) || Promise.resolve({}),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['track-rating', trackId] });
      toast.success('Calificación guardada');
    },
    onError: () => {
      toast.error('Error al guardar calificación');
    },
  });

  const handleRating = (newRating: number) => {
    setRating(newRating);
    rateMutation.mutate(newRating);
  };

  return (
    <div className="flex items-center gap-2">
      <span className="text-sm text-gray-400">Calificar:</span>
      <div className="flex gap-1">
        {[1, 2, 3, 4, 5].map((star) => (
          <button
            key={star}
            onClick={() => handleRating(star)}
            onMouseEnter={() => setHoverRating(star)}
            onMouseLeave={() => setHoverRating(0)}
            className="transition-colors"
          >
            <Star
              className={`w-5 h-5 ${
                star <= (hoverRating || rating)
                  ? 'fill-yellow-400 text-yellow-400'
                  : 'text-gray-500'
              }`}
            />
          </button>
        ))}
      </div>
      {rating > 0 && (
        <span className="text-sm text-white font-medium">{rating}/5</span>
      )}
    </div>
  );
}


