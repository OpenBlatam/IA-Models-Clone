'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { StarRating } from './star-rating';
import { formatManualDate } from '@/lib/utils/format';
import { pluralize } from '@/lib/utils/pluralize';
import { EmptyState } from '../ui/empty-state';
import { MessageSquare } from 'lucide-react';
import { MESSAGES } from '@/lib/constants';
import type { RatingsListProps } from '@/lib/types/components';

export const RatingsList = ({ ratings }: RatingsListProps): JSX.Element => {
  if (!ratings || ratings.length === 0) {
    return (
      <EmptyState
        icon={MessageSquare}
        title={MESSAGES.EMPTY.NO_RATINGS}
        description={MESSAGES.EMPTY.NO_RATINGS_DESC}
      />
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Calificaciones</CardTitle>
        <CardDescription>
          {ratings.length} {pluralize(ratings.length, 'calificación', 'calificaciones')}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {ratings.map((ratingItem) => (
            <div key={ratingItem.id} className="border-b pb-4 last:border-0">
              <div className="flex items-center space-x-4 mb-2">
                <StarRating rating={ratingItem.rating} readonly size="sm" />
                <span className="text-sm text-gray-600">
                  {formatManualDate(ratingItem.created_at)}
                </span>
              </div>
              {ratingItem.comment && (
                <p className="text-sm text-gray-700 mt-2">{ratingItem.comment}</p>
              )}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};

