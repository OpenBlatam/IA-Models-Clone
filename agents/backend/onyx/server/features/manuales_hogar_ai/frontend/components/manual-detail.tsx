'use client';

import { useManual, useRatings } from '@/lib/hooks/use-manuals';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { LoadingState } from './ui/loading-state';
import { ErrorState } from './ui/error-state';
import { FavoriteButton } from './manual/favorite-button';
import { ExportMenu } from './manual/export-menu';
import { RatingForm } from './manual/rating-form';
import { RatingsList } from './manual/ratings-list';
import { ManualMetadata } from './manual/manual-metadata';
import { MESSAGES } from '@/lib/constants';

interface ManualDetailProps {
  manualId: number;
}

export const ManualDetail = ({ manualId }: ManualDetailProps): JSX.Element => {
  const { data: manual, isLoading, error } = useManual(manualId);
  const { data: ratings } = useRatings(manualId);

  if (isLoading) {
    return <LoadingState title={MESSAGES.MANUAL.LOADING} />;
  }

  if (error || !manual) {
    return <ErrorState title="Error" message={MESSAGES.MANUAL.LOAD_ERROR} />;
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <CardTitle className="text-2xl mb-2">{manual.problem_description}</CardTitle>
              <CardDescription>
                <ManualMetadata manual={manual} />
              </CardDescription>
            </div>
            <div className="flex items-center space-x-2">
              <FavoriteButton manualId={manualId} />
              <ExportMenu manualId={manualId} />
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="prose max-w-none">
            <pre className="whitespace-pre-wrap font-sans text-sm bg-gray-50 p-4 rounded-lg">
              {manual.manual_content}
            </pre>
          </div>
        </CardContent>
      </Card>

      <RatingForm manualId={manualId} />
      <RatingsList ratings={ratings} />
    </div>
  );
};

