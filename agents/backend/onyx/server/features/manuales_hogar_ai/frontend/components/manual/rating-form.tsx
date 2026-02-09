'use client';

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { FormField } from '../ui/form-field';
import { FormTextarea } from '../ui/form-textarea';
import { SubmitButton } from '../ui/submit-button';
import { StarRating } from './star-rating';
import { useAddRating } from '@/lib/hooks/use-manuals';
import { showErrorToast, showSuccessToast } from '@/lib/utils/error-handler';
import { ratingSchema } from '@/lib/utils/validation';
import { DEFAULT_USER_ID, MESSAGES } from '@/lib/constants';
import type { RatingFormProps } from '@/lib/types/components';

export const RatingForm = ({ manualId, userId = DEFAULT_USER_ID }: RatingFormProps): JSX.Element => {
  const [rating, setRating] = useState(0);
  const [comment, setComment] = useState('');
  const addRating = useAddRating();

  const handleSubmit = async (): Promise<void> => {
    if (rating === 0) {
      showErrorToast(new Error(MESSAGES.RATING.SELECT_REQUIRED));
      return;
    }

    try {
      const validatedData = ratingSchema.parse({ rating, comment: comment || undefined });
      await addRating.mutateAsync({
        manualId,
        request: validatedData,
        userId,
      });
      showSuccessToast(MESSAGES.RATING.ADD_SUCCESS);
      setRating(0);
      setComment('');
    } catch (error) {
      showErrorToast(error);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Calificar Manual</CardTitle>
        <CardDescription>
          Comparte tu opinión sobre este manual
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <FormField label="Calificación" htmlFor="rating">
          <StarRating
            rating={rating}
            onRatingChange={setRating}
            size="lg"
          />
        </FormField>
        <FormTextarea
          id="comment"
          label="Comentario (opcional)"
          value={comment}
          onChange={(e) => setComment(e.target.value)}
          placeholder="Escribe tu comentario..."
          rows={3}
          maxLength={500}
        />
        <SubmitButton
          onClick={handleSubmit}
          isLoading={addRating.isPending}
          loadingText="Enviando..."
          disabled={rating === 0}
          asSubmit={false}
        >
          Enviar Calificación
        </SubmitButton>
      </CardContent>
    </Card>
  );
};

