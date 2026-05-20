import { Card, CardContent, CardHeader, CardTitle } from './card';
import type { ErrorStateProps } from '@/lib/types/components';

export const ErrorState = ({
  title = 'Error',
  message = 'Ocurrió un error al cargar los datos',
}: ErrorStateProps): JSX.Element => {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-red-500">{message}</p>
      </CardContent>
    </Card>
  );
};

