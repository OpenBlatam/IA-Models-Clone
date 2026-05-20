import { Loader2 } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from './card';
import type { LoadingStateProps } from '@/lib/types/components';

export const LoadingState = ({ title = 'Cargando...' }: LoadingStateProps): JSX.Element => {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex items-center justify-center py-8">
          <Loader2 className="h-8 w-8 animate-spin text-gray-400" />
        </div>
      </CardContent>
    </Card>
  );
};

