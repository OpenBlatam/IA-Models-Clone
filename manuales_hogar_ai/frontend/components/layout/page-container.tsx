import { Navigation } from '../navigation';
import type { PageContainerProps } from '@/lib/types/components';

export const PageContainer = ({ children, className = '' }: PageContainerProps): JSX.Element => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <Navigation />
      <main className={`container mx-auto px-4 py-8 ${className}`}>
        {children}
      </main>
    </div>
  );
};

