import Link from 'next/link';
import { Button } from '@/components/ui/button';

const NotFound = () => {
  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center">
      <div className="text-center">
        <h1 className="text-6xl font-bold text-gray-900 mb-4">404</h1>
        <p className="text-xl text-gray-600 mb-8">Página no encontrada</p>
        <Link href="/dashboard">
          <Button variant="primary">Volver al Dashboard</Button>
        </Link>
      </div>
    </div>
  );
};

export default NotFound;

