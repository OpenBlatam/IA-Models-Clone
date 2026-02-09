import Link from 'next/link';
import PageLayout from '@/components/Layout/PageLayout';
import Button from '@/components/UI/Button';

const NotFound = (): JSX.Element => {
  return (
    <PageLayout>
      <div className="max-w-md mx-auto text-center">
        <h1 className="text-6xl font-bold mb-4">404</h1>
        <p className="text-xl text-gray-600 mb-8">Page not found</p>
        <Link href="/">
          <Button>Go Home</Button>
        </Link>
      </div>
    </PageLayout>
  );
};

export default NotFound;



