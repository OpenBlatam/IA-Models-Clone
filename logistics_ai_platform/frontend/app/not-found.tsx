import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

const NotFound = () => {
  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-4">
      <Card className="w-full max-w-md text-center">
        <CardHeader>
          <CardTitle className="text-4xl">404</CardTitle>
          <CardDescription>Page Not Found</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-muted-foreground">
            The page you are looking for does not exist or has been moved.
          </p>
          <Link href="/en/dashboard">
            <Button className="w-full" aria-label="Go to dashboard">
              Go to Dashboard
            </Button>
          </Link>
        </CardContent>
      </Card>
    </div>
  );
};

export default NotFound;




