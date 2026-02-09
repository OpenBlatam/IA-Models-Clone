import { memo } from 'react';
import Navbar from './Navbar';
import Breadcrumbs from '@/components/UI/Breadcrumbs';
import PageHeader from './PageHeader';
import { cn } from '@/lib/utils';

interface BreadcrumbItem {
  label: string;
  href?: string;
}

interface PageLayoutProps {
  children: React.ReactNode;
  className?: string;
  breadcrumbs?: BreadcrumbItem[];
  title?: string;
  description?: string;
}

const PageLayout = memo(({
  children,
  className = '',
  breadcrumbs,
  title,
  description,
}: PageLayoutProps): JSX.Element => {
  return (
    <div className={cn('min-h-screen bg-gray-50', className)}>
      <Navbar />
      <main className="container mx-auto px-4 py-8" role="main">
        {breadcrumbs && breadcrumbs.length > 0 && (
          <div className="mb-6">
            <Breadcrumbs items={breadcrumbs} />
          </div>
        )}
        {title && (
          <PageHeader
            title={title}
            description={description}
            className="mb-8"
          />
        )}
        {children}
      </main>
    </div>
  );
});

PageLayout.displayName = 'PageLayout';

export default PageLayout;
