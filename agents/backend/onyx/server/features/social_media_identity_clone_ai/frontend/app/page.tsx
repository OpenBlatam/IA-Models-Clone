'use client';

import { useRouter } from 'next/navigation';
import { memo, useCallback } from 'react';
import Card from '@/components/UI/Card';

interface NavigationCardProps {
  href: string;
  icon: string;
  title: string;
  description: string;
  ariaLabel: string;
}

const NavigationCard = memo(({ href, icon, title, description, ariaLabel }: NavigationCardProps): JSX.Element => {
  const router = useRouter();

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLDivElement>): void => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        router.push(href);
      }
    },
    [href, router]
  );

  const handleClick = useCallback((): void => {
    router.push(href);
  }, [href, router]);

  return (
    <Card
      className="hover:shadow-lg transition-all duration-200 cursor-pointer group"
      role="link"
      tabIndex={0}
      aria-label={ariaLabel}
      onClick={handleClick}
      onKeyDown={handleKeyDown}
    >
      <div className="text-5xl mb-4 group-hover:scale-110 transition-transform duration-200">
        {icon}
      </div>
      <h2 className="text-xl font-semibold mb-2 group-hover:text-primary-600 transition-colors">
        {title}
      </h2>
      <p className="text-gray-600">{description}</p>
    </Card>
  );
});

NavigationCard.displayName = 'NavigationCard';

const HomePage = (): JSX.Element => {
  const navigationCards: NavigationCardProps[] = [
    {
      href: '/extract-profile',
      icon: '🔍',
      title: 'Extract Profile',
      description: 'Extract profiles from TikTok, Instagram, or YouTube',
      ariaLabel: 'Navigate to Extract Profile page',
    },
    {
      href: '/build-identity',
      icon: '🏗️',
      title: 'Build Identity',
      description: 'Build a complete identity profile from social media profiles',
      ariaLabel: 'Navigate to Build Identity page',
    },
    {
      href: '/generate-content',
      icon: '✨',
      title: 'Generate Content',
      description: 'Generate authentic content based on cloned identity',
      ariaLabel: 'Navigate to Generate Content page',
    },
    {
      href: '/dashboard',
      icon: '📊',
      title: 'Dashboard',
      description: 'View analytics, metrics, and system overview',
      ariaLabel: 'Navigate to Dashboard page',
    },
    {
      href: '/identities',
      icon: '👤',
      title: 'Identities',
      description: 'Manage and view all cloned identities',
      ariaLabel: 'Navigate to Identities page',
    },
    {
      href: '/analytics',
      icon: '📈',
      title: 'Analytics',
      description: 'Deep dive into analytics and trends',
      ariaLabel: 'Navigate to Analytics page',
    },
    {
      href: '/search',
      icon: '🔎',
      title: 'Search',
      description: 'Search identities and content',
      ariaLabel: 'Navigate to Search page',
    },
    {
      href: '/templates',
      icon: '📝',
      title: 'Templates',
      description: 'Create and manage content templates',
      ariaLabel: 'Navigate to Templates page',
    },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-50 via-white to-primary-50">
      <div className="container mx-auto px-4 py-16">
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-gray-900 mb-4">
            Social Media Identity Clone AI
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Clone social media identities and generate authentic content based on your cloned
            identity
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 max-w-7xl mx-auto">
          {navigationCards.map((card) => (
            <NavigationCard key={card.href} {...card} />
          ))}
        </div>
      </div>
    </div>
  );
};

export default HomePage;
