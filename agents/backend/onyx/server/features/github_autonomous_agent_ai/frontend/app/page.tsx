'use client';

import { useState, useEffect, Suspense } from 'react';
import {
  Header,
  HeroSection,
  FeaturesSection,
  UseCasesSection,
  PricingSection,
  BlogSection,
  DownloadSection,
  Footer,
  DecorativeDots,
} from './components/home';

/**
 * Main Home component - Professional landing page with decorative animations
 * 
 * Features:
 * - SSR compatible with client-side mounting
 * - Optimized performance with component splitting
 * - Accessible navigation and interactions
 * - Responsive design for all screen sizes
 * 
 * @returns {JSX.Element} The home page component
 */
export default function Home() {
  const [isMounted, setIsMounted] = useState(false);

  // Handle client-side mounting for SSR compatibility
  useEffect(() => {
    setIsMounted(true);
  }, []);

  // Don't render decorative dots until mounted (SSR compatibility)
  if (!isMounted) {
    return (
      <div className="min-h-screen bg-white text-black relative" role="main">
        <div className="decorative-dots" aria-hidden="true" />
        <div className="sr-only" role="status" aria-live="polite">
          Loading page content...
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-white text-black relative" role="main">
      <Suspense fallback={<div className="decorative-dots" aria-hidden="true" />}>
        <DecorativeDots />
      </Suspense>
      <Header />
      <HeroSection />
      <FeaturesSection />
      <UseCasesSection />
      <PricingSection />
      <BlogSection />
      <DownloadSection />
      <Footer />
    </div>
  );
}
