'use client';

import { useRouter } from 'next/navigation';
import { motion, type Variants } from 'framer-motion';
import { Header } from './components/home';
import { useCallback } from 'react';

/**
 * Animation configuration constants
 * Centralized animation settings for consistent behavior
 */
const ANIMATION_CONFIG = {
  container: {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.6 },
  },
  heading404: {
    initial: { opacity: 0, scale: 0.8 },
    animate: { opacity: 1, scale: 1 },
    transition: { duration: 0.5, delay: 0.1 },
  },
  title: {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.6, delay: 0.2 },
  },
  description: {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.6, delay: 0.3 },
  },
  buttons: {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.6, delay: 0.4 },
  },
  buttonHover: {
    scale: 1.02,
  },
  buttonTap: {
    scale: 0.98,
  },
  buttonSpring: {
    type: "spring" as const,
    stiffness: 400,
    damping: 17,
  },
} as const;

/**
 * Content constants
 * Centralized text content for easy maintenance and i18n support
 */
const CONTENT = {
  heading: '404',
  title: 'Page Not Found',
  description: "The page you're looking for doesn't exist or has been moved.",
  buttons: {
    goHome: 'Go to Home',
    goBack: 'Go Back',
  },
  ariaLabels: {
    goHome: 'Navigate to home page',
    goBack: 'Navigate back to previous page',
  },
} as const;

/**
 * Custom hook for navigation handlers
 * Provides memoized navigation functions with error handling
 */
function useNotFoundNavigation() {
  const router = useRouter();

  const handleGoHome = useCallback(() => {
    try {
      router.push('/');
    } catch (error) {
      console.error('Navigation error:', error);
      // Fallback to window.location if router fails
      window.location.href = '/';
    }
  }, [router]);

  const handleGoBack = useCallback(() => {
    try {
      if (window.history.length > 1) {
        router.back();
      } else {
        // If no history, go to home instead
        router.push('/');
      }
    } catch (error) {
      console.error('Navigation error:', error);
      // Fallback to window.location if router fails
      window.location.href = '/';
    }
  }, [router]);

  return { handleGoHome, handleGoBack };
}

/**
 * Action Button Component
 * Reusable button component with consistent styling and animations
 */
interface ActionButtonProps {
  onClick: () => void;
  children: React.ReactNode;
  ariaLabel: string;
  variant?: 'primary' | 'secondary';
}

function ActionButton({ 
  onClick, 
  children, 
  ariaLabel, 
  variant = 'primary' 
}: ActionButtonProps) {
  const baseClasses = "px-6 py-3 rounded-md font-normal text-sm focus:outline-none focus:ring-2 focus:ring-offset-2 transition-all duration-200";
  
  const variantClasses = variant === 'primary'
    ? "bg-black text-white hover:bg-[#1a1a1a] focus:ring-black"
    : "text-black underline hover:opacity-70 focus:ring-gray-400";

  return (
    <motion.button
      onClick={onClick}
      className={`${baseClasses} ${variantClasses}`}
      whileHover={ANIMATION_CONFIG.buttonHover}
      whileTap={ANIMATION_CONFIG.buttonTap}
      transition={ANIMATION_CONFIG.buttonSpring}
      aria-label={ariaLabel}
      type="button"
    >
      {children}
    </motion.button>
  );
}

/**
 * 404 Not Found Page Component
 * 
 * Displays when a route is not found in the application.
 * Matches the design system of the main application.
 * 
 * @features
 * - Accessible navigation buttons with proper ARIA labels
 * - Smooth animations with framer-motion
 * - Responsive design (mobile and desktop)
 * - Consistent styling with the rest of the app
 * - Error handling for navigation
 * - SEO-friendly structure
 * 
 * @accessibility
 * - Semantic HTML structure
 * - Proper heading hierarchy (h1, h2)
 * - Keyboard navigation support
 * - Screen reader friendly
 * - Focus management
 */
export default function NotFound() {
  const { handleGoHome, handleGoBack } = useNotFoundNavigation();

  return (
    <div className="min-h-screen bg-white text-black relative">
      <Header />
      
      <main 
        className="flex flex-col items-center justify-center min-h-[calc(100vh-200px)] px-4 py-16"
        role="main"
        aria-label="404 error page"
      >
        <motion.div
          className="text-center max-w-2xl mx-auto"
          {...ANIMATION_CONFIG.container}
        >
          {/* 404 Number */}
          <motion.h1
            className="text-8xl md:text-9xl font-normal text-black mb-4"
            {...ANIMATION_CONFIG.heading404}
          >
            {CONTENT.heading}
          </motion.h1>

          {/* Title */}
          <motion.h2
            className="text-3xl md:text-4xl font-normal text-black mb-4"
            {...ANIMATION_CONFIG.title}
          >
            {CONTENT.title}
          </motion.h2>

          {/* Description */}
          <motion.p
            className="text-base md:text-lg text-gray-600 mb-8 leading-relaxed"
            {...ANIMATION_CONFIG.description}
          >
            {CONTENT.description}
          </motion.p>

          {/* Action Buttons */}
          <motion.nav
            className="flex flex-col sm:flex-row gap-4 justify-center items-center"
            {...ANIMATION_CONFIG.buttons}
            aria-label="Navigation options"
          >
            <ActionButton
              onClick={handleGoHome}
              ariaLabel={CONTENT.ariaLabels.goHome}
              variant="primary"
            >
              {CONTENT.buttons.goHome}
            </ActionButton>
            
            <ActionButton
              onClick={handleGoBack}
              ariaLabel={CONTENT.ariaLabels.goBack}
              variant="secondary"
            >
              {CONTENT.buttons.goBack}
            </ActionButton>
          </motion.nav>
        </motion.div>
      </main>
    </div>
  );
}
