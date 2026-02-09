import { useCallback } from 'react';
import { useRouter } from 'next/navigation';

interface UseKeyboardNavigationOptions {
  href?: string;
  onAction?: () => void;
}

export const useKeyboardNavigation = ({ href, onAction }: UseKeyboardNavigationOptions) => {
  const router = useRouter();

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLElement>): void => {
      if (e.key !== 'Enter' && e.key !== ' ') {
        return;
      }

      e.preventDefault();

      if (onAction) {
        onAction();
        return;
      }

      if (href) {
        router.push(href);
      }
    },
    [href, onAction, router]
  );

  return { handleKeyDown };
};



