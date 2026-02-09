import { memo, useEffect, useState } from 'react';
import { createPortal } from 'react-dom';

interface PortalProps {
  children: React.ReactNode;
  container?: HTMLElement | null;
  className?: string;
}

const Portal = memo(({ children, container, className = '' }: PortalProps): JSX.Element | null => {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted || typeof window === 'undefined') {
    return null;
  }

  const targetContainer = container || document.body;

  if (className) {
    const wrapper = document.createElement('div');
    wrapper.className = className;
    targetContainer.appendChild(wrapper);

    return createPortal(children, wrapper);
  }

  return createPortal(children, targetContainer);
});

Portal.displayName = 'Portal';

export default Portal;



