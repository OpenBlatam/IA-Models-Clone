import { useState } from 'react';
import { cn } from '@/lib/utils';

interface ExpandableSectionProps {
  title: string;
  children: React.ReactNode;
  defaultExpanded?: boolean;
  className?: string;
}

const ExpandableSection = ({
  title,
  children,
  defaultExpanded = false,
  className = '',
}: ExpandableSectionProps): JSX.Element => {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

  const handleToggle = (): void => {
    setIsExpanded(!isExpanded);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLButtonElement>): void => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleToggle();
    }
  };

  return (
    <div className={cn('border rounded-lg', className)}>
      <button
        onClick={handleToggle}
        onKeyDown={handleKeyDown}
        className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-50 transition-colors"
        aria-expanded={isExpanded}
        aria-controls={`expandable-content-${title}`}
        tabIndex={0}
      >
        <span className="font-semibold text-left">{title}</span>
        <span className="text-xl" aria-hidden="true">
          {isExpanded ? '−' : '+'}
        </span>
      </button>
      {isExpanded && (
        <div
          id={`expandable-content-${title}`}
          className="px-4 py-3 border-t"
          role="region"
          aria-labelledby={`expandable-title-${title}`}
        >
          {children}
        </div>
      )}
    </div>
  );
};

export default ExpandableSection;



