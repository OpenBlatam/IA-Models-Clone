import { memo } from 'react';
import { cn } from '@/lib/utils';

interface TagsListProps {
  tags: string[];
  variant?: 'primary' | 'gray' | 'green';
  maxTags?: number;
  className?: string;
}

const VARIANT_CLASSES = {
  primary: 'bg-primary-100 text-primary-700',
  gray: 'bg-gray-100 text-gray-700',
  green: 'bg-green-100 text-green-700',
};

const TagsList = memo(({ tags, variant = 'primary', maxTags, className = '' }: TagsListProps): JSX.Element => {
  if (tags.length === 0) {
    return null;
  }

  const displayTags = maxTags ? tags.slice(0, maxTags) : tags;
  const variantClass = VARIANT_CLASSES[variant];

  return (
    <div className={cn('flex flex-wrap gap-2', className)}>
      {displayTags.map((tag, idx) => (
        <span key={idx} className={cn('px-3 py-1 rounded-full text-sm', variantClass)}>
          {tag}
        </span>
      ))}
      {maxTags && tags.length > maxTags && (
        <span className="px-3 py-1 rounded-full text-sm bg-gray-100 text-gray-500">
          +{tags.length - maxTags} more
        </span>
      )}
    </div>
  );
});

TagsList.displayName = 'TagsList';

export default TagsList;



