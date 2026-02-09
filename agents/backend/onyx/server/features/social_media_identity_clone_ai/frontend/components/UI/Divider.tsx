import { cn } from '@/lib/utils';

interface DividerProps {
  label?: string;
  orientation?: 'horizontal' | 'vertical';
  className?: string;
}

const Divider = ({ label, orientation = 'horizontal', className = '' }: DividerProps): JSX.Element => {
  if (orientation === 'vertical') {
    return (
      <div
        className={cn('w-px bg-gray-200 self-stretch', className)}
        role="separator"
        aria-orientation="vertical"
      />
    );
  }

  if (label) {
    return (
      <div className={cn('flex items-center my-4', className)}>
        <div className="flex-1 border-t border-gray-200" />
        <span className="px-4 text-sm text-gray-500">{label}</span>
        <div className="flex-1 border-t border-gray-200" />
      </div>
    );
  }

  return (
    <hr
      className={cn('border-t border-gray-200 my-4', className)}
      role="separator"
      aria-orientation="horizontal"
    />
  );
};

export default Divider;



