'use client';

import { ArrowLeft } from 'lucide-react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { cn } from '@/lib/utils';

interface BackButtonProps {
  href?: string;
  label?: string;
  className?: string;
}

const BackButton = ({ href, label = 'Volver', className }: BackButtonProps) => {
  const router = useRouter();

  const handleClick = () => {
    if (href) {
      router.push(href);
    } else {
      router.back();
    }
  };

  const content = (
    <div
      className={cn('inline-flex items-center text-gray-600 hover:text-gray-900 cursor-pointer', className)}
      onClick={handleClick}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          handleClick();
        }
      }}
      tabIndex={0}
      role="button"
      aria-label={label}
    >
      <ArrowLeft className="w-4 h-4 mr-2" />
      {label}
    </div>
  );

  if (href) {
    return (
      <Link href={href} className={cn('inline-flex items-center text-gray-600 hover:text-gray-900', className)}>
        <ArrowLeft className="w-4 h-4 mr-2" />
        {label}
      </Link>
    );
  }

  return content;
};

export { BackButton };

