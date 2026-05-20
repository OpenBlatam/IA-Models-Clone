"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { usePerformance } from "@/components/providers/performance-provider";
import { cn } from "@/lib/utils";
import { startTransition } from "react";

interface OptimizedLinkProps {
  href: string;
  children: React.ReactNode;
  className?: string;
  prefetch?: boolean;
  replace?: boolean;
  scroll?: boolean;
  shallow?: boolean;
}

export function OptimizedLink({ 
  href, 
  children, 
  className, 
  prefetch, 
  replace = false,
  scroll = true,
  shallow = false,
  ...props 
}: OptimizedLinkProps) {
  const router = useRouter();
  const { prefetchEnabled } = usePerformance();
  
  const shouldPrefetch = prefetch !== false && prefetchEnabled;

  const handleClick = (e: React.MouseEvent) => {
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    
    e.preventDefault();
    
    startTransition(() => {
      if (replace) {
        router.replace(href, { scroll });
      } else {
        router.push(href, { scroll });
      }
    });
  };

  return (
    <Link
      href={href}
      prefetch={shouldPrefetch}
      className={cn("transition-colors duration-200", className)}
      onClick={handleClick}
      {...props}
    >
      {children}
    </Link>
  );
}
