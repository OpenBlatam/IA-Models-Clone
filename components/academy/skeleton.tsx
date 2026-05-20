"use client";

import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

interface AcademySkeletonProps {
  variant?: "grid" | "list";
  className?: string;
}

export function AcademySkeleton({ variant = "grid", className }: AcademySkeletonProps) {
  return (
    <div className={cn(variant === "grid" ? "space-y-3" : "flex gap-4", className)}>
      <div className={variant === "grid" ? "w-full" : "w-64 flex-shrink-0"}>
        <Skeleton className="aspect-video rounded-xl" />
      </div>
      <div className={variant === "grid" ? "space-y-2" : "flex-1 space-y-3"}>
        <div className="space-y-2">
          <Skeleton className="h-5 w-3/4" />
          <Skeleton className="h-4 w-full" />
        </div>
        <div className="flex items-center gap-2">
          <Skeleton className="h-10 w-10 rounded-full" />
          <Skeleton className="h-4 w-24" />
        </div>
        <div className="space-y-2">
          <div className="flex gap-2">
            <Skeleton className="h-5 w-16" />
            <Skeleton className="h-5 w-16" />
          </div>
          <Skeleton className="h-1 w-full" />
        </div>
      </div>
    </div>
  );
} 