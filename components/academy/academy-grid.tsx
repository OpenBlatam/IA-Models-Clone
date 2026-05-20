"use client";

import { Academy } from "@/lib/types/academy";
import { AcademyCard } from "./academy-card";
import { AcademySkeleton } from "./skeleton";
import { useInfiniteScroll } from "@/hooks/use-infinite-scroll";
import { cn } from "@/lib/utils";

interface AcademyGridProps {
  academies: Academy[];
  onLoadMore: () => Promise<void>;
  hasMore: boolean;
  isLoading: boolean;
  viewMode: "grid" | "list";
  className?: string;
}

export function AcademyGrid({
  academies,
  onLoadMore,
  hasMore,
  isLoading,
  viewMode,
  className,
}: AcademyGridProps) {
  const { ref } = useInfiniteScroll({
    onLoadMore,
    hasMore,
    isLoading,
  });

  return (
    <div className={cn("space-y-6", className)}>
      <div
        className={cn(
          "grid gap-6",
          viewMode === "grid"
            ? "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3"
            : "grid-cols-1"
        )}
      >
        {academies.map((academy) => (
          <AcademyCard
            key={academy.id}
            academy={academy}
            variant={viewMode}
          />
        ))}
      </div>

      {/* Loading States */}
      {isLoading && (
        <div
          className={cn(
            "grid gap-6",
            viewMode === "grid"
              ? "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3"
              : "grid-cols-1"
          )}
        >
          {Array.from({ length: 3 }).map((_, index) => (
            <AcademySkeleton key={index} variant={viewMode} />
          ))}
        </div>
      )}

      {/* Infinite Scroll Trigger */}
      {hasMore && <div ref={ref} className="h-4" />}
    </div>
  );
} 