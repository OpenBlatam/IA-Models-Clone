'use client';

export function LoadingSkeleton() {
  return (
    <div className="animate-pulse space-y-4">
      <div className="h-8 bg-white/10 rounded w-3/4" />
      <div className="h-4 bg-white/10 rounded w-1/2" />
      <div className="h-32 bg-white/10 rounded" />
    </div>
  );
}

export function TrackCardSkeleton() {
  return (
    <div className="animate-pulse flex items-center gap-3 p-3 bg-white/5 rounded-lg">
      <div className="w-12 h-12 bg-white/10 rounded" />
      <div className="flex-1 space-y-2">
        <div className="h-4 bg-white/10 rounded w-3/4" />
        <div className="h-3 bg-white/10 rounded w-1/2" />
      </div>
    </div>
  );
}

