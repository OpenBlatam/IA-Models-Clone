"use client";

import React, { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";

interface BundleStats {
  totalSize: number;
  gzippedSize: number;
  chunks: Array<{
    name: string;
    size: number;
    modules: number;
  }>;
}

export function BundleAnalyzer() {
  const [stats, setStats] = useState<BundleStats | null>(null);

  useEffect(() => {
    if (process.env.NODE_ENV === 'development') {
      const mockStats: BundleStats = {
        totalSize: 2.4 * 1024 * 1024,
        gzippedSize: 0.8 * 1024 * 1024,
        chunks: [
          { name: 'main', size: 1.2 * 1024 * 1024, modules: 45 },
          { name: 'vendors', size: 0.8 * 1024 * 1024, modules: 120 },
          { name: 'animations', size: 0.2 * 1024 * 1024, modules: 8 },
          { name: 'games', size: 0.2 * 1024 * 1024, modules: 12 },
        ]
      };
      setStats(mockStats);
    }
  }, []);

  if (!stats || process.env.NODE_ENV !== 'development') {
    return null;
  }

  const formatSize = (bytes: number) => {
    const mb = bytes / (1024 * 1024);
    return `${mb.toFixed(2)} MB`;
  };

  return (
    <Card className="p-4 fixed bottom-4 right-4 w-80 z-50 bg-background/95 backdrop-blur">
      <h3 className="font-semibold mb-2">Bundle Analysis</h3>
      <div className="space-y-2 text-sm">
        <div>Total: {formatSize(stats.totalSize)}</div>
        <div>Gzipped: {formatSize(stats.gzippedSize)}</div>
        <div className="space-y-1">
          {stats.chunks.map((chunk) => (
            <div key={chunk.name} className="flex justify-between">
              <span>{chunk.name}</span>
              <span>{formatSize(chunk.size)}</span>
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
}
