"use client";

import dynamic from "next/dynamic";
import { Suspense } from "react";
import { Skeleton } from "@/components/ui/skeleton";
import { Card } from "@/components/ui/card";

const MarketingFlappy = dynamic(() => import("@/components/games/marketing-flappy").then(mod => ({ default: mod.MarketingFlappy })), {
  ssr: false,
  loading: () => (
    <Card className="p-6">
      <Skeleton className="h-[400px] w-full" />
    </Card>
  )
});

export function FlappyGameClient() {
  return (
    <Suspense fallback={<Card className="p-6"><Skeleton className="h-[400px] w-full" /></Card>}>
      <MarketingFlappy />
    </Suspense>
  );
}
