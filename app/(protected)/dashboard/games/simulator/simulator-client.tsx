"use client";

import dynamic from "next/dynamic";
import { Suspense } from "react";
import { Skeleton } from "@/components/ui/skeleton";
import { Card } from "@/components/ui/card";

const MarketingSimulator = dynamic(() => import("@/components/games/marketing-simulator").then(mod => ({ default: mod.MarketingSimulator })), {
  ssr: false,
  loading: () => (
    <Card className="p-6">
      <div className="grid grid-cols-2 gap-4">
        <Skeleton className="h-[300px]" />
        <Skeleton className="h-[300px]" />
      </div>
    </Card>
  )
});

export function SimulatorGameClient() {
  return (
    <Suspense fallback={<Card className="p-6"><div className="grid grid-cols-2 gap-4"><Skeleton className="h-[300px]" /><Skeleton className="h-[300px]" /></div></Card>}>
      <MarketingSimulator />
    </Suspense>
  );
}
