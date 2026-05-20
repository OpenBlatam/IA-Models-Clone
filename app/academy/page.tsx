"use client";

import { Suspense } from "react";
import dynamic from "next/dynamic";

const AcademyClient = dynamic(() => import("./academy-client"), {
  ssr: false,
  loading: () => <div className="container py-8">
    <div className="space-y-4">
      <h1 className="text-4xl font-bold tracking-tight">Academias</h1>
      <p className="text-xl text-muted-foreground">
        Cargando academias...
      </p>
    </div>
  </div>
});

export default function AcademyPage() {
  return (
    <Suspense fallback={<div className="container py-8">
      <div className="space-y-4">
        <h1 className="text-4xl font-bold tracking-tight">Academias</h1>
        <p className="text-xl text-muted-foreground">
          Cargando academias...
        </p>
      </div>
    </div>}>
      <AcademyClient />
    </Suspense>
  );
}          