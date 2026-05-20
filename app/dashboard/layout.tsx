"use client";

import { DashboardNav } from "@/components/dashboard/dashboard-nav";
import dynamic from 'next/dynamic';
import React from "react";
import FloatingStatusWidget from "@/components/FloatingStatusWidget";
import { ThemeToggle } from "@/components/theme-toggle";
import { UserNav } from "@/components/user-nav";
import { EB_Garamond } from "next/font/google";
import { usePathname } from "next/navigation";

const ebGaramond = EB_Garamond({ subsets: ['latin'] });

const InteractiveTTSPanel = dynamic(() => import("@/agents/InteractiveTTSPanel"), {
  ssr: false
});

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();

  React.useEffect(() => {
    const script = document.createElement('script');
    script.src = "https://elevenlabs.io/convai-widget/index.js";
    script.async = true;
    script.type = "text/javascript";
    document.body.appendChild(script);
    return () => {
      document.body.removeChild(script);
    };
  }, []);

  return (
    <div className={`relative min-h-screen bg-background ${ebGaramond.className}`}>
      <header className="sticky top-0 z-40 border-b border-border/40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container flex h-16 items-center justify-between py-4">
          <div className="flex items-center gap-6">
            <DashboardNav />
          </div>
          <div className="flex items-center gap-4">
            <ThemeToggle />
            <UserNav />
          </div>
        </div>
      </header>
      <div className="container py-6">
        {/* Only show TTS widgets if not on background-remover */}
        {pathname !== "/dashboard/mkt-ia/background-remover" && (
          <>
            <InteractiveTTSPanel />
            <FloatingStatusWidget
              message="Llamando a IA"
              color="purple"
              visible={true}
              onCancel={() => {}}
            />
          </>
        )}
        <main className="mt-6">
          {children}
        </main>
      </div>
      {/* ElevenLabs Convai Widget */}
      <div 
        dangerouslySetInnerHTML={{
          __html: '<elevenlabs-convai agent-id="agent_01jw2hqjcmf5js8yr208bavbaq"></elevenlabs-convai>'
        }}
      />
    </div>
  );
}   