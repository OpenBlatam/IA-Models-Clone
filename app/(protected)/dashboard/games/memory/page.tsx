import { getCurrentUser } from "@/lib/session";
import { constructMetadata } from "@/lib/utils";
import { DashboardHeader } from "@/components/dashboard/header";
import { MarketingMemory } from "@/components/games/marketing-memory";

export const metadata = constructMetadata({
  title: "Memory – Learning Platform",
  description: "Mejora tu memoria con nuestro juego de parejas.",
});

export default async function MemoryPage() {
  const user = await getCurrentUser();
  if (!user?.id) return null;

  return (
    <div className="space-y-8">
      <DashboardHeader
        heading="Memory"
        text="Encuentra las parejas de conceptos de marketing digital."
      />
      <div className="grid gap-8">
        {/* Aquí irá el contenido del juego de memoria */}
        <MarketingMemory />
      </div>
    </div>
  );
} 