import { getCurrentUser } from "@/lib/session";
import { constructMetadata } from "@/lib/utils";
import { DashboardHeader } from "@/components/dashboard/header";
import { MarketingFlappy } from "@/components/games/marketing-flappy";

export const metadata = constructMetadata({
  title: "Flappy – Learning Platform",
  description: "Juego de destreza con temática de marketing.",
});

export default async function FlappyPage() {
  const user = await getCurrentUser();
  if (!user?.id) return null;

  return (
    <div className="space-y-8">
      <DashboardHeader
        heading="Flappy"
        text="Juego de destreza con temática de marketing."
      />
      <div className="grid gap-8">
        {/* Aquí irá el contenido del juego Flappy */}
        <MarketingFlappy />
      </div>
    </div>
  );
} 