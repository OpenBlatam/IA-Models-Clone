import { getCurrentUser } from "@/lib/session";
import { constructMetadata } from "@/lib/utils";
import { DashboardHeader } from "@/components/dashboard/header";
import { MarketingSimulator } from "@/components/games/marketing-simulator";

export const metadata = constructMetadata({
  title: "Simulator – Learning Platform",
  description: "Simula estrategias de marketing digital.",
});

export default async function SimulatorPage() {
  const user = await getCurrentUser();
  if (!user?.id) return null;

  return (
    <div className="space-y-8">
      <DashboardHeader
        heading="Simulator"
        text="Simula estrategias de marketing digital."
      />
      <div className="grid gap-8">
        {/* Aquí irá el contenido del simulador */}
        <MarketingSimulator />
      </div>
    </div>
  );
} 