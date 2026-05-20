import { getCurrentUser } from "@/lib/session";
import { constructMetadata } from "@/lib/utils";
import { DashboardHeader } from "@/components/dashboard/header";

export const metadata = constructMetadata({
  title: "Match – Learning Platform",
  description: "Empareja conceptos relacionados con marketing digital.",
});

export default async function MatchPage() {
  const user = await getCurrentUser();
  if (!user?.id) return null;

  return (
    <div className="space-y-8">
      <DashboardHeader
        heading="Match"
        text="Empareja conceptos relacionados con marketing digital."
      />
      <div className="grid gap-8">
        <div className="p-8 border border-gray-200 rounded-lg bg-white">
          <p className="text-gray-500 text-center">Juego de emparejamiento en desarrollo</p>
        </div>
      </div>
    </div>
  );
}   