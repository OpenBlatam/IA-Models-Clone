import { getCurrentUser } from "@/lib/session";
import { constructMetadata } from "@/lib/utils";
import { DashboardHeader } from "@/components/dashboard/header";
import { MarketingTrivia } from "@/components/games/marketing-trivia";

export const metadata = constructMetadata({
  title: "Trivia – Learning Platform",
  description: "Pon a prueba tus conocimientos con nuestro juego de trivia.",
});

export default async function TriviaPage() {
  const user = await getCurrentUser();
  if (!user?.id) return null;

  return (
    <div className="space-y-8">
      <DashboardHeader
        heading="Trivia"
        text="Pon a prueba tus conocimientos con preguntas sobre marketing digital."
      />
      <div className="grid gap-8">
        {/* Aquí irá el contenido del juego de trivia */}
        <MarketingTrivia />
      </div>
    </div>
  );
} 