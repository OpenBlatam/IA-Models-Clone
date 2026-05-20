import { ProgressDisplay } from "@/components/game/progress-display";

export default function ProgressPage() {
  return (
    <div className="container mx-auto py-8">
      <h1 className="mb-8 text-3xl font-bold">Your Progress</h1>
      <ProgressDisplay />
    </div>
  );
} 