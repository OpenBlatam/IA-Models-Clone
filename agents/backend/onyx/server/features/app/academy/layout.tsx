import { Metadata } from "next";

export const metadata: Metadata = {
  title: "Academia | Next SaaS Stripe Starter",
  description: "Aprende sobre IA, ChatGPT y Midjourney con nuestros cursos especializados.",
};

export default function AcademyLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {children}
    </div>
  );
} 