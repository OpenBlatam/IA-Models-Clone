import { Metadata } from "next";

export const metadata: Metadata = {
  title: "Autenticación",
  description: "Páginas de autenticación para la plataforma",
};

export default function AuthLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen bg-background">
      {children}
    </div>
  );
}
