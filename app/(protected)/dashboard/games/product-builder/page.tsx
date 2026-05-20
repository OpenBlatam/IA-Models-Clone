import { getCurrentUser } from "@/lib/session";
import { constructMetadata } from "@/lib/utils";
import { DashboardHeader } from "@/components/dashboard/header";
import { ProductBuilder } from "@/components/games/product-builder";

export const metadata = constructMetadata({
  title: "Product Builder – Learning Platform",
  description: "Crea y personaliza productos virtuales.",
});

export default async function ProductBuilderPage() {
  const user = await getCurrentUser();
  if (!user?.id) return null;

  return (
    <div className="space-y-8">
      <DashboardHeader
        heading="Product Builder"
        text="Crea y personaliza productos virtuales."
      />
      <div className="grid gap-8">
        {/* Aquí irá el contenido del constructor de productos */}
        <ProductBuilder />
      </div>
    </div>
  );
} 