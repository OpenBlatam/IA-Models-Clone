import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

const navItems = [
  { href: "/dashboard", label: "Dashboard" },
  { href: "/dashboard/lecciones", label: "Lecciones" },
  { href: "/dashboard/videos", label: "Videos" },
  { href: "/dashboard/academy", label: "Academy" },
  { href: "/dashboard/ejercicios", label: "Ejercicios" },
  { href: "/dashboard/juegos", label: "Juegos" },
  { href: "/dashboard/colab-ia", label: "Colab IA" },
  { href: "/dashboard/ads-ia", label: "ADS IA" },
  { href: "/dashboard/mkt-ia", label: "MKT IA" },
];

export function DashboardNav() {
  const pathname = usePathname();

  return (
    <nav className="flex items-center space-x-6">
      {navItems.map((item) => (
        <Link
          key={item.href}
          href={item.href}
          className={cn(
            "text-sm font-medium transition-colors hover:text-[#FFD700]",
            pathname === item.href
              ? "text-[#FFD700] border-b-2 border-[#FFD700]"
              : "text-gray-600"
          )}
        >
          {item.label}
        </Link>
      ))}
    </nav>
  );
} 