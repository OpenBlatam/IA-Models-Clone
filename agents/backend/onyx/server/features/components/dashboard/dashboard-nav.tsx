'use client';

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { Home, BookOpen, Target, LineChart, Trophy, Gamepad2, Video, Megaphone, Users } from "lucide-react";
import { motion } from "framer-motion";
import { memo } from "react";

const navItems = [
  {
    title: "MKT IA",
    href: "/dashboard/mkt-ia",
    icon: Megaphone,
  },
  {
    title: "Dashboard",
    href: "/dashboard",
    icon: Home,
  },
  {
    title: "Lecciones",
    href: "/dashboard/lessons",
    icon: BookOpen,
  },
  {
    title: "Videos",
    href: "/dashboard/videos",
    icon: Video,
  },
  {
    title: "Academy",
    href: "/dashboard/academy",
    icon: Target,
  },
  {
    title: "Ejercicios",
    href: "/dashboard/exercises",
    icon: LineChart,
  },
  {
    title: "Juegos",
    href: "/dashboard/games",
    icon: Gamepad2,
  },
  {
    title: "Colab IA",
    href: "/dashboard/colaboracion-ia",
    icon: Users,
  },
  {
    title: "ADS IA",
    href: "/dashboard/ads-ia",
    icon: Megaphone,
  },
  {
    title: "Logros",
    href: "/dashboard/achievements",
    icon: Trophy,
  },
] as const;

const NavItem = memo(function NavItem({ 
  item, 
  isActive 
}: { 
  item: typeof navItems[number]; 
  isActive: boolean;
}) {
  const Icon = item.icon;
  
  return (
    <Link
      href={item.href}
      className={cn(
        "group relative flex items-center text-sm font-medium transition-all duration-300",
        isActive 
          ? "text-primary" 
          : "text-muted-foreground hover:text-primary"
      )}
    >
      <motion.div
        className="flex items-center gap-2 px-3 py-2"
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        layout
      >
        <Icon className={cn(
          "h-4 w-4 transition-colors duration-300",
          isActive ? "text-primary" : "text-muted-foreground group-hover:text-primary"
        )} />
        <span className="tracking-wide">{item.title}</span>
        {isActive && (
          <motion.div
            className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary"
            layoutId="activeNav"
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
          />
        )}
      </motion.div>
    </Link>
  );
});

export const DashboardNav = memo(function DashboardNav() {
  const pathname = usePathname();

  return (
    <>
      {/* Mobile: show logo/title only */}
      <div className="flex items-center sm:hidden">
        <span className="text-xl font-bold tracking-wider text-primary">SaaS</span>
      </div>
      {/* Desktop: show nav */}
      <nav className="hidden sm:flex items-center space-x-1">
        {navItems.map((item) => (
          <NavItem
            key={item.href}
            item={item}
            isActive={pathname === item.href}
          />
        ))}
      </nav>
    </>
  );
});     