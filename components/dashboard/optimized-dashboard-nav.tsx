"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { Home, BookOpen, Target, LineChart, Trophy, Gamepad2, Video, Megaphone, Users } from "lucide-react";
import { motion } from "framer-motion";
import { memo, useCallback } from "react";
import { OptimizedLink } from "@/components/ui/optimized-link";

const navItems = [
  {
    title: "Dashboard",
    href: "/dashboard",
    icon: Home,
  },
  {
    title: "Academy",
    href: "/dashboard/academy",
    icon: BookOpen,
  },
  {
    title: "Videos",
    href: "/dashboard/videos",
    icon: Video,
  },
  {
    title: "Lessons",
    href: "/dashboard/lessons",
    icon: Target,
  },
  {
    title: "Games",
    href: "/dashboard/games",
    icon: Gamepad2,
  },
  {
    title: "Progress",
    href: "/dashboard/progress",
    icon: LineChart,
  },
  {
    title: "Settings",
    href: "/dashboard/settings",
    icon: Trophy,
  },
];

export const OptimizedDashboardNav = memo(function OptimizedDashboardNav() {
  const pathname = usePathname();

  const isActive = useCallback((href: string) => {
    return pathname === href || (pathname && pathname.startsWith(href + "/"));
  }, [pathname]);

  return (
    <nav className="flex items-center space-x-4 lg:space-x-6">
      {navItems.map((item, index) => {
        const Icon = item.icon;
        const active = isActive(item.href);
        
        return (
          <OptimizedLink
            key={item.href}
            href={item.href}
            prefetch={index < 4}
            className={cn(
              "flex items-center text-sm font-medium transition-colors hover:text-primary",
              active ? "text-primary" : "text-muted-foreground"
            )}
          >
            <motion.div
              className="flex items-center space-x-2"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              transition={{ type: "spring", stiffness: 400, damping: 17 }}
            >
              <Icon className="h-4 w-4" />
              <span>{item.title}</span>
            </motion.div>
          </OptimizedLink>
        );
      })}
    </nav>
  );
});
