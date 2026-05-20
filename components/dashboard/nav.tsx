"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import {
  BookOpen,
  GraduationCap,
  Dumbbell,
  Gamepad2,
  Trophy,
  FileText,
  Video,
} from "lucide-react";

const routes = [
  {
    label: "Lecciones",
    icon: BookOpen,
    href: "/dashboard/lessons",
    color: "text-violet-500",
  },
  {
    label: "Videos",
    icon: Video,
    href: "/dashboard/videos",
    color: "text-blue-500",
  },
  {
    label: "Academy",
    icon: GraduationCap,
    href: "/dashboard/academy",
    color: "text-pink-700",
  },
  {
    label: "Ejercicios",
    icon: Dumbbell,
    href: "/dashboard/exercises",
    color: "text-orange-700",
  },
  {
    label: "Juegos",
    icon: Gamepad2,
    href: "/dashboard/games",
    color: "text-emerald-500",
  },
  {
    label: "Logros",
    icon: Trophy,
    href: "/dashboard/achievements",
    color: "text-yellow-500",
  },
  {
    label: "Apuntes",
    icon: FileText,
    href: "/dashboard/notes",
    color: "text-[#00F5A0]",
  },
];

export function DashboardNav() {
  const pathname = usePathname();

  return (
    <div className="space-y-4 py-4 flex flex-col h-full bg-[#0A0A0A] text-white">
      <div className="px-3 py-2 flex-1">
        <div className="space-y-1">
          {routes.map((route) => (
            <Link
              key={route.href}
              href={route.href}
              className={cn(
                "text-sm group flex p-3 w-full justify-start font-medium cursor-pointer hover:text-white hover:bg-white/10 rounded-lg transition",
                pathname === route.href ? "text-white bg-white/10" : "text-zinc-400",
              )}
            >
              <div className="flex items-center flex-1">
                <route.icon className={cn("h-5 w-5 mr-3", route.color)} />
                {route.label}
              </div>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
} 