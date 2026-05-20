"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { useNotifications } from "@/hooks/use-notifications";
import { Notifications } from "./notifications";

interface DashboardHeaderProps {
  heading: string;
  text?: string;
  children?: React.ReactNode;
}

export function DashboardHeader({
  heading,
  text,
  children,
}: DashboardHeaderProps) {
  const pathname = usePathname();
  const { notifications, markAsRead } = useNotifications();

  const navigation = [
    { name: "Dashboard", href: "/dashboard" },
    { name: "Lecciones", href: "/dashboard/lessons" },
    { name: "Ejercicios", href: "/dashboard/exercises" },
    { name: "Logros", href: "/dashboard/achievements" },
  ];

  return (
    <div className="flex items-center justify-between px-2">
      <div className="grid gap-1">
        <h1 className="font-heading text-3xl md:text-4xl">{heading}</h1>
        {text && <p className="text-lg text-muted-foreground">{text}</p>}
      </div>
      {children}
    </div>
  );
}
