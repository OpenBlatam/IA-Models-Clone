import { UserRole } from "@prisma/client";

import { SidebarNavItem } from "types";

export const sidebarLinks: SidebarNavItem[] = [
  {
    title: "MENU",
    items: [
      {
        href: "/admin",
        icon: "laptop",
        title: "Admin Panel",
        authorizeOnly: UserRole.ADMIN,
      },
      { href: "/dashboard", icon: "dashboard", title: "Dashboard" },
      {
        href: "/dashboard/billing",
        icon: "billing",
        title: "Billing",
        authorizeOnly: UserRole.USER,
      },
      { href: "/dashboard/charts", icon: "lineChart", title: "Charts" },
      {
        href: "/admin/orders",
        icon: "package",
        title: "Orders",
        badge: 2,
        authorizeOnly: UserRole.ADMIN,
      },
      {
        href: "/dashboard/calendar",
        icon: "calendar",
        title: "Calendario",
        authorizeOnly: UserRole.USER
      },
      {
        href: "/dashboard/Blatam IA",
        icon: "calendar",
        title: "Blatam IA",
        authorizeOnly: UserRole.USER
      },
      {
        href: "/dashboard/exercises",
        icon: "lineChart",
        title: "Ejercicios",
        authorizeOnly: UserRole.USER
      },
      {
        href: "/dashboard/games",
        icon: "target",
        title: "Juegos",
        authorizeOnly: UserRole.USER
      },
      {
        href: "/dashboard/colaboracion-ia",
        icon: "user",
        title: "Colaboración IA",
        authorizeOnly: UserRole.USER
      },
      {
        href: "/dashboard/ads-ia",
        icon: "calendar",
        title: "ADS IA",
        authorizeOnly: UserRole.USER
      },
      {
        href: "/dashboard/achievements",
        icon: "trophy",
        title: "Logros",
        authorizeOnly: UserRole.USER
      },
      {
        href: "#/dashboard/posts",
        icon: "post",
        title: "User Posts",
        authorizeOnly: UserRole.USER,
        disabled: true,
      },
    ],
  },
  {
    title: "OPTIONS",
    items: [
      { href: "/dashboard/settings", icon: "settings", title: "Settings" },
      { href: "/", icon: "home", title: "Homepage" },
      { href: "/docs", icon: "bookOpen", title: "Documentation" },
      {
        href: "#",
        icon: "messages",
        title: "Support",
        authorizeOnly: UserRole.USER,
        disabled: true,
      },
    ],
  },
]; 