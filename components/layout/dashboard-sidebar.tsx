"use client";

import { Fragment, useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { NavItem, SidebarNavItem } from "@/types";
import { Menu, PanelLeftClose, PanelRightClose, Gamepad2, Pin, PinOff } from "lucide-react";

import { siteConfig } from "@/config/site";
import { cn } from "@/lib/utils";
import { useMediaQuery } from "@/hooks/use-media-query";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import ProjectSwitcher from "@/components/dashboard/project-switcher";
import { UpgradeCard } from "@/components/dashboard/upgrade-card";
import { Icons } from "@/components/shared/icons";

interface DashboardSidebarProps {
  links: SidebarNavItem[];
}

export function DashboardSidebar({ links }: DashboardSidebarProps) {
  const path = usePathname();

  // NOTE: Use this if you want save in local storage -- Credits: Hosna Qasmei
  //
  // const [isSidebarExpanded, setIsSidebarExpanded] = useState(() => {
  //   if (typeof window !== "undefined") {
  //     const saved = window.localStorage.getItem("sidebarExpanded");
  //     return saved !== null ? JSON.parse(saved) : true;
  //   }
  //   return true;
  // });

  // useEffect(() => {
  //   if (typeof window !== "undefined") {
  //     window.localStorage.setItem(
  //       "sidebarExpanded",
  //       JSON.stringify(isSidebarExpanded),
  //     );
  //   }
  // }, [isSidebarExpanded]);

  const { isTablet } = useMediaQuery();
  const [isSidebarExpanded, setIsSidebarExpanded] = useState(!isTablet);
  const [isPinned, setIsPinned] = useState(false);
  const [typedText, setTypedText] = useState("");

  const toggleSidebar = () => {
    setIsSidebarExpanded(!isSidebarExpanded);
  };

  const togglePin = () => {
    setIsPinned(!isPinned);
  };

  useEffect(() => {
    if (!isPinned) {
      setIsSidebarExpanded(!isTablet);
    }
  }, [isTablet, isPinned]);

  useEffect(() => {
    if (isSidebarExpanded) {
      const text = "blatam";
      let index = 0;
      const interval = setInterval(() => {
        if (index < text.length) {
          setTypedText(text.slice(0, index + 1));
          index++;
        } else {
          clearInterval(interval);
        }
      }, 100);
      return () => clearInterval(interval);
    } else {
      setTypedText("");
    }
  }, [isSidebarExpanded]);

  return (
    <TooltipProvider delayDuration={0}>
      <div className="sticky top-0 h-full">
        <ScrollArea className="h-full overflow-y-auto border-r">
          <aside
            className={cn(
              isSidebarExpanded ? "w-[200px] xl:w-[220px]" : "w-[90px]",
              "hidden h-screen md:block fixed left-0 top-0 z-50",
              "transition-all duration-300 ease-in-out"
            )}
            onMouseEnter={() => !isPinned && setIsSidebarExpanded(true)}
            onMouseLeave={() => !isPinned && setIsSidebarExpanded(false)}
          >
            <div className="flex h-full max-h-screen flex-1 flex-col gap-2 bg-background">
              <div className="flex flex-col items-center justify-center h-20 p-4 lg:h-[90px]">
                <div className="flex flex-col items-center">
                  {(!isSidebarExpanded || isPinned) && (
                    <span className="mb-1">
                      <span
                        className="inline-flex items-center justify-center w-20 h-20 rounded bg-white shadow-lg cursor-pointer hover:scale-105 transition-transform"
                        style={{
                          boxShadow: "0 0 24px 8px #fff, 0 2px 12px #0002, 0 1.5px 0 #fff8",
                          border: "1.5px solid rgba(255,255,255,0.7)",
                          background: "linear-gradient(135deg, #fff 80%, #f3f3f3 100%)",
                          transition: "box-shadow 0.2s, border 0.2s"
                        }}
                      >
                        <img src="/b_logo.png" alt="Blatam Academy Assistant" className="w-16 h-16 object-contain" />
                      </span>
                    </span>
                  )}
                  {!isSidebarExpanded && !isPinned && (
                    <div className="relative group">
                      <span
                        className="text-sm mt-1 inline-block transition-all duration-300 ease-in-out group-hover:-translate-y-0.5 bg-gradient-to-br from-white/95 to-white/75 bg-clip-text text-transparent tracking-[0.2em]"
                        style={{
                          fontFamily: "-apple-system, BlinkMacSystemFont, 'SF Pro Display', 'San Francisco', sans-serif",
                          textShadow: "0 1px 2px rgba(0,0,0,0.15)",
                          fontWeight: "500",
                          fontSize: "0.8125rem",
                          letterSpacing: "0.2em",
                          WebkitFontSmoothing: "antialiased",
                          MozOsxFontSmoothing: "grayscale"
                        }}
                      >
                        blatam
                      </span>
                      <div className="absolute bottom-0 left-0 w-full h-[0.5px] bg-gradient-to-r from-transparent via-white/90 to-transparent scale-x-0 group-hover:scale-x-100 transition-transform duration-300 ease-in-out" />
                    </div>
                  )}
                </div>
                {isSidebarExpanded && !isPinned && (
                  <span
                    className="text-2xl font-extrabold tracking-wider text-white transition-all duration-200 mb-2"
                    style={{ fontFamily: "'Montserrat', Arial, sans-serif", textShadow: "0 2px 12px #00000066, 0 1.5px 0 #fff" }}
                  >
                    {typedText}
                  </span>
                )}
                {isSidebarExpanded && <ProjectSwitcher />}
              </div>

              <div className="flex h-14 items-center p-4 lg:h-[60px]">
                <Button
                  variant="ghost"
                  size="icon"
                  className="ml-auto size-9 lg:size-8"
                  onClick={toggleSidebar}
                >
                  {isSidebarExpanded ? (
                    <PanelLeftClose
                      size={18}
                      className="stroke-muted-foreground"
                    />
                  ) : (
                    <PanelRightClose
                      size={18}
                      className="stroke-muted-foreground"
                    />
                  )}
                  <span className="sr-only">Toggle Sidebar</span>
                </Button>
                {isSidebarExpanded && (
                  <Button
                    variant="ghost"
                    size="icon"
                    className="ml-2 size-9 lg:size-8"
                    onClick={togglePin}
                  >
                    {isPinned ? (
                      <Pin className="size-5" />
                    ) : (
                      <PinOff className="size-5" />
                    )}
                    <span className="sr-only">Pin Sidebar</span>
                  </Button>
                )}
              </div>

              <nav className="flex flex-1 flex-col gap-8 px-4 pt-4">
                {links.map((section) => (
                  <section
                    key={section.title}
                    className="flex flex-col gap-0.5"
                  >
                    {isSidebarExpanded ? (
                      <p className="text-xs text-muted-foreground">
                        {section.title}
                      </p>
                    ) : (
                      <div className="h-4" />
                    )}
                    {section.items.map((item) => {
                      const Icon = item.icon ? Icons[item.icon as keyof typeof Icons] : Icons.arrowRight;
                      return (
                        <Fragment key={item.title}>
                          {item.href ? (
                            <Link
                              href={item.href}
                              className={cn(
                                "group flex items-center rounded-md px-3 py-2 text-sm font-medium hover:bg-accent hover:text-accent-foreground",
                                path === item.href
                                  ? "bg-accent"
                                  : "transparent",
                                !isSidebarExpanded && "justify-center"
                              )}
                            >
                              <Icon
                                className={cn(
                                  "mr-2 h-4 w-4",
                                  !isSidebarExpanded && "mr-0"
                                )}
                              />
                              {isSidebarExpanded && <span>{item.title}</span>}
                            </Link>
                          ) : (
                            <span
                              className={cn(
                                "group flex items-center rounded-md px-3 py-2 text-sm font-medium hover:bg-accent hover:text-accent-foreground",
                                !isSidebarExpanded && "justify-center"
                              )}
                            >
                              <Icon
                                className={cn(
                                  "mr-2 h-4 w-4",
                                  !isSidebarExpanded && "mr-0"
                                )}
                              />
                              {isSidebarExpanded && <span>{item.title}</span>}
                            </span>
                          )}
                        </Fragment>
                      );
                    })}
                  </section>
                ))}
              </nav>

              <div className="mt-auto xl:p-4">
                {isSidebarExpanded ? <UpgradeCard /> : null}
              </div>
            </div>
          </aside>
        </ScrollArea>
      </div>
      <div 
        className={cn(
          "hidden md:block",
          isSidebarExpanded ? "w-[200px] xl:w-[220px]" : "w-[90px]",
          "transition-all duration-300 ease-in-out"
        )} 
      />
    </TooltipProvider>
  );
}

export function MobileSheetSidebar({ links }: DashboardSidebarProps) {
  const path = usePathname();
  const [open, setOpen] = useState(false);
  const { isSm, isMobile } = useMediaQuery();

  if (isSm || isMobile) {
    return (
      <Sheet open={open} onOpenChange={setOpen}>
        <SheetTrigger asChild>
          <Button
            variant="outline"
            size="icon"
            className="size-9 shrink-0 md:hidden"
          >
            <Menu className="size-5" />
            <span className="sr-only">Toggle navigation menu</span>
          </Button>
        </SheetTrigger>
        <SheetContent side="left" className="flex flex-col p-0">
          <ScrollArea className="h-full overflow-y-auto">
            <div className="flex h-screen flex-col">
              <nav className="flex flex-1 flex-col gap-y-8 p-6 text-lg font-medium">
                <Link
                  href="#"
                  className="flex items-center gap-2 text-lg font-semibold"
                >
                  <Icons.logo className="size-6" />
                  <span className="font-urban text-xl font-bold">
                    {siteConfig.name}
                  </span>
                </Link>

                <ProjectSwitcher large />

                <div className="flex flex-col gap-2">
                  <Link
                    href="/dashboard"
                    className={cn(
                      "flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium hover:bg-accent hover:text-accent-foreground",
                      path === "/dashboard"
                        ? "bg-accent text-accent-foreground"
                        : "text-muted-foreground"
                    )}
                  >
                    <Icons.dashboard className="size-4" />
                    <span>Dashboard</span>
                  </Link>

                  <Link
                    href="/dashboard/lessons"
                    className={cn(
                      "flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium hover:bg-accent hover:text-accent-foreground",
                      path === "/dashboard/lessons"
                        ? "bg-accent text-accent-foreground"
                        : "text-muted-foreground"
                    )}
                  >
                    <Icons.bookOpen className="size-4" />
                    <span>Lecciones</span>
                  </Link>

                  <Link
                    href="/dashboard/academy"
                    className={cn(
                      "flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium hover:bg-accent hover:text-accent-foreground",
                      path === "/dashboard/academy"
                        ? "bg-accent text-accent-foreground"
                        : "text-muted-foreground"
                    )}
                  >
                    <Icons.target className="size-4" />
                    <span>Academy</span>
                  </Link>

                  <Link
                    href="/dashboard/exercises"
                    className={cn(
                      "flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium hover:bg-accent hover:text-accent-foreground",
                      path === "/dashboard/exercises"
                        ? "bg-accent text-accent-foreground"
                        : "text-muted-foreground"
                    )}
                  >
                    <Icons.lineChart className="size-4" />
                    <span>Ejercicios</span>
                  </Link>

                  <Link
                    href="/dashboard/games"
                    className={cn(
                      "flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium hover:bg-accent hover:text-accent-foreground",
                      path === "/dashboard/games"
                        ? "bg-accent text-accent-foreground"
                        : "text-muted-foreground"
                    )}
                  >
                    <Gamepad2 className="size-4" />
                    <span>Juegos</span>
                  </Link>

                  <Link
                    href="/dashboard/achievements"
                    className={cn(
                      "flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium hover:bg-accent hover:text-accent-foreground",
                      path === "/dashboard/achievements"
                        ? "bg-accent text-accent-foreground"
                        : "text-muted-foreground"
                    )}
                  >
                    <Icons.trophy className="size-4" />
                    <span>Logros</span>
                  </Link>
                </div>
              </nav>

              <div className="mt-auto p-6">
                <UpgradeCard />
              </div>
            </div>
          </ScrollArea>
        </SheetContent>
      </Sheet>
    );
  }

  return (
    <div className="flex size-9 animate-pulse rounded-lg bg-muted md:hidden" />
  );
}
