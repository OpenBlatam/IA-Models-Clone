"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { useUser } from "@/lib/hooks/use-user";
import { UserNav } from "@/components/user-nav";
import { Menu } from "lucide-react";
import { useState } from "react";

export function Nav() {
  const pathname = usePathname();
  const { user, isLoading } = useUser();
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const isActive = (path: string) => pathname === path;

  const navItems = [
    { href: "/", label: "Inicio" },
    { href: "/academy", label: "Academia" },
    { href: "/pricing", label: "Precios" },
  ];

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-14 items-center">
        {/* Desktop Navigation */}
        <div className="hidden md:flex flex-1 items-center justify-between space-x-2 md:justify-end">
          <nav className="flex items-center space-x-6">
            {navItems.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "text-sm font-medium transition-colors hover:text-primary",
                  isActive(item.href)
                    ? "text-foreground"
                    : "text-foreground/60"
                )}
              >
                {item.label}
              </Link>
            ))}
          </nav>
          <div className="flex items-center space-x-4">
            {isLoading ? (
              <div className="h-8 w-8 animate-pulse rounded-full bg-gray-200 dark:bg-gray-800" />
            ) : user ? (
              <UserNav />
            ) : (
              <Button asChild>
                <Link href="/login">Iniciar Sesión</Link>
              </Button>
            )}
          </div>
        </div>

        {/* Mobile Navigation */}
        <div className="flex md:hidden flex-1 items-center justify-end">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setIsMenuOpen(!isMenuOpen)}
          >
            <Menu className="h-5 w-5" />
          </Button>
        </div>
      </div>

      {/* Mobile Menu */}
      {isMenuOpen && (
        <div className="md:hidden">
          <nav className="container flex flex-col space-y-4 py-4">
            {navItems.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "text-sm font-medium transition-colors hover:text-primary",
                  isActive(item.href)
                    ? "text-foreground"
                    : "text-foreground/60"
                )}
                onClick={() => setIsMenuOpen(false)}
              >
                {item.label}
              </Link>
            ))}
            {!isLoading && !user && (
              <Button asChild className="w-full">
                <Link href="/login">Iniciar Sesión</Link>
              </Button>
            )}
          </nav>
        </div>
      )}
    </header>
  );
}  