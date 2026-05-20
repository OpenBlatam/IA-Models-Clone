"use client"

import * as React from "react"
import { useTheme } from "next-themes"
import { useEffect, useState } from "react"

import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Icons } from "@/components/shared/icons"
import { Switch } from "@/components/ui/switch"

export function ModeToggle() {
  const { theme, setTheme } = useTheme();
  const isDark = theme === "dark";
  const [highContrast, setHighContrast] = useState(
    typeof window !== "undefined" ? localStorage.getItem("highContrast") === "true" : false
  );

  useEffect(() => {
    if (highContrast) {
      document.body.classList.add("high-contrast");
      localStorage.setItem("highContrast", "true");
    } else {
      document.body.classList.remove("high-contrast");
      localStorage.setItem("highContrast", "false");
    }
  }, [highContrast]);

  return (
    <div className="flex items-center gap-6">
      <span className="text-2xl font-normal">
        {highContrast ? "Disable high contrast" : "Enable high contrast"}
      </span>
      <Switch
        checked={highContrast}
        onCheckedChange={setHighContrast}
        aria-label={highContrast ? "Disable high contrast" : "Enable high contrast"}
      />
    </div>
  );
}
