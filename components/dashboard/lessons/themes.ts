import { Themes } from './types';

export const themes: Themes = {
  light: {
    background: "from-zinc-50 via-white to-zinc-50",
    card: "bg-white/90",
    border: "border-zinc-200/80",
    text: {
      primary: "text-zinc-900",
      secondary: "text-zinc-600",
      accent: "from-amber-600 via-rose-500 to-amber-600"
    },
    button: {
      primary: "from-amber-500 via-rose-500 to-amber-500 hover:from-amber-600 hover:via-rose-600 hover:to-amber-600",
      secondary: "bg-white/90 border-zinc-200 hover:bg-zinc-50/90"
    },
    accent: {
      primary: "from-amber-400 via-rose-400 to-amber-400",
      secondary: "from-amber-50/80 via-rose-50/80 to-amber-50/80"
    },
    shadow: "shadow-[0_8px_30px_rgb(0,0,0,0.04)]",
    hoverShadow: "hover:shadow-[0_8px_30px_rgb(0,0,0,0.08)]"
  },
  dark: {
    background: "from-zinc-900 via-zinc-950 to-zinc-900",
    card: "bg-zinc-900/90",
    border: "border-zinc-800/80",
    text: {
      primary: "text-zinc-50",
      secondary: "text-zinc-400",
      accent: "from-amber-400 via-rose-400 to-amber-400"
    },
    button: {
      primary: "from-amber-500 via-rose-500 to-amber-500 hover:from-amber-600 hover:via-rose-600 hover:to-amber-600",
      secondary: "bg-zinc-800/90 border-zinc-700 hover:bg-zinc-700/90"
    },
    accent: {
      primary: "from-amber-500 via-rose-500 to-amber-500",
      secondary: "from-amber-900/80 via-rose-900/80 to-amber-900/80"
    },
    shadow: "shadow-[0_8px_30px_rgb(0,0,0,0.2)]",
    hoverShadow: "hover:shadow-[0_8px_30px_rgb(0,0,0,0.3)]"
  }
}; 