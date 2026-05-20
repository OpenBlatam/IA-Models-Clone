"use client";
import { useState } from "react";
import { DashboardNav } from "@/components/dashboard/dashboard-nav";
import MobileBottomNavWrapper from "@/components/MobileBottomNavWrapper";
import { Player } from "@lottiefiles/react-lottie-player";
import { DayPicker } from "react-day-picker";
import "react-day-picker/dist/style.css";
import Link from "next/link";
import Image from "next/image";
import { signIn } from "next-auth/react";

// Add Google Fonts in _document.js or <Head> in your app for 'EB Garamond' and 'Inter'
// <link href="https://fonts.googleapis.com/css2?family=EB+Garamond:wght@700&family=Inter:wght@400;600&display=swap" rel="stylesheet" />

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const [theme, setTheme] = useState("light");
  const [selected, setSelected] = useState();
  const toggleTheme = () => setTheme(theme === "light" ? "dark" : "light");

  const palette = theme === "light"
    ? {
        bg: "#F8FAFF",
        card: "#FFFFFF",
        header: "#FFFFFF",
        border: "#232346",
        text: "#232346",
        accent: "#00E6A0",
        primary: "#6C63FF",
        blue: "#00CFFF",
        pink: "#FF5A7A",
        btn: "#232346",
        btnHover: "#6C63FF",
      }
    : {
        bg: "#181830",
        card: "#232346",
        header: "#232346",
        border: "#E0E0FF",
        text: "#E0E0FF",
        accent: "#00E6A0",
        primary: "#6C63FF",
        blue: "#00CFFF",
        pink: "#FF5A7A",
        btn: "#E0E0FF",
        btnHover: "#FF5A7A",
      };

  return (
    <div
      className="min-h-screen flex flex-col"
      style={{
        background: palette.bg,
        color: palette.text,
        fontFamily: "'Inter', 'EB Garamond', serif",
      }}
    >
      <header
        className="sticky top-0 z-40 border-b"
        style={{
          background: palette.header,
          borderColor: palette.border,
        }}
      >
        <div className="container flex h-20 items-center justify-between py-4">
          <DashboardNav />
          <button
            onClick={toggleTheme}
            className="ml-4 px-6 py-2 rounded-full font-bold shadow-md transition-colors duration-200 border-2 border-black tracking-widest uppercase"
            style={{
              background: palette.btn,
              color: theme === "light" ? "#fff" : "#232346",
              letterSpacing: "0.2em",
              fontFamily: "'EB Garamond', serif",
              fontSize: "1.1rem",
            }}
          >
            {theme === "light" ? "🌙 " : "☀️ "}
          </button>
        </div>
      </header>
      <main className="flex-1 p-8 pt-20">
        <div className="max-w-7xl mx-auto">
          {children}
        </div>
      </main>
      <MobileBottomNavWrapper />
    </div>
  );
} 