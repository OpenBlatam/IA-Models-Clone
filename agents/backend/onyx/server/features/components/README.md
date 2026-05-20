# UI Component Library

<div align="center">

![Status](https://img.shields.io/badge/status-active-success.svg)
![Framework](https://img.shields.io/badge/framework-React--NextJS-blue.svg)
![Styling](https://img.shields.io/badge/styling-Tailwind--Radix-purple.svg)

**Comprehensive React component library for the Blatam Academy frontend, featuring high-performance visualization, interactive dashboards, and atomic UI elements.**

</div>

---

## 📋 Overview

The **Components** directory is the heart of the frontend architecture. It provides a highly modular, reusable, and themed set of UI components. From atomic elements like `ThemeToggle` to complex organisms like `ChatLayout` and interactive `Konva` canvases, this library ensures visual consistency and rapid UI development across the entire application.

## 🚀 Key Categories

| Category | Description |
|---------|-------------|
| **Core Layout** | Shell, Navigation, Mobile Nav, and Error Boundaries. |
| **Interactive** | Chat Interfaces, Text-to-Speech Panels, and Gemini Modals. |
| **Data Viz** | Chart components and real-time analytics widgets. |
| **Specialized** | OnlyOffice Editor, Konva wrappers, and Web3 components. |
| **Animation** | Parallax scroll effects and Framer Motion wrappers. |

## 📁 Structure

```
components/
├── ui/                   # Atomic Radix/Tailwind components (Button, Input, etc.)
├── layout/               # Global page structures and shell
├── dashboard/            # Specialized widgets for the admin/user dashboard
├── collections/          # Feature-specific components (BlogPost, FacebookPost)
├── realtime/             # Live collaboration and websocket-driven UI
└── shared/               # Cross-cutting UI utilities
```

## 💻 Tech Stack

- **Framework**: Next.js 14+ (App Router compatible)
- **Styling**: Tailwind CSS
- **Primitives**: Radix UI
- **Canvas**: React-Konva
- **Animation**: Framer Motion
- **Contexts**: Theme, Auth, and Collaboration providers

---

<div align="center">
  <b>Built with ❤️ by Blatam Academy</b><br>
  Part of the Onyx Server Architecture<br>
  <a href="../README.md">← Back to Main README</a>
</div>
