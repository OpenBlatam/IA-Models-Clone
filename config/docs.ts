import { DocsConfig } from "types";

export const docsConfig: DocsConfig = {
  mainNav: [
    {
      title: "Documentation",
      href: "/docs",
    },
    {
      title: "Guides",
      href: "/guides",
    },
  ],
  sidebarNav: [
    {
      title: "Marketing",
      items: [
        {
          title: "Introducción",
          href: "/docs",
        },
        {
          title: "¿Que se necesita para empezar?",
          href: "/docs/installation",
        },
      ],
    },
    {
      title: "Configuración",
      items: [
        {
          title: "Conceptos Básicos",
          href: "/docs/configuration/authentification",
        },
        {
          title: "Herramientas Esenciales del Marketing Digital",
          href: "/docs/configuration/blog",
        },
        {
          title: "Habilidades de Comunicación en Marketing Digital",
          href: "/docs/configuration/components",
        },
        {
          title: "Automatización de Marketing",
          href: "/docs/configuration/config-files",
        },
        {
          title: "Database",
          href: "/docs/configuration/database",
        },
        {
          title: "Email",
          href: "/docs/configuration/email",
        },
        {
          title: "Layouts",
          href: "/docs/configuration/layouts",
        },
        {
          title: "Markdown files",
          href: "/docs/configuration/markdown-files",
        },
        {
          title: "Subscriptions",
          href: "/docs/configuration/subscriptions",
        },
      ],
    },
  ],
};
