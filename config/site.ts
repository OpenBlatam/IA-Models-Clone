import { SidebarNavItem, SiteConfig } from "types";
import { env } from "@/env.mjs";

const site_url = env.NEXT_PUBLIC_APP_URL;

export const siteConfig: SiteConfig = {
  name: "Blatam Academy",
  description:
    "Empowering the next generation of tech leaders through innovative education and practical experience.",
  url: site_url,
  ogImage: `${site_url}/_static/og.jpg`,
  links: {
    twitter: "https://twitter.com/blatamacademy",
    github: "https://github.com/blatamacademy",
  },
  mailSupport: "support@blatamacademy.com",
};

export const footerLinks: SidebarNavItem[] = [
  {
    title: "Academy",
    items: [
      { title: "About Us", href: "/about" },
      { title: "Our Mission", href: "/mission" },
      { title: "Terms", href: "/terms" },
      { title: "Privacy", href: "/privacy" },
    ],
  },
  {
    title: "Programs",
    items: [
      { title: "Courses", href: "/courses" },
      { title: "Workshops", href: "/workshops" },
      { title: "Events", href: "/events" },
      { title: "Career Support", href: "/career" },
    ],
  },
  {
    title: "Resources",
    items: [
      { title: "Blog", href: "/blog" },
      { title: "Student Portal", href: "/portal" },
      { title: "FAQ", href: "/faq" },
      { title: "Contact", href: "/contact" },
    ],
  },
];
