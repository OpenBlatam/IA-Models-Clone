import Link from "next/link";
import { notFound } from "next/navigation";

import { Mdx } from "@/components/content/mdx-components";
import { DocsPageHeader } from "@/components/docs/page-header";
import { Icons } from "@/components/shared/icons";
import { DashboardTableOfContents } from "@/components/shared/toc";
import { getTableOfContents } from "@/lib/toc";

import "@/styles/mdx.css";

import { Metadata } from "next";

import MaxWidthWrapper from "@/components/shared/max-width-wrapper";
import { buttonVariants } from "@/components/ui/button";
import { cn, constructMetadata } from "@/lib/utils";

export async function generateStaticParams() {
  return [
    { slug: "getting-started" },
    { slug: "nextauth" },
    { slug: "deployment" }
  ];
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string }>;
}): Promise<Metadata | undefined> {
  const resolvedParams = await params;
  
  const guides = {
    "getting-started": {
      title: "Getting Started with Next.js 15",
      description: "Learn the fundamentals of Next.js 15 and build your first application."
    },
    "nextauth": {
      title: "Authentication with NextAuth.js", 
      description: "Implement secure authentication in your Next.js application."
    },
    "deployment": {
      title: "Deployment Guide",
      description: "Deploy your Next.js application to production."
    }
  };

  const guide = guides[resolvedParams.slug as keyof typeof guides];

  if (!guide) {
    return;
  }

  const { title, description } = guide;

  return constructMetadata({
    title: `${title} – SaaS Starter`,
    description: description,
  });
}

export default async function GuidePage({
  params,
}: {
  params: Promise<{
    slug: string;
  }>;
}) {
  const resolvedParams = await params;
  
  const guides = {
    "getting-started": {
      title: "Getting Started with Next.js 15",
      description: "Learn the fundamentals of Next.js 15 and build your first application.",
      body: { raw: "# Getting Started with Next.js 15\n\nWelcome to the comprehensive guide..." }
    },
    "nextauth": {
      title: "Authentication with NextAuth.js", 
      description: "Implement secure authentication in your Next.js application.",
      body: { raw: "# Authentication with NextAuth.js\n\nLearn how to implement secure authentication..." }
    },
    "deployment": {
      title: "Deployment Guide",
      description: "Deploy your Next.js application to production.",
      body: { raw: "# Deployment Guide\n\nStep-by-step instructions for deploying your application..." }
    }
  };

  const guide = guides[resolvedParams.slug as keyof typeof guides];

  if (!guide) {
    notFound();
  }

  const toc = await getTableOfContents(guide.body.raw);
  
  return (
    <MaxWidthWrapper>
      <div className="relative py-6 lg:grid lg:grid-cols-[1fr_300px] lg:gap-10 lg:py-10 xl:gap-20">
        <div>
          <DocsPageHeader heading={guide.title} text={guide.description} />
          <Mdx content={guide.body.raw} />
          <hr className="my-4" />
          <div className="flex justify-center py-6 lg:py-10">
            <Link
              href="/guides"
              className={cn(buttonVariants({ variant: "ghost" }))}
            >
              <Icons.chevronLeft className="mr-2 size-4" />
              See all guides
            </Link>
          </div>
        </div>
        <div className="hidden text-sm lg:block">
          <div className="sticky top-16 -mt-10 max-h-[calc(var(--vh)-4rem)] overflow-y-auto pt-10">
            <DashboardTableOfContents toc={toc} />
          </div>
        </div>
      </div>
    </MaxWidthWrapper>
  );
}
