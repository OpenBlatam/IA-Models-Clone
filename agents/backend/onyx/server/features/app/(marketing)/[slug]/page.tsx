import { notFound } from "next/navigation";
import { Metadata } from "next";
import { Mdx } from "@/components/mdx-components";

interface Page {
  slug: string;
  title: string;
  description: string;
  content: string;
}

const pages: Page[] = [
  {
    slug: "privacy",
    title: "Privacy Policy",
    description: "Our privacy policy and how we handle your data.",
    content: `
# Privacy Policy

Last updated: March 20, 2024

## Introduction

This Privacy Policy describes how we collect, use, and handle your personal information when you use our service.

## Information We Collect

We collect information that you provide directly to us, including:

- Name and contact information
- Account credentials
- Payment information
- Usage data

## How We Use Your Information

We use the information we collect to:

- Provide and maintain our service
- Process your transactions
- Send you updates and marketing communications
- Improve our service

## Data Security

We implement appropriate security measures to protect your personal information.

## Contact Us

If you have any questions about this Privacy Policy, please contact us.
    `
  },
  {
    slug: "terms",
    title: "Terms of Service",
    description: "Our terms of service and conditions of use.",
    content: `
# Terms of Service

Last updated: March 20, 2024

## Agreement to Terms

By accessing or using our service, you agree to be bound by these Terms of Service.

## Use of Service

You agree to use our service only for lawful purposes and in accordance with these Terms.

## User Accounts

You are responsible for maintaining the confidentiality of your account credentials.

## Intellectual Property

All content and materials available through our service are protected by intellectual property rights.

## Limitation of Liability

We shall not be liable for any indirect, incidental, special, consequential, or punitive damages.

## Contact Us

If you have any questions about these Terms, please contact us.
    `
  }
];

export async function generateStaticParams() {
  return pages.map((page) => ({
    slug: page.slug,
  }));
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string }>;
}): Promise<Metadata> {
  const resolvedParams = await params;
  const page = pages.find((page) => page.slug === resolvedParams.slug);
  if (!page) {
    return {};
  }

  return {
    title: page.title,
    description: page.description,
  };
}

export default async function PagePage({
  params,
}: {
  params: Promise<{
    slug: string;
  }>;
}) {
  const resolvedParams = await params;
  const page = pages.find((page) => page.slug === resolvedParams.slug);

  if (!page) {
    notFound();
  }

  return (
    <article className="container relative max-w-3xl py-6 lg:py-10">
      <div>
        <h1 className="mt-2 inline-block text-4xl font-bold leading-tight lg:text-5xl">
          {page.title}
        </h1>
        {page.description && (
          <p className="mt-4 text-xl text-muted-foreground">
            {page.description}
          </p>
        )}
      </div>
      <Mdx code={page.content} />
    </article>
  );
}
