import { notFound } from "next/navigation";

import { Mdx } from "@/components/content/mdx-components";

import "@/styles/mdx.css";

import { Metadata } from "next";
import Link from "next/link";

import { BLOG_CATEGORIES } from "@/config/blog";
import { getTableOfContents } from "@/lib/toc";
import {
  cn,
  constructMetadata,
  formatDate,
  getBlurDataURL,
  placeholderBlurhash,
} from "@/lib/utils";
import { buttonVariants } from "@/components/ui/button";
import Author from "@/components/content/author";
import BlurImage from "@/components/shared/blur-image";
import MaxWidthWrapper from "@/components/shared/max-width-wrapper";
import { DashboardTableOfContents } from "@/components/shared/toc";

export async function generateStaticParams() {
  return [
    { slug: "server-client-components" },
    { slug: "dynamic-routing-static-regeneration" },
    { slug: "preview-mode-headless-cms" }
  ];
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string }>;
}): Promise<Metadata | undefined> {
  const resolvedParams = await params;
  
  const posts = {
    "server-client-components": {
      title: "Understanding Server and Client Components",
      description: "Learn the differences between server and client components in Next.js 13+",
      image: "/blog/server-client-components.jpg"
    },
    "dynamic-routing-static-regeneration": {
      title: "Dynamic Routing and Static Regeneration",
      description: "Master dynamic routing and ISR in Next.js applications",
      image: "/blog/dynamic-routing.jpg"
    },
    "preview-mode-headless-cms": {
      title: "Preview Mode for Headless CMS",
      description: "Implement preview mode for your headless CMS content",
      image: "/blog/preview-mode.jpg"
    }
  };

  const post = posts[resolvedParams.slug as keyof typeof posts];

  if (!post) {
    return;
  }

  const { title, description, image } = post;

  return constructMetadata({
    title: `${title} – SaaS Starter`,
    description: description,
    image,
  });
}

export default async function PostPage({
  params,
}: {
  params: Promise<{
    slug: string;
  }>;
}) {
  const resolvedParams = await params;
  
  const posts = {
    "server-client-components": {
      _id: "1",
      slug: "/blog/server-client-components",
      title: "Understanding Server and Client Components",
      description: "Learn the differences between server and client components in Next.js 13+",
      image: "/blog/server-client-components.jpg",
      date: "2024-03-20",
      authors: ["Admin"],
      categories: ["nextjs"],
      body: { raw: "# Understanding Server and Client Components\n\nNext.js 13 introduced a new paradigm..." },
      images: []
    },
    "dynamic-routing-static-regeneration": {
      _id: "2", 
      slug: "/blog/dynamic-routing-static-regeneration",
      title: "Dynamic Routing and Static Regeneration",
      description: "Master dynamic routing and ISR in Next.js applications",
      image: "/blog/dynamic-routing.jpg",
      date: "2024-03-15",
      authors: ["Admin"],
      categories: ["nextjs"],
      body: { raw: "# Dynamic Routing and Static Regeneration\n\nLearn how to implement dynamic routes..." },
      images: []
    },
    "preview-mode-headless-cms": {
      _id: "3",
      slug: "/blog/preview-mode-headless-cms",
      title: "Preview Mode for Headless CMS", 
      description: "Implement preview mode for your headless CMS content",
      image: "/blog/preview-mode.jpg",
      date: "2024-03-10",
      authors: ["Admin"],
      categories: ["cms"],
      body: { raw: "# Preview Mode for Headless CMS\n\nEnable content preview for your headless CMS..." },
      images: []
    }
  };

  const post = posts[resolvedParams.slug as keyof typeof posts];

  if (!post) {
    notFound();
  }

  const category = BLOG_CATEGORIES.find(
    (category) => category.slug === post.categories[0],
  ) || BLOG_CATEGORIES[0];

  const relatedArticles = Object.values(posts)
    .filter((p) => p._id !== post._id)
    .filter((p) => p.categories && post.categories && p.categories.some((c) => post.categories.includes(c)))
    .slice(0, 2);

  const toc = await getTableOfContents(post.body.raw);

  const [thumbnailBlurhash, images] = await Promise.all([
    getBlurDataURL(post.image),
    Promise.resolve([])
  ]);

  return (
    <>
      <MaxWidthWrapper className="pt-6 md:pt-10">
        <div className="flex flex-col space-y-4">
          <div className="flex items-center space-x-4">
            <Link
              href={`/blog/category/${category.slug}`}
              className={cn(
                buttonVariants({
                  variant: "outline",
                  size: "sm",
                }),
                "h-8 rounded-lg",
              )}
            >
              {category.title}
            </Link>
            <time
              dateTime={post.date}
              className="text-sm font-medium text-muted-foreground"
            >
              {formatDate(post.date)}
            </time>
          </div>
          <h1 className="font-heading text-3xl text-foreground sm:text-4xl">
            {post.title}
          </h1>
          <p className="text-base text-muted-foreground md:text-lg">
            {post.description}
          </p>
          <div className="flex flex-nowrap items-center space-x-5 pt-1 md:space-x-8">
            {post.authors.map((author) => (
              <Author username={author} key={post._id + author} />
            ))}
          </div>
        </div>
      </MaxWidthWrapper>

      <div className="relative">
        <div className="absolute top-52 w-full border-t" />

        <MaxWidthWrapper className="grid grid-cols-4 gap-10 pt-8 max-md:px-0">
          <div className="relative col-span-4 mb-10 flex flex-col space-y-8 border-y bg-background md:rounded-xl md:border lg:col-span-3">
            <BlurImage
              alt={post.title}
              blurDataURL={thumbnailBlurhash ?? placeholderBlurhash}
              className="aspect-[1200/630] border-b object-cover md:rounded-t-xl"
              width={1200}
              height={630}
              priority
              placeholder="blur"
              src={post.image}
              sizes="(max-width: 768px) 770px, 1000px"
            />
            <div className="px-[.8rem] pb-10 md:px-8">
              <div className="prose prose-lg max-w-none">
                <div dangerouslySetInnerHTML={{ __html: post.body.raw.replace(/\n/g, '<br />') }} />
              </div>
            </div>
          </div>

          <div className="sticky top-20 col-span-1 mt-52 hidden flex-col divide-y divide-muted self-start pb-24 lg:flex">
            <DashboardTableOfContents toc={toc} />
          </div>
        </MaxWidthWrapper>
      </div>

      <MaxWidthWrapper>
        {relatedArticles.length > 0 && (
          <div className="flex flex-col space-y-4 pb-16">
            <p className="font-heading text-2xl text-foreground">
              More Articles
            </p>

            <div className="grid grid-cols-1 gap-3 md:grid-cols-2 lg:gap-6">
              {relatedArticles.map((relatedPost) => (
                <Link
                  key={relatedPost.slug}
                  href={relatedPost.slug}
                  className="flex flex-col space-y-2 rounded-xl border p-5 transition-colors duration-300 hover:bg-muted/80"
                >
                  <h3 className="font-heading text-xl text-foreground">
                    {relatedPost.title}
                  </h3>
                  <p className="line-clamp-2 text-[15px] text-muted-foreground">
                    {relatedPost.description}
                  </p>
                  <p className="text-sm text-muted-foreground">
                    {formatDate(relatedPost.date)}
                  </p>
                </Link>
              ))}
            </div>
          </div>
        )}
      </MaxWidthWrapper>
    </>
  );
}
