import { Metadata } from "next";
import { notFound } from "next/navigation";

import { BLOG_CATEGORIES } from "@/config/blog";
import { constructMetadata, getBlurDataURL } from "@/lib/utils";
import { BlogCard } from "@/components/content/blog-card";

export async function generateStaticParams() {
  return BLOG_CATEGORIES.map((category) => ({
    slug: category.slug,
  }));
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string }>;
}): Promise<Metadata | undefined> {
  const resolvedParams = await params;
  const category = BLOG_CATEGORIES.find(
    (category) => category.slug === resolvedParams.slug,
  );
  if (!category) {
    return;
  }

  const { title, description } = category;

  return constructMetadata({
    title: `${title} Posts – Next SaaS Starter`,
    description,
  });
}

export default async function BlogCategory({
  params,
}: {
  params: Promise<{
    slug: string;
  }>;
}) {
  const resolvedParams = await params;
  const category = BLOG_CATEGORIES.find((ctg) => ctg.slug === resolvedParams.slug);

  if (!category) {
    notFound();
  }

  const staticArticles = [
    {
      _id: "1",
      slug: "/blog/server-client-components",
      title: "Understanding Server and Client Components",
      description: "Learn the differences between server and client components in Next.js 13+",
      image: "/blog/server-client-components.jpg",
      date: "2024-03-20",
      authors: ["Admin"],
      categories: ["nextjs"]
    },
    {
      _id: "2",
      slug: "/blog/dynamic-routing-static-regeneration",
      title: "Dynamic Routing and Static Regeneration",
      description: "Master dynamic routing and ISR in Next.js applications",
      image: "/blog/dynamic-routing.jpg",
      date: "2024-03-15",
      authors: ["Admin"],
      categories: ["nextjs"]
    },
    {
      _id: "3",
      slug: "/blog/preview-mode-headless-cms",
      title: "Preview Mode for Headless CMS",
      description: "Implement preview mode for your headless CMS content",
      image: "/blog/preview-mode.jpg",
      date: "2024-03-10",
      authors: ["Admin"],
      categories: ["cms"]
    }
  ];

  const allArticles = await Promise.all(
    staticArticles.map(async (article) => ({
      ...article,
      blurDataURL: await getBlurDataURL(article.image)
    }))
  );

  const articles = allArticles.filter(article => 
    article.categories.includes(category.slug)
  );

  return (
    <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
      {articles.length === 0 && (
        <div className="col-span-full text-center py-8">
          <p className="text-muted-foreground">No hay artículos disponibles en esta categoría.</p>
        </div>
      )}
      {articles.map((article, idx) => (
        <BlogCard key={article._id} data={article} priority={idx <= 2} />
      ))}
    </div>
  );
}
