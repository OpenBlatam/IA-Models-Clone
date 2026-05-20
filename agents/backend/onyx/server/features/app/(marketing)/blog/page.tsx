
import { constructMetadata, getBlurDataURL } from "@/lib/utils";
import { BlogPosts } from "@/components/content/blog-posts";

export const metadata = constructMetadata({
  title: "Blog – SaaS Starter",
  description: "Latest news and updates from Next SaaS Starter.",
});

export default async function BlogPage() {
  const posts = await Promise.all([
    {
      _id: "1",
      slug: "/blog/server-client-components",
      title: "Understanding Server and Client Components",
      description: "Learn the differences between server and client components in Next.js 13+",
      image: "/blog/server-client-components.jpg",
      date: "2024-03-20",
      authors: ["Admin"],
      blurDataURL: await getBlurDataURL("/blog/server-client-components.jpg")
    },
    {
      _id: "2",
      slug: "/blog/dynamic-routing-static-regeneration",
      title: "Dynamic Routing and Static Regeneration",
      description: "Master dynamic routing and ISR in Next.js applications",
      image: "/blog/dynamic-routing.jpg",
      date: "2024-03-15",
      authors: ["Admin"],
      blurDataURL: await getBlurDataURL("/blog/dynamic-routing.jpg")
    },
    {
      _id: "3",
      slug: "/blog/preview-mode-headless-cms",
      title: "Preview Mode for Headless CMS", 
      description: "Implement preview mode for your headless CMS content",
      image: "/blog/preview-mode.jpg",
      date: "2024-03-10",
      authors: ["Admin"],
      blurDataURL: await getBlurDataURL("/blog/preview-mode.jpg")
    }
  ]);

  return <BlogPosts posts={posts} />;
}
