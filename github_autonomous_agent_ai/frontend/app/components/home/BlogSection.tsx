'use client';

import { useMemo } from 'react';

interface BlogPost {
  id: string;
  date: string;
  category: string;
  title: string;
  ariaLabel: string;
}

const BLOG_POSTS: BlogPost[] = [
  {
    id: 'nano-banana-pro',
    date: 'Nov 20, 2025',
    category: 'Product',
    title: 'Nano Banana Pro in bulk',
    ariaLabel: 'Read blog post: Nano Banana Pro in bulk',
  },
  {
    id: 'introducing-antigravity',
    date: 'Nov 18, 2025',
    category: 'Product',
    title: 'Introducing bulk',
    ariaLabel: 'Read blog post: Introducing bulk',
  },
];

export function BlogSection() {
  const blogPosts = useMemo(() => BLOG_POSTS, []);

  return (
    <section 
      className="py-16 md:py-20 lg:py-24 border-b border-gray-200 relative z-10"
      aria-labelledby="blog-heading"
    >
      <div className="max-w-[1920px] mx-auto px-4 md:px-6 lg:px-8">
        <div className="flex justify-between items-center mb-10 md:mb-12">
          <h2 id="blog-heading" className="text-black text-3xl font-normal leading-[1.2] tracking-[-0.02em]">Latest Updates</h2>
          <button 
            className="text-black hover:opacity-70 underline transition-opacity duration-200 ease-in-out text-base leading-normal font-normal focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 rounded"
            type="button"
            aria-label="View all blog posts"
          >
            View blog
          </button>
        </div>
        <div className="grid md:grid-cols-2 gap-10">
          {blogPosts.map((post) => (
            <article key={post.id}>
              <a 
                href="#" 
                className="block space-y-3 hover:opacity-80 transition-opacity duration-200 ease-in-out group focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 rounded p-2 -m-2"
                aria-label={post.ariaLabel}
              >
                <div className="text-gray-500 text-sm leading-normal font-medium">{post.date} • {post.category}</div>
                <h3 className="text-black text-2xl font-normal leading-[1.3] tracking-[-0.01em] group-hover:underline">{post.title}</h3>
                <span className="text-black underline inline-block text-sm leading-normal font-normal">Read blog</span>
              </a>
            </article>
          ))}
        </div>
        <div className="flex justify-center gap-3 mt-12" role="group" aria-label="Blog navigation controls">
          <button 
            className="text-black hover:opacity-70 p-2 rounded-full hover:bg-gray-100 transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2"
            aria-label="Previous blog posts"
            type="button"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 19l-7-7 7-7" />
            </svg>
          </button>
          <button 
            className="text-black hover:opacity-70 p-2 rounded-full hover:bg-gray-100 transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2"
            aria-label="Next blog posts"
            type="button"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5l7 7-7 7" />
            </svg>
          </button>
        </div>
      </div>
    </section>
  );
}

