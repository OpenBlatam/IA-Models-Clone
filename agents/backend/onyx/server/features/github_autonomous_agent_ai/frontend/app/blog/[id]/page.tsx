'use client';

import { useParams, useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import { PageLayout } from '../../components/pages';
import { BLOG_POSTS_CONTENT } from '../../components/pages/data/blog-posts-content';
import { useMemo } from 'react';

export default function BlogPostPage() {
  const params = useParams();
  const router = useRouter();
  const postId = params.id as string;

  const post = useMemo(() => {
    return BLOG_POSTS_CONTENT[postId];
  }, [postId]);

  if (!post) {
    return (
      <PageLayout>
        <div className="text-center">
          <h1 className="text-3xl font-normal mb-4">Article not found</h1>
          <button
            onClick={() => router.push('/blog')}
            className="text-black underline hover:opacity-70 transition-opacity"
          >
            Back to Blog
          </button>
        </div>
      </PageLayout>
    );
  }

  return (
    <PageLayout>
          {/* Close Button */}
          <div className="mb-8">
            <button
              onClick={() => router.push('/blog')}
              className="text-black hover:opacity-70 transition-opacity focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 rounded p-2 -m-2"
              aria-label="Close article and return to blog"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Article Image */}
          {post.image && (
            <div className="mb-8">
              <img
                src={post.image}
                alt={post.title}
                className="w-full h-auto rounded-lg"
              />
            </div>
          )}

          {/* Article Content */}
          <article className="max-w-4xl mx-auto">
            {/* Metadata */}
            <div className="mb-6 text-gray-500 text-sm">
              {post.category} {post.date}
            </div>

            {/* Title */}
            <motion.h1
              className="text-4xl md:text-5xl lg:text-6xl font-normal text-black mb-4"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
            >
              {post.title}
            </motion.h1>

            {/* Author */}
            <div className="mb-12 text-gray-500 text-sm">
              {post.author}
            </div>

            {/* Content */}
            <motion.div
              className="prose prose-lg max-w-none text-black"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
            >
              <div className="whitespace-pre-line leading-relaxed text-base md:text-lg">
                {post.content}
              </div>
            </motion.div>

            {/* Download Button */}
            <div className="mt-12 pt-8 border-t border-gray-200">
              <a
                href="#"
                className="inline-block text-black underline hover:opacity-70 transition-opacity text-base focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 rounded p-2 -m-2"
              >
                Download bulk
              </a>
            </div>
          </article>
    </PageLayout>
  );
}

