'use client';

import { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import { FeaturedPost, CategoryTabs, BlogPostCard } from '../components/pages';
import { ALL_BLOG_POSTS, CATEGORIES } from '../components/pages/data/blog-posts';

export default function BlogPage() {
  const [selectedCategory, setSelectedCategory] = useState<string>('All');

  const featuredPost = useMemo(() => {
    return ALL_BLOG_POSTS.find((post) => post.featured) || ALL_BLOG_POSTS[0];
  }, []);

  const filteredPosts = useMemo(() => {
    if (selectedCategory === 'All') {
      return ALL_BLOG_POSTS.filter((post) => !post.featured);
    }
    return ALL_BLOG_POSTS.filter(
      (post) => post.category === selectedCategory && !post.featured
    );
  }, [selectedCategory]);

  return (
    <div className="min-h-screen bg-black text-white relative">
      <div className="relative z-10">
        <motion.header 
          className="relative z-10 bg-black"
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.5, ease: "easeOut" }}
        >
          <div className="max-w-[1920px] mx-auto px-4 md:px-6 lg:px-8 py-3.5 md:py-4">
            <div className="flex items-center justify-between">
              <motion.a 
                href="/" 
                className="flex items-center gap-2.5 text-base text-white hover:opacity-80 transition-opacity duration-200 no-underline"
                aria-label="Home - bulk"
                whileHover={{ scale: 1.01 }}
                whileTap={{ scale: 0.99 }}
                transition={{ type: "spring", stiffness: 400, damping: 17 }}
              >
                <div className="flex items-center gap-2.5">
                  <div className="w-6 h-6 flex items-center justify-center flex-shrink-0">
                    <svg
                      viewBox="0 0 24 24"
                      className="w-full h-full"
                      aria-hidden="true"
                      role="img"
                      preserveAspectRatio="xMidYMid meet"
                    >
                      <defs>
                        <linearGradient id="gradient-header-blog" x1="0%" y1="0%" x2="0%" y2="100%">
                          <stop offset="0%" stopColor="#8800ff" />
                          <stop offset="16.66%" stopColor="#0000ff" />
                          <stop offset="33.33%" stopColor="#0088ff" />
                          <stop offset="50%" stopColor="#00ff00" />
                          <stop offset="66.66%" stopColor="#ffdd00" />
                          <stop offset="83.33%" stopColor="#ff8800" />
                          <stop offset="100%" stopColor="#ff0000" />
                        </linearGradient>
                      </defs>
                      <path d="M7 20L12 4L17 20H14.5L12 12.5L9.5 20H7Z" fill="url(#gradient-header-blog)" />
                    </svg>
                  </div>
                  <span className="font-normal text-white whitespace-nowrap text-base">bulk</span>
                </div>
              </motion.a>
              
              <nav className="hidden md:flex items-center gap-7 lg:gap-8" aria-label="Main navigation">
                <motion.button
                  className="text-white hover:opacity-70 transition-opacity duration-200 font-normal text-sm focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 focus:ring-offset-black rounded"
                  aria-label="Navigate to Product page"
                  onClick={() => window.location.href = '/product'}
                  whileHover={{ opacity: 0.7 }}
                  whileTap={{ scale: 0.98 }}
                >
                  Product
                </motion.button>
                <motion.button
                  className="text-white hover:opacity-70 transition-opacity duration-200 font-normal text-sm focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 focus:ring-offset-black rounded"
                  aria-label="Navigate to Use Cases page"
                  onClick={() => window.location.href = '/use-cases'}
                  whileHover={{ opacity: 0.7 }}
                  whileTap={{ scale: 0.98 }}
                >
                  Use Cases
                </motion.button>
                <motion.button
                  className="text-white hover:opacity-70 transition-opacity duration-200 font-normal text-sm focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 focus:ring-offset-black rounded"
                  aria-label="Navigate to Pricing page"
                  onClick={() => window.location.href = '/pricing'}
                  whileHover={{ opacity: 0.7 }}
                  whileTap={{ scale: 0.98 }}
                >
                  Pricing
                </motion.button>
                <motion.button
                  className="text-white hover:opacity-70 transition-opacity duration-200 font-normal text-sm focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 focus:ring-offset-black rounded"
                  aria-label="Navigate to Blog page"
                  onClick={() => window.location.href = '/blog'}
                  whileHover={{ opacity: 0.7 }}
                  whileTap={{ scale: 0.98 }}
                >
                  Blog
                </motion.button>
                <motion.a 
                  href="#overview" 
                  className="text-white hover:opacity-70 transition-opacity duration-200 font-normal text-sm focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 focus:ring-offset-black rounded no-underline"
                  aria-label="See overview"
                  whileHover={{ opacity: 0.7 }}
                  whileTap={{ scale: 0.98 }}
                  transition={{ type: "spring", stiffness: 400, damping: 17 }}
                >
                  See overview
                </motion.a>
                <motion.button 
                  className="bg-white text-black px-4 py-2 rounded-md hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-white focus:ring-offset-2 focus:ring-offset-black transition-colors duration-200 font-normal text-sm flex items-center gap-1.5 disabled:opacity-50 disabled:cursor-not-allowed whitespace-nowrap"
                  aria-label="Download bulk"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  transition={{ type: "spring", stiffness: 400, damping: 17 }}
                  type="button"
                >
                  Download
                  <svg className="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </motion.button>
              </nav>
            </div>
          </div>
        </motion.header>
      </div>
      <main className="relative z-10 pt-20 pb-16">
        <div className="max-w-[1920px] mx-auto px-4 md:px-6 lg:px-8">
          {/* Background "Blog" text */}
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <h1 className="text-[200px] md:text-[300px] lg:text-[400px] font-bold text-white opacity-[0.03] font-sans antialiased select-none">
              Blog
            </h1>
          </div>

          {/* Featured Blog Post Section */}
          <section className="mb-16 md:mb-20 relative z-10">
            <FeaturedPost
              id={featuredPost.id}
              title={featuredPost.title}
              ariaLabel={featuredPost.ariaLabel}
            />
          </section>

          {/* Latest Blog Section */}
          <section className="relative z-10">
            <div className="mb-8 md:mb-12">
              <h2 className="text-3xl md:text-4xl font-normal text-white mb-6 font-sans antialiased">Latest Blog</h2>
              
              {/* Category Tabs */}
              <CategoryTabs
                categories={CATEGORIES}
                selectedCategory={selectedCategory}
                onCategoryChange={setSelectedCategory}
              />
            </div>

            {/* Blog Posts Grid */}
            <div
              role="tabpanel"
              id={`tabpanel-${selectedCategory.toLowerCase()}`}
              aria-labelledby={`tab-${selectedCategory.toLowerCase()}`}
              className="grid md:grid-cols-2 gap-8 md:gap-12"
            >
              {filteredPosts.map((post, index) => (
                <BlogPostCard
                  key={post.id}
                  id={post.id}
                  date={post.date}
                  category={post.category}
                  title={post.title}
                  ariaLabel={post.ariaLabel}
                  index={index}
                />
              ))}
            </div>
          </section>
        </div>
      </main>
      <footer className="border-t border-gray-700 py-12 md:py-16 relative z-10" role="contentinfo">
        <div className="max-w-[1920px] mx-auto px-4 md:px-6 lg:px-8">
          <div className="grid md:grid-cols-2 gap-12 md:gap-16 mb-12 md:mb-16">
            <nav className="space-y-4" aria-label="Footer navigation - Product">
              <div className="space-y-3">
                <a href="/download" className="block text-white hover:opacity-70 transition-opacity text-sm focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 focus:ring-offset-black rounded">Download</a>
                <a href="/product" className="block text-white hover:opacity-70 transition-opacity text-sm focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 focus:ring-offset-black rounded">Product</a>
                <a href="#" className="block text-white hover:opacity-70 transition-opacity text-sm focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 focus:ring-offset-black rounded">Doc</a>
                <a href="#" className="block text-white hover:opacity-70 transition-opacity text-sm focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 focus:ring-offset-black rounded">Changelog</a>
              </div>
            </nav>
            <nav className="space-y-4" aria-label="Footer navigation - Resources">
              <div className="space-y-3">
                <a href="/blog" className="block text-white hover:opacity-70 transition-opacity text-sm focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 focus:ring-offset-black rounded">Blog</a>
                <a href="/pricing" className="block text-white hover:opacity-70 transition-opacity text-sm focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 focus:ring-offset-black rounded">Pricing</a>
                <a href="/use-cases" className="block text-white hover:opacity-70 transition-opacity text-sm focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 focus:ring-offset-black rounded">Use Cases</a>
              </div>
            </nav>
          </div>
          
          <div className="border-t border-gray-700 pt-8">
            <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6">
              <a href="/" className="text-white hover:opacity-70 transition-opacity text-lg font-normal focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 focus:ring-offset-black rounded" aria-label="Home - bulk">
                bulk
              </a>
              <nav className="flex flex-wrap gap-6 text-sm" aria-label="Footer legal navigation">
                <a href="#" className="text-white hover:opacity-70 transition-opacity focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 focus:ring-offset-black rounded">About bulk</a>
                <a href="#" className="text-white hover:opacity-70 transition-opacity focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 focus:ring-offset-black rounded">bulk Product</a>
                <a href="#" className="text-white hover:opacity-70 transition-opacity focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 focus:ring-offset-black rounded">Privacy</a>
                <a href="#" className="text-white hover:opacity-70 transition-opacity focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 focus:ring-offset-black rounded">Term</a>
              </nav>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

