'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { PageLayout, DownloadButton } from '../components/pages';
import { FEATURES } from '../components/pages/data/features';

export default function ProductPage() {
  const [currentFeatureIndex, setCurrentFeatureIndex] = useState(0);

  const nextFeature = () => {
    setCurrentFeatureIndex((prev) => (prev + 1) % FEATURES.length);
  };

  const prevFeature = () => {
    setCurrentFeatureIndex((prev) => (prev - 1 + FEATURES.length) % FEATURES.length);
  };

  const currentFeature = FEATURES[currentFeatureIndex];

  return (
    <PageLayout>
      {/* Hero Section */}
      <section className="mb-16 md:mb-20">
        <motion.div
          className="text-center"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <h1 className="text-5xl md:text-6xl lg:text-7xl xl:text-8xl font-bold text-black mb-8 md:mb-12 leading-[1.08] tracking-[-0.03em] font-sans antialiased max-w-5xl mx-auto">
            Agents that help you achieve liftoff
            <span className="inline-block w-0.5 h-[1.2em] ml-1 relative align-middle">
              <svg className="absolute inset-0 w-full h-full" viewBox="0 0 2 24" preserveAspectRatio="none" aria-hidden="true">
                <defs>
                  <linearGradient id="cursor-gradient-product" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" stopColor="#8800ff" />
                    <stop offset="16.66%" stopColor="#0000ff" />
                    <stop offset="33.33%" stopColor="#0088ff" />
                    <stop offset="50%" stopColor="#00ff00" />
                    <stop offset="66.66%" stopColor="#ffdd00" />
                    <stop offset="83.33%" stopColor="#ff8800" />
                    <stop offset="100%" stopColor="#ff0000" />
                  </linearGradient>
                </defs>
                <rect width="2" height="24" fill="url(#cursor-gradient-product)" className="animate-pulse" />
              </svg>
            </span>
          </h1>
          
          {/* Download Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            <DownloadButton
              label="Download for x64"
              ariaLabel="Download for x64"
              variant="primary"
            />
            <DownloadButton
              label="Download for ARM64"
              ariaLabel="Download for ARM64"
              variant="secondary"
            />
          </div>
        </motion.div>
      </section>

      {/* Features Section */}
      <section className="mb-16 md:mb-20">
        <div className="max-w-4xl mx-auto">
          {/* Navigation Controls */}
          <div className="flex items-center justify-between mb-8">
            <div className="text-sm text-gray-500 font-normal">
              Scroll to explore feature
            </div>
            <div className="flex gap-2">
              <button
                onClick={prevFeature}
                className="text-black hover:opacity-70 transition-opacity p-2 rounded-full hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2"
                aria-label="Previous feature"
                type="button"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 19l-7-7 7-7" />
                </svg>
              </button>
              <button
                onClick={nextFeature}
                className="text-black hover:opacity-70 transition-opacity p-2 rounded-full hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2"
                aria-label="Next feature"
                type="button"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5l7 7-7 7" />
                </svg>
              </button>
            </div>
          </div>

          {/* Feature Display */}
          <motion.div
            key={currentFeatureIndex}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.4 }}
            className="space-y-6"
          >
            <h2 className="text-3xl md:text-4xl font-normal text-black font-sans antialiased">
              {currentFeature.title}
            </h2>
            <p className="text-lg md:text-xl text-black leading-relaxed font-normal font-sans antialiased">
              {currentFeature.description}
            </p>
          </motion.div>

          {/* Feature Indicators */}
          <div className="flex gap-2 justify-center mt-12">
            {FEATURES.map((_, index) => (
              <button
                key={index}
                onClick={() => setCurrentFeatureIndex(index)}
                className={`w-2 h-2 rounded-full transition-all focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 ${
                  index === currentFeatureIndex
                    ? 'bg-black w-8'
                    : 'bg-gray-300 hover:bg-gray-400'
                }`}
                aria-label={`Go to feature ${index + 1}`}
                type="button"
              />
            ))}
          </div>
        </div>
      </section>

      {/* Download Section */}
      <section className="border-t border-gray-200 pt-16">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl md:text-4xl font-normal text-black mb-8 font-sans antialiased">
            Ready to get started?
          </h2>
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            <DownloadButton
              label="Download for x64"
              ariaLabel="Download for x64"
              variant="primary"
            />
            <DownloadButton
              label="Download for ARM64"
              ariaLabel="Download for ARM64"
              variant="secondary"
            />
          </div>
        </div>
      </section>
    </PageLayout>
  );
}

