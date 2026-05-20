'use client';

import { motion } from 'framer-motion';
import { PageLayout, DownloadButton } from '../components/pages';
import { PLATFORMS } from '../components/pages/data/platforms';

export default function DownloadPage() {
  return (
    <PageLayout>
      <div className="max-w-4xl mx-auto">
        {/* Page Title */}
        <motion.div
          className="text-center mb-12 md:mb-16"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-black mb-4 leading-[1.08] tracking-[-0.03em] font-sans antialiased">
            Download bulk
          </h1>
        </motion.div>

        {/* Platform Sections */}
        <div className="space-y-12 md:space-y-16">
          {PLATFORMS.map((platform, platformIndex) => (
            <motion.section
              key={platform.id}
              className="space-y-6"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: platformIndex * 0.1 }}
            >
              <h2 className="text-2xl md:text-3xl font-normal text-black font-sans antialiased">
                {platform.name}
              </h2>
              
              <div className="flex flex-col sm:flex-row gap-4">
                {platform.downloads.map((download) => (
                  <DownloadButton
                    key={download.id}
                    label={download.label}
                    ariaLabel={download.ariaLabel}
                    variant="primary"
                  />
                ))}
              </div>
            </motion.section>
          ))}
        </div>

        {/* Additional Info */}
        <motion.div
          className="mt-16 pt-8 border-t border-gray-200 text-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.4 }}
        >
          <p className="text-sm text-gray-500 mb-4">
            Need help? Check out our{' '}
            <a
              href="#"
              className="text-black underline hover:opacity-70 transition-opacity"
            >
              documentation
            </a>
            {' '}or{' '}
            <a
              href="#"
              className="text-black underline hover:opacity-70 transition-opacity"
            >
              support
            </a>
            .
          </p>
        </motion.div>
      </div>
    </PageLayout>
  );
}

