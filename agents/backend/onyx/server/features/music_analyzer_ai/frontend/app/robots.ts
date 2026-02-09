/**
 * Robots.txt configuration.
 * Controls search engine crawling behavior.
 */

import { MetadataRoute } from 'next';
import { appConfig } from '@/lib/config/app';

/**
 * Robots.txt configuration.
 * @returns Robots configuration
 */
export default function robots(): MetadataRoute.Robots {
  return {
    rules: [
      {
        userAgent: '*',
        allow: '/',
        disallow: ['/api/', '/_next/', '/admin/'],
      },
      {
        userAgent: 'Googlebot',
        allow: '/',
        disallow: ['/api/', '/_next/'],
      },
    ],
    sitemap: `${appConfig.url}/sitemap.xml`,
  };
}

