/**
 * Sitemap configuration.
 * Generates XML sitemap for search engines.
 */

import { MetadataRoute } from 'next';
import { appConfig } from '@/lib/config/app';
import { ROUTES } from '@/lib/constants';

/**
 * Sitemap configuration.
 * @returns Sitemap entries
 */
export default function sitemap(): MetadataRoute.Sitemap {
  const baseUrl = appConfig.url;

  return [
    {
      url: baseUrl,
      lastModified: new Date(),
      changeFrequency: 'daily',
      priority: 1,
    },
    {
      url: `${baseUrl}${ROUTES.MUSIC}`,
      lastModified: new Date(),
      changeFrequency: 'daily',
      priority: 0.9,
    },
    {
      url: `${baseUrl}${ROUTES.ROBOT}`,
      lastModified: new Date(),
      changeFrequency: 'daily',
      priority: 0.9,
    },
  ];
}

