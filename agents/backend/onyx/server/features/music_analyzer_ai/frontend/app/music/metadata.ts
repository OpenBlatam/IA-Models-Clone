/**
 * Metadata configuration for the music page.
 * Provides SEO and metadata information.
 */

import type { Metadata } from 'next';

/**
 * Music page metadata.
 * Optimized for SEO and social sharing.
 */
export const musicPageMetadata: Metadata = {
  title: 'Music Analyzer AI | Blatam Academy',
  description:
    'Analiza canciones, obtén insights musicales y coaching personalizado con IA. Descubre patrones musicales, compara tracks y recibe recomendaciones inteligentes.',
  keywords: [
    'music analysis',
    'AI music analyzer',
    'music insights',
    'track analysis',
    'music recommendations',
    'musical coaching',
  ],
  authors: [{ name: 'Blatam Academy' }],
  openGraph: {
    title: 'Music Analyzer AI | Blatam Academy',
    description:
      'Analiza canciones con IA y obtén insights musicales personalizados',
    type: 'website',
    locale: 'es_ES',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Music Analyzer AI',
    description: 'Analiza canciones con IA',
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
};

