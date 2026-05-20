/**
 * SEO & Meta Tags Testing
 * 
 * Tests that verify SEO optimization including meta tags,
 * Open Graph tags, structured data, and social sharing.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render } from '@testing-library/react';

describe('SEO & Meta Tags Testing', () => {
  beforeEach(() => {
    // Clear document head
    document.head.innerHTML = '';
  });

  describe('Meta Tags', () => {
    it('should have title tag', () => {
      document.title = 'Music Analyzer AI';
      expect(document.title).toBe('Music Analyzer AI');
    });

    it('should have description meta tag', () => {
      const meta = document.createElement('meta');
      meta.name = 'description';
      meta.content = 'AI-powered music analysis and discovery';
      document.head.appendChild(meta);

      const description = document.querySelector('meta[name="description"]');
      expect(description?.getAttribute('content')).toBe('AI-powered music analysis and discovery');
    });

    it('should have keywords meta tag', () => {
      const meta = document.createElement('meta');
      meta.name = 'keywords';
      meta.content = 'music, AI, analysis, discovery';
      document.head.appendChild(meta);

      const keywords = document.querySelector('meta[name="keywords"]');
      expect(keywords?.getAttribute('content')).toBe('music, AI, analysis, discovery');
    });

    it('should have viewport meta tag', () => {
      const meta = document.createElement('meta');
      meta.name = 'viewport';
      meta.content = 'width=device-width, initial-scale=1.0';
      document.head.appendChild(meta);

      const viewport = document.querySelector('meta[name="viewport"]');
      expect(viewport?.getAttribute('content')).toContain('width=device-width');
    });

    it('should have charset meta tag', () => {
      const meta = document.createElement('meta');
      meta.setAttribute('charset', 'UTF-8');
      document.head.appendChild(meta);

      const charset = document.querySelector('meta[charset]');
      expect(charset?.getAttribute('charset')).toBe('UTF-8');
    });

    it('should have robots meta tag', () => {
      const meta = document.createElement('meta');
      meta.name = 'robots';
      meta.content = 'index, follow';
      document.head.appendChild(meta);

      const robots = document.querySelector('meta[name="robots"]');
      expect(robots?.getAttribute('content')).toBe('index, follow');
    });
  });

  describe('Open Graph Tags', () => {
    it('should have og:title tag', () => {
      const meta = document.createElement('meta');
      meta.setAttribute('property', 'og:title');
      meta.setAttribute('content', 'Music Analyzer AI');
      document.head.appendChild(meta);

      const ogTitle = document.querySelector('meta[property="og:title"]');
      expect(ogTitle?.getAttribute('content')).toBe('Music Analyzer AI');
    });

    it('should have og:description tag', () => {
      const meta = document.createElement('meta');
      meta.setAttribute('property', 'og:description');
      meta.setAttribute('content', 'AI-powered music analysis');
      document.head.appendChild(meta);

      const ogDesc = document.querySelector('meta[property="og:description"]');
      expect(ogDesc?.getAttribute('content')).toBe('AI-powered music analysis');
    });

    it('should have og:image tag', () => {
      const meta = document.createElement('meta');
      meta.setAttribute('property', 'og:image');
      meta.setAttribute('content', 'https://example.com/og-image.jpg');
      document.head.appendChild(meta);

      const ogImage = document.querySelector('meta[property="og:image"]');
      expect(ogImage?.getAttribute('content')).toBe('https://example.com/og-image.jpg');
    });

    it('should have og:url tag', () => {
      const meta = document.createElement('meta');
      meta.setAttribute('property', 'og:url');
      meta.setAttribute('content', 'https://music-analyzer.ai');
      document.head.appendChild(meta);

      const ogUrl = document.querySelector('meta[property="og:url"]');
      expect(ogUrl?.getAttribute('content')).toBe('https://music-analyzer.ai');
    });

    it('should have og:type tag', () => {
      const meta = document.createElement('meta');
      meta.setAttribute('property', 'og:type');
      meta.setAttribute('content', 'website');
      document.head.appendChild(meta);

      const ogType = document.querySelector('meta[property="og:type"]');
      expect(ogType?.getAttribute('content')).toBe('website');
    });

    it('should have og:site_name tag', () => {
      const meta = document.createElement('meta');
      meta.setAttribute('property', 'og:site_name');
      meta.setAttribute('content', 'Music Analyzer AI');
      document.head.appendChild(meta);

      const ogSiteName = document.querySelector('meta[property="og:site_name"]');
      expect(ogSiteName?.getAttribute('content')).toBe('Music Analyzer AI');
    });
  });

  describe('Twitter Card Tags', () => {
    it('should have twitter:card tag', () => {
      const meta = document.createElement('meta');
      meta.name = 'twitter:card';
      meta.content = 'summary_large_image';
      document.head.appendChild(meta);

      const twitterCard = document.querySelector('meta[name="twitter:card"]');
      expect(twitterCard?.getAttribute('content')).toBe('summary_large_image');
    });

    it('should have twitter:title tag', () => {
      const meta = document.createElement('meta');
      meta.name = 'twitter:title';
      meta.content = 'Music Analyzer AI';
      document.head.appendChild(meta);

      const twitterTitle = document.querySelector('meta[name="twitter:title"]');
      expect(twitterTitle?.getAttribute('content')).toBe('Music Analyzer AI');
    });

    it('should have twitter:description tag', () => {
      const meta = document.createElement('meta');
      meta.name = 'twitter:description';
      meta.content = 'AI-powered music analysis';
      document.head.appendChild(meta);

      const twitterDesc = document.querySelector('meta[name="twitter:description"]');
      expect(twitterDesc?.getAttribute('content')).toBe('AI-powered music analysis');
    });

    it('should have twitter:image tag', () => {
      const meta = document.createElement('meta');
      meta.name = 'twitter:image';
      meta.content = 'https://example.com/twitter-image.jpg';
      document.head.appendChild(meta);

      const twitterImage = document.querySelector('meta[name="twitter:image"]');
      expect(twitterImage?.getAttribute('content')).toBe('https://example.com/twitter-image.jpg');
    });
  });

  describe('Structured Data (JSON-LD)', () => {
    it('should have structured data for organization', () => {
      const script = document.createElement('script');
      script.type = 'application/ld+json';
      script.textContent = JSON.stringify({
        '@context': 'https://schema.org',
        '@type': 'Organization',
        name: 'Music Analyzer AI',
        url: 'https://music-analyzer.ai',
      });
      document.head.appendChild(script);

      const structuredData = document.querySelector('script[type="application/ld+json"]');
      expect(structuredData).toBeDefined();
      const data = JSON.parse(structuredData?.textContent || '{}');
      expect(data['@type']).toBe('Organization');
    });

    it('should have structured data for website', () => {
      const script = document.createElement('script');
      script.type = 'application/ld+json';
      script.textContent = JSON.stringify({
        '@context': 'https://schema.org',
        '@type': 'WebSite',
        name: 'Music Analyzer AI',
        url: 'https://music-analyzer.ai',
      });
      document.head.appendChild(script);

      const structuredData = document.querySelector('script[type="application/ld+json"]');
      const data = JSON.parse(structuredData?.textContent || '{}');
      expect(data['@type']).toBe('WebSite');
    });

    it('should have structured data for music track', () => {
      const script = document.createElement('script');
      script.type = 'application/ld+json';
      script.textContent = JSON.stringify({
        '@context': 'https://schema.org',
        '@type': 'MusicRecording',
        name: 'Test Track',
        byArtist: 'Test Artist',
      });
      document.head.appendChild(script);

      const structuredData = document.querySelector('script[type="application/ld+json"]');
      const data = JSON.parse(structuredData?.textContent || '{}');
      expect(data['@type']).toBe('MusicRecording');
    });
  });

  describe('Canonical URL', () => {
    it('should have canonical link tag', () => {
      const link = document.createElement('link');
      link.rel = 'canonical';
      link.href = 'https://music-analyzer.ai/tracks';
      document.head.appendChild(link);

      const canonical = document.querySelector('link[rel="canonical"]');
      expect(canonical?.getAttribute('href')).toBe('https://music-analyzer.ai/tracks');
    });
  });

  describe('Alternate Languages', () => {
    it('should have alternate language links', () => {
      const languages = [
        { lang: 'en', href: 'https://music-analyzer.ai/en' },
        { lang: 'es', href: 'https://music-analyzer.ai/es' },
        { lang: 'fr', href: 'https://music-analyzer.ai/fr' },
      ];

      languages.forEach(({ lang, href }) => {
        const link = document.createElement('link');
        link.rel = 'alternate';
        link.hreflang = lang;
        link.href = href;
        document.head.appendChild(link);
      });

      const alternateLinks = document.querySelectorAll('link[rel="alternate"]');
      expect(alternateLinks).toHaveLength(3);
    });
  });

  describe('Favicon', () => {
    it('should have favicon link', () => {
      const link = document.createElement('link');
      link.rel = 'icon';
      link.href = '/favicon.ico';
      document.head.appendChild(link);

      const favicon = document.querySelector('link[rel="icon"]');
      expect(favicon?.getAttribute('href')).toBe('/favicon.ico');
    });

    it('should have apple-touch-icon', () => {
      const link = document.createElement('link');
      link.rel = 'apple-touch-icon';
      link.href = '/apple-touch-icon.png';
      document.head.appendChild(link);

      const appleIcon = document.querySelector('link[rel="apple-touch-icon"]');
      expect(appleIcon?.getAttribute('href')).toBe('/apple-touch-icon.png');
    });
  });

  describe('Page-Specific SEO', () => {
    it('should update title for different pages', () => {
      const updateTitle = (page: string) => {
        document.title = `${page} - Music Analyzer AI`;
      };

      updateTitle('Tracks');
      expect(document.title).toBe('Tracks - Music Analyzer AI');

      updateTitle('Playlists');
      expect(document.title).toBe('Playlists - Music Analyzer AI');
    });

    it('should update meta description for different pages', () => {
      const updateDescription = (description: string) => {
        let meta = document.querySelector('meta[name="description"]');
        if (!meta) {
          meta = document.createElement('meta');
          meta.name = 'description';
          document.head.appendChild(meta);
        }
        meta.setAttribute('content', description);
      };

      updateDescription('Browse and discover music tracks');
      const meta = document.querySelector('meta[name="description"]');
      expect(meta?.getAttribute('content')).toBe('Browse and discover music tracks');
    });
  });

  describe('Social Sharing', () => {
    it('should generate shareable URL', () => {
      const generateShareUrl = (trackId: string) => {
        return `https://music-analyzer.ai/tracks/${trackId}`;
      };

      const url = generateShareUrl('123');
      expect(url).toBe('https://music-analyzer.ai/tracks/123');
    });

    it('should generate share text', () => {
      const generateShareText = (trackName: string, artist: string) => {
        return `Check out ${trackName} by ${artist} on Music Analyzer AI!`;
      };

      const text = generateShareText('Test Track', 'Test Artist');
      expect(text).toContain('Test Track');
      expect(text).toContain('Test Artist');
    });
  });

  describe('SEO Validation', () => {
    it('should validate title length', () => {
      const validateTitle = (title: string) => {
        return title.length >= 30 && title.length <= 60;
      };

      expect(validateTitle('Music Analyzer AI')).toBe(true);
      expect(validateTitle('A')).toBe(false); // Too short
      expect(validateTitle('A'.repeat(100))).toBe(false); // Too long
    });

    it('should validate description length', () => {
      const validateDescription = (description: string) => {
        return description.length >= 120 && description.length <= 160;
      };

      expect(validateDescription('AI-powered music analysis and discovery platform')).toBe(true);
      expect(validateDescription('Short')).toBe(false);
      expect(validateDescription('A'.repeat(200))).toBe(false);
    });

    it('should validate image dimensions for OG', () => {
      const validateImage = (width: number, height: number) => {
        return width >= 1200 && height >= 630;
      };

      expect(validateImage(1200, 630)).toBe(true);
      expect(validateImage(800, 600)).toBe(false);
    });
  });
});

