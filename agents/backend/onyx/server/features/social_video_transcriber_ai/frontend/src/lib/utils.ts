import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatDuration(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);

  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }
  return `${minutes}:${secs.toString().padStart(2, '0')}`;
}

export function formatTimestamp(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

export function formatDate(dateString: string): string {
  const date = new Date(dateString);
  return new Intl.DateTimeFormat('es-ES', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date);
}

export function getStatusColor(status: string): string {
  const statusColors: Record<string, string> = {
    pending: 'bg-yellow-100 text-yellow-800',
    downloading: 'bg-blue-100 text-blue-800',
    extracting_audio: 'bg-purple-100 text-purple-800',
    transcribing: 'bg-indigo-100 text-indigo-800',
    analyzing: 'bg-pink-100 text-pink-800',
    completed: 'bg-green-100 text-green-800',
    failed: 'bg-red-100 text-red-800',
  };
  return statusColors[status] || 'bg-gray-100 text-gray-800';
}

export function getFrameworkName(framework: string): string {
  const frameworkNames: Record<string, string> = {
    hook_story_offer: 'Hook-Story-Offer',
    problem_agitate_solve: 'Problem-Agitate-Solve',
    aida: 'AIDA',
    star: 'STAR',
    bab: 'Before-After-Bridge',
    educational: 'Educational',
    storytelling: 'Storytelling',
    listicle: 'Listicle',
    tutorial: 'Tutorial',
    review: 'Review',
    news: 'News',
    entertainment: 'Entertainment',
    motivational: 'Motivational',
    custom: 'Custom',
  };
  return frameworkNames[framework] || framework;
}

export function validateVideoUrl(url: string): boolean {
  const patterns = [
    /^https?:\/\/(www\.)?(youtube\.com|youtu\.be)\/.+/i,
    /^https?:\/\/(www\.)?(tiktok\.com|vm\.tiktok\.com)\/.+/i,
    /^https?:\/\/(www\.)?instagram\.com\/(reel|p)\/.+/i,
  ];
  return patterns.some((pattern) => pattern.test(url));
}

export function extractVideoId(url: string): { platform: string; id: string } | null {
  // YouTube
  const youtubeMatch = url.match(/(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?#]+)/);
  if (youtubeMatch) {
    return { platform: 'youtube', id: youtubeMatch[1] };
  }

  // TikTok
  const tiktokMatch = url.match(/tiktok\.com\/@[\w.-]+\/video\/(\d+)/);
  if (tiktokMatch) {
    return { platform: 'tiktok', id: tiktokMatch[1] };
  }

  // Instagram
  const instagramMatch = url.match(/instagram\.com\/(?:reel|p)\/([^\/\?]+)/);
  if (instagramMatch) {
    return { platform: 'instagram', id: instagramMatch[1] };
  }

  return null;
}

export function downloadFile(content: string, filename: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

export function copyToClipboard(text: string): Promise<void> {
  return navigator.clipboard.writeText(text);
}




