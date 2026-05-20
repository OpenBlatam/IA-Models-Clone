'use client';

import React from 'react';
import { clsx } from 'clsx';

interface MarkdownProps {
  content: string;
  className?: string;
}

export const Markdown: React.FC<MarkdownProps> = ({ content, className }) => {
  const parseMarkdown = (text: string): React.ReactNode => {
    const lines = text.split('\n');
    const elements: React.ReactNode[] = [];
    let currentParagraph: string[] = [];
    let inCodeBlock = false;
    let codeBlockContent: string[] = [];
    let codeLanguage = '';

    const processParagraph = () => {
      if (currentParagraph.length > 0) {
        const paragraphText = currentParagraph.join(' ');
        elements.push(
          <p key={elements.length} className="mb-4 text-gray-700 dark:text-gray-300">
            {parseInlineMarkdown(paragraphText)}
          </p>
        );
        currentParagraph = [];
      }
    };

    lines.forEach((line, index) => {
      if (line.startsWith('```')) {
        if (inCodeBlock) {
          inCodeBlock = false;
          elements.push(
            <pre
              key={elements.length}
              className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg overflow-x-auto mb-4"
            >
              <code className={`language-${codeLanguage}`}>
                {codeBlockContent.join('\n')}
              </code>
            </pre>
          );
          codeBlockContent = [];
        } else {
          processParagraph();
          inCodeBlock = true;
          codeLanguage = line.slice(3).trim() || 'text';
        }
        return;
      }

      if (inCodeBlock) {
        codeBlockContent.push(line);
        return;
      }

      if (line.trim() === '') {
        processParagraph();
        return;
      }

      if (line.startsWith('# ')) {
        processParagraph();
        elements.push(
          <h1 key={elements.length} className="text-3xl font-bold mb-4 text-gray-900 dark:text-white">
            {line.slice(2)}
          </h1>
        );
        return;
      }

      if (line.startsWith('## ')) {
        processParagraph();
        elements.push(
          <h2 key={elements.length} className="text-2xl font-bold mb-3 text-gray-900 dark:text-white">
            {line.slice(3)}
          </h2>
        );
        return;
      }

      if (line.startsWith('### ')) {
        processParagraph();
        elements.push(
          <h3 key={elements.length} className="text-xl font-bold mb-2 text-gray-900 dark:text-white">
            {line.slice(4)}
          </h3>
        );
        return;
      }

      if (line.startsWith('- ') || line.startsWith('* ')) {
        processParagraph();
        elements.push(
          <li key={elements.length} className="ml-4 mb-1 text-gray-700 dark:text-gray-300">
            {parseInlineMarkdown(line.slice(2))}
          </li>
        );
        return;
      }

      currentParagraph.push(line);
    });

    processParagraph();

    return elements;
  };

  const parseInlineMarkdown = (text: string): React.ReactNode => {
    const parts: React.ReactNode[] = [];
    let currentIndex = 0;

    const patterns = [
      { regex: /\*\*(.+?)\*\*/g, render: (match: string) => <strong key={currentIndex++}>{match}</strong> },
      { regex: /\*(.+?)\*/g, render: (match: string) => <em key={currentIndex++}>{match}</em> },
      { regex: /`(.+?)`/g, render: (match: string) => <code key={currentIndex++} className="bg-gray-100 dark:bg-gray-800 px-1 rounded">{match}</code> },
      { regex: /\[(.+?)\]\((.+?)\)/g, render: (text: string, url: string) => <a key={currentIndex++} href={url} className="text-primary-600 dark:text-primary-400 hover:underline">{text}</a> },
    ];

    let processedText = text;
    patterns.forEach((pattern) => {
      processedText = processedText.replace(pattern.regex, (match, ...args) => {
        const rendered = pattern.render(...args);
        parts.push(rendered);
        return `__MARKDOWN_PLACEHOLDER_${currentIndex - 1}__`;
      });
    });

    const textParts = processedText.split(/__MARKDOWN_PLACEHOLDER_\d+__/);
    const result: React.ReactNode[] = [];

    textParts.forEach((part, index) => {
      if (part) {
        result.push(part);
      }
      if (index < parts.length) {
        result.push(parts[index]);
      }
    });

    return result.length > 0 ? result : text;
  };

  return (
    <div className={clsx('prose dark:prose-invert max-w-none', className)}>
      {parseMarkdown(content)}
    </div>
  );
};


