import { useEffect } from 'react';

interface MetaTag {
  name?: string;
  property?: string;
  content: string;
}

export const useMetaTags = (tags: MetaTag[]): void => {
  useEffect(() => {
    const elements: HTMLMetaElement[] = [];

    tags.forEach((tag) => {
      const selector = tag.name
        ? `meta[name="${tag.name}"]`
        : `meta[property="${tag.property}"]`;

      let element = document.querySelector(selector) as HTMLMetaElement;

      if (!element) {
        element = document.createElement('meta');
        if (tag.name) {
          element.setAttribute('name', tag.name);
        }
        if (tag.property) {
          element.setAttribute('property', tag.property);
        }
        document.head.appendChild(element);
      }

      element.setAttribute('content', tag.content);
      elements.push(element);
    });

    return () => {
      elements.forEach((element) => {
        if (element.parentNode) {
          element.parentNode.removeChild(element);
        }
      });
    };
  }, [tags]);
};



