'use client';

interface CategoryTabsProps {
  categories: string[];
  selectedCategory: string;
  onCategoryChange: (category: string) => void;
}

export function CategoryTabs({ categories, selectedCategory, onCategoryChange }: CategoryTabsProps) {
  return (
    <div
      role="tablist"
      className="flex gap-0 border-b border-gray-700"
      aria-label="Blog category filters"
    >
      {categories.map((category) => (
        <button
          key={category}
          role="tab"
          aria-selected={selectedCategory === category}
          aria-controls={`tabpanel-${category.toLowerCase()}`}
          onClick={() => onCategoryChange(category)}
          className={`
            px-4 py-2 text-sm font-normal transition-colors relative
            focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 focus:ring-offset-black
            ${
              selectedCategory === category
                ? 'text-white'
                : 'text-gray-400 hover:text-white'
            }
          `}
        >
          {category}
          {selectedCategory === category && (
            <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-white" />
          )}
        </button>
      ))}
    </div>
  );
}

