'use client';

import { motion } from 'framer-motion';

interface PricingPlanCardProps {
  id: string;
  badge: string;
  title: string;
  price?: string;
  description: string;
  features?: string[];
  buttonText: string;
  buttonAriaLabel: string;
  comingSoon?: boolean;
  index?: number;
}

export function PricingPlanCard({
  badge,
  title,
  price,
  description,
  features,
  buttonText,
  buttonAriaLabel,
  comingSoon = false,
  index = 0,
}: PricingPlanCardProps) {
  const buttonClassName = comingSoon
    ? 'bg-white border border-[#000000] text-black px-6 py-3 rounded-lg hover:bg-[#f5f5f5] hover:border-[#000000] transition-colors font-normal text-base focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2'
    : 'bg-black text-white px-6 py-3 rounded-lg hover:bg-[#1a1a1a] transition-colors font-normal text-base focus:outline-none focus:ring-2 focus:ring-black focus:ring-offset-2';

  return (
    <motion.article
      className="space-y-6"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: index * 0.1 }}
    >
      {/* Badge */}
      <div className="mb-4">
        <span className="text-xs font-normal text-gray-500 uppercase tracking-wide">
          {badge}
        </span>
      </div>

      {/* Title */}
      <h2 className="text-3xl md:text-4xl font-normal text-black font-sans antialiased">
        {title}
        {price && (
          <span className="block text-2xl md:text-3xl mt-2 text-black font-normal font-sans antialiased">
            {price}
          </span>
        )}
      </h2>

      {/* Description */}
      <p className="text-base md:text-lg text-black font-normal font-sans antialiased">
        {description}
      </p>

      {/* Button */}
      <div>
        <button
          className={buttonClassName}
          aria-label={buttonAriaLabel}
          type="button"
          disabled={comingSoon}
        >
          {buttonText}
        </button>
      </div>

      {/* Features List */}
      {features && (
        <ul className="space-y-3 mt-6">
          {features.map((feature, featureIndex) => (
            <li key={featureIndex} className="flex items-start gap-3 text-sm text-black">
              <svg
                className="w-5 h-5 flex-shrink-0 mt-0.5 text-black"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M5 13l4 4L19 7"
                />
              </svg>
              <span>{feature}</span>
            </li>
          ))}
        </ul>
      )}
    </motion.article>
  );
}

