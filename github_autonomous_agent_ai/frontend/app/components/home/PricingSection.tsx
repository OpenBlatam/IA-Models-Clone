'use client';

interface PricingTier {
  id: string;
  badge: string;
  title: string;
  description: string;
  buttonText: string;
  buttonAriaLabel: string;
  buttonClassName: string;
}

const PRICING_TIERS: PricingTier[] = [
  {
    id: 'developers',
    badge: 'Available at no charge',
    title: 'For developer',
    description: 'Achieve new height',
    buttonText: 'Download',
    buttonAriaLabel: 'Download for developers',
    buttonClassName: 'bg-black text-white px-8 py-4 rounded-lg hover:bg-[#1a1a1a] transition-colors duration-200 ease-in-out font-normal text-base leading-normal focus:outline-none focus:ring-2 focus:ring-black focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed',
  },
  {
    id: 'organizations',
    badge: 'Coming soon',
    title: 'For organizations',
    description: 'Enterprise agent orchestration at scale',
    buttonText: 'Notify me',
    buttonAriaLabel: 'Notify me when enterprise version is available',
    buttonClassName: 'bg-white border border-[#000000] text-black px-8 py-4 rounded-lg hover:bg-[#f5f5f5] hover:border-[#000000] transition-colors duration-200 ease-in-out font-normal text-base leading-normal focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed',
  },
];

export function PricingSection() {
  return (
    <section 
      className="py-16 md:py-20 lg:py-24 border-b border-gray-200 relative z-10"
      aria-labelledby="pricing-heading"
    >
      <div className="max-w-[1920px] mx-auto px-4 md:px-6 lg:px-8">
        <h2 id="pricing-heading" className="sr-only">
          Pricing Plans
        </h2>
        <div className="grid md:grid-cols-2 gap-12 md:gap-16">
          {PRICING_TIERS.map((tier) => (
            <article key={tier.id} className="space-y-6">
              <div className="text-gray-500 text-xs uppercase tracking-wide font-medium leading-normal">{tier.badge}</div>
              <h3 className="text-black text-3xl font-normal leading-[1.2] tracking-[-0.02em]">{tier.title}</h3>
              <p className="text-black text-lg font-normal leading-[1.5] tracking-[-0.01em]">{tier.description}</p>
              <button 
                className={tier.buttonClassName}
                aria-label={tier.buttonAriaLabel}
                type="button"
              >
                {tier.buttonText}
              </button>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}

