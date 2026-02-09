'use client';

import { motion } from 'framer-motion';
import { PageLayout, AnnouncementBanner, PricingPlanCard } from '../components/pages';
import { PRICING_PLANS } from '../components/pages/data/pricing-plans';

export default function PricingPage() {
  return (
    <PageLayout>
      {/* Hero Section */}
      <section className="mb-16 md:mb-20 text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <h1 className="text-5xl md:text-6xl lg:text-7xl xl:text-8xl font-bold text-black mb-8 md:mb-12 leading-[1.08] tracking-[-0.03em] font-sans antialiased max-w-5xl mx-auto">
            Choose the perfect plan for your journey
          </h1>
        </motion.div>
      </section>

      {/* Announcement Banner */}
      <AnnouncementBanner text="bulk is now available!" />

      {/* Pricing Plans */}
      <div className="grid md:grid-cols-3 gap-8 md:gap-12">
        {PRICING_PLANS.map((plan, index) => (
          <PricingPlanCard
            key={plan.id}
            id={plan.id}
            badge={plan.badge}
            title={plan.title}
            price={plan.price}
            description={plan.description}
            features={plan.features}
            buttonText={plan.buttonText}
            buttonAriaLabel={plan.buttonAriaLabel}
            comingSoon={plan.comingSoon}
            index={index}
          />
        ))}
      </div>
    </PageLayout>
  );
}

