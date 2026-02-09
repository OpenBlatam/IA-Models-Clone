'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { PageLayout, RoleSelector } from '../components/pages';
import { USE_CASES, ROLES } from '../components/pages/data/use-cases';

export default function UseCasesPage() {
  const [selectedRole, setSelectedRole] = useState<string>('professional');

  const currentUseCase = USE_CASES.find((uc) => uc.id === selectedRole) || USE_CASES[0];

  return (
    <PageLayout>
      {/* Use Cases Section */}
      <section className="mb-16 md:mb-20">
        <div className="max-w-4xl mx-auto text-center mb-16">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="space-y-6"
          >
            <h1 className="text-5xl md:text-6xl lg:text-7xl xl:text-8xl font-bold text-black leading-[1.08] tracking-[-0.03em] font-sans antialiased">
              {currentUseCase.title}
            </h1>
            <p className="text-lg md:text-xl text-black leading-relaxed font-normal font-sans antialiased max-w-3xl mx-auto">
              {currentUseCase.description}
            </p>
            <a
              href="#"
              className="inline-block text-black hover:opacity-70 underline transition-opacity text-base focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 rounded"
              aria-label={`Explore ${currentUseCase.title} use case`}
            >
              Explore use case
            </a>
          </motion.div>
        </div>

        {/* Role Selection */}
        <RoleSelector
          roles={ROLES}
          selectedRole={selectedRole}
          onRoleChange={setSelectedRole}
        />
      </section>
    </PageLayout>
  );
}

