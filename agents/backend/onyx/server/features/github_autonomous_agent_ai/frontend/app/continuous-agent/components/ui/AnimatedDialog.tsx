"use client";

import React from "react";
import * as Dialog from "@radix-ui/react-dialog";
import { motion, AnimatePresence } from "framer-motion";
import type { ReactNode } from "react";
import { modalAnimations } from "../../constants/animations";

type AnimatedDialogProps = {
  readonly open: boolean;
  readonly onOpenChange: (open: boolean) => void;
  readonly children: ReactNode;
  readonly className?: string;
};

/**
 * Animated Dialog wrapper component
 * 
 * Provides:
 * - Radix UI Dialog with accessibility
 * - Framer Motion animations
 * - Consistent styling
 */
export const AnimatedDialog = ({
  open,
  onOpenChange,
  children,
  className = "fixed left-[50%] top-[50%] z-50 grid w-full max-w-2xl translate-x-[-50%] translate-y-[-50%] gap-4 border bg-background p-6 shadow-lg duration-200 rounded-lg max-h-[90vh] overflow-y-auto",
}: AnimatedDialogProps): JSX.Element => {
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <AnimatePresence>
        {open && (
          <Dialog.Portal forceMount>
            <Dialog.Overlay asChild>
              <motion.div
                className="fixed inset-0 bg-black/50 z-50"
                {...modalAnimations.overlay}
              />
            </Dialog.Overlay>
            <Dialog.Content asChild className={className}>
              <motion.div {...modalAnimations.content}>
                {children}
              </motion.div>
            </Dialog.Content>
          </Dialog.Portal>
        )}
      </AnimatePresence>
    </Dialog.Root>
  );
};

