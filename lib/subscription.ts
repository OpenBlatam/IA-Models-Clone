import { pricingData } from "@/config/subscriptions";
import { prisma } from "@/lib/db";
import { stripe } from "@/lib/stripe";
import { UserSubscriptionPlan } from "types";

export async function getUserSubscriptionPlan(
  userId: string
): Promise<UserSubscriptionPlan> {
  if (!userId) {
    return {
      ...pricingData[0],
      stripeCustomerId: null,
      stripeSubscriptionId: null,
      stripePriceId: null,
      stripeCurrentPeriodEnd: 0,
      isPaid: false,
      interval: null,
      isCanceled: false
    };
  }

  const user = await prisma.user.findFirst({
    where: {
      id: userId,
    },
    select: {
      stripeSubscriptionId: true,
      stripeCurrentPeriodEnd: true,
      stripeCustomerId: true,
      stripePriceId: true,
    },
  });

  if (!user) {
    return {
      ...pricingData[0],
      stripeCustomerId: null,
      stripeSubscriptionId: null,
      stripePriceId: null,
      stripeCurrentPeriodEnd: 0,
      isPaid: false,
      interval: null,
      isCanceled: false
    };
  }

  // Check if user is on a paid plan.
  const isPaid =
    user.stripePriceId &&
    user.stripeCurrentPeriodEnd &&
    user.stripeCurrentPeriodEnd.getTime() + 86_400_000 > Date.now() ? true : false;

  // Find the pricing data corresponding to the user's plan
  const userPlan =
    pricingData.find((plan) => plan.stripeIds.monthly === user.stripePriceId) ||
    pricingData.find((plan) => plan.stripeIds.yearly === user.stripePriceId);

  const plan = isPaid && userPlan ? userPlan : pricingData[0];

  const interval = isPaid
    ? userPlan?.stripeIds.monthly === user.stripePriceId
      ? "month"
      : userPlan?.stripeIds.yearly === user.stripePriceId
      ? "year"
      : null
    : null;

  let isCanceled = false;
  if (isPaid && user.stripeSubscriptionId) {
    const stripePlan = await stripe.subscriptions.retrieve(
      user.stripeSubscriptionId
    );
    isCanceled = stripePlan.cancel_at_period_end;
  }

  return {
    ...plan,
    ...user,
    stripeCurrentPeriodEnd: user.stripeCurrentPeriodEnd?.getTime() || 0,
    isPaid,
    interval,
    isCanceled,
  };
}
