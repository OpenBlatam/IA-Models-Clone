import { NextRequest, NextResponse } from 'next/server';
import { auth } from '@/lib/auth';
import Stripe from 'stripe';

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY!, {
  apiVersion: '2024-12-18.acacia',
});

export const POST = async (request: NextRequest) => {
  const session = await auth();

  if (!session) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  try {
    const body = await request.json();
    const { invoiceId, amount, currency } = body;

    const checkoutSession = await stripe.checkout.sessions.create({
      payment_method_types: ['card'],
      line_items: [
        {
          price_data: {
            currency: currency || 'usd',
            product_data: {
              name: `Invoice ${invoiceId}`,
            },
            unit_amount: Math.round(amount * 100),
          },
          quantity: 1,
        },
      ],
      mode: 'payment',
      success_url: `${request.nextUrl.origin}/invoices?success=true`,
      cancel_url: `${request.nextUrl.origin}/invoices?canceled=true`,
      metadata: {
        invoiceId,
        userId: session.user.id,
      },
    });

    return NextResponse.json({ sessionId: checkoutSession.id });
  } catch (error: any) {
    console.error('Stripe error:', error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
};




