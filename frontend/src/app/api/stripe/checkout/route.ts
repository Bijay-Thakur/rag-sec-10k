import { createClient } from "@supabase/supabase-js";
import { NextRequest, NextResponse } from "next/server";
import Stripe from "stripe";

function getStripe(): Stripe | null {
  const key = process.env.STRIPE_SECRET_KEY?.trim();
  if (!key) return null;
  return new Stripe(key);
}

async function verifySupabaseUser(
  authHeader: string | null,
): Promise<{ userId: string; email: string } | null> {
  const url = process.env.SUPABASE_URL?.trim();
  const anonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY?.trim();
  if (!authHeader?.startsWith("Bearer ") || !url || !anonKey) {
    return null;
  }
  const token = authHeader.slice(7);
  const supabase = createClient(url, anonKey);
  const { data, error } = await supabase.auth.getUser(token);
  if (error || !data.user) {
    return null;
  }
  return {
    userId: data.user.id,
    email: data.user.email || "",
  };
}

export async function POST(request: NextRequest) {
  const stripe = getStripe();
  const priceId = process.env.STRIPE_PRICE_ID?.trim();
  const siteUrl = process.env.NEXT_PUBLIC_SITE_URL?.trim() || "http://localhost:3000";

  if (!stripe || !priceId) {
    return NextResponse.json(
      { error: "Stripe is not configured on this deployment." },
      { status: 503 },
    );
  }

  const user = await verifySupabaseUser(request.headers.get("authorization"));
  if (!user) {
    return NextResponse.json({ error: "Authentication required." }, { status: 401 });
  }

  try {
    const session = await stripe.checkout.sessions.create({
      mode: "subscription",
      line_items: [{ price: priceId, quantity: 1 }],
      success_url: `${siteUrl}/pricing?success=1`,
      cancel_url: `${siteUrl}/pricing?canceled=1`,
      customer_email: user.email || undefined,
      client_reference_id: user.userId,
      metadata: { supabase_user_id: user.userId },
      subscription_data: {
        metadata: { supabase_user_id: user.userId },
      },
    });

    return NextResponse.json({ url: session.url });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Checkout failed";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
