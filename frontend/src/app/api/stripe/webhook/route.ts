import { NextRequest, NextResponse } from "next/server";
import Stripe from "stripe";

function getStripe(): Stripe | null {
  const key = process.env.STRIPE_SECRET_KEY?.trim();
  if (!key) return null;
  return new Stripe(key);
}

async function upgradeUserToPro(
  userId: string,
  stripeCustomerId: string | null,
): Promise<void> {
  const supabaseUrl = process.env.SUPABASE_URL?.trim();
  const serviceKey = process.env.SUPABASE_SERVICE_ROLE_KEY?.trim();
  const proLimit = Number(process.env.PRO_TIER_LLM_CALLS_PER_DAY || "100");

  if (!supabaseUrl || !serviceKey) {
    throw new Error("Supabase service credentials are not configured.");
  }

  const payload: Record<string, unknown> = {
    id: userId,
    plan: "pro",
    llm_calls_used: 0,
    llm_calls_limit: proLimit,
    daily_tokens_used: 0,
  };
  if (stripeCustomerId) {
    payload.stripe_customer_id = stripeCustomerId;
  }

  const res = await fetch(`${supabaseUrl}/rest/v1/user_profiles`, {
    method: "POST",
    headers: {
      apikey: serviceKey,
      Authorization: `Bearer ${serviceKey}`,
      "Content-Type": "application/json",
      Prefer: "resolution=merge-duplicates,return=minimal",
    },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Failed to update user profile: ${text}`);
  }
}

async function downgradeUserToFree(userId: string): Promise<void> {
  const supabaseUrl = process.env.SUPABASE_URL?.trim();
  const serviceKey = process.env.SUPABASE_SERVICE_ROLE_KEY?.trim();
  const freeLimit = Number(process.env.FREE_TIER_LLM_CALLS || "3");

  if (!supabaseUrl || !serviceKey) return;

  await fetch(`${supabaseUrl}/rest/v1/user_profiles?id=eq.${userId}`, {
    method: "PATCH",
    headers: {
      apikey: serviceKey,
      Authorization: `Bearer ${serviceKey}`,
      "Content-Type": "application/json",
      Prefer: "return=minimal",
    },
    body: JSON.stringify({
      plan: "free",
      llm_calls_limit: freeLimit,
      llm_calls_used: 0,
      daily_tokens_used: 0,
    }),
  });
}

export async function POST(request: NextRequest) {
  const stripe = getStripe();
  const webhookSecret = process.env.STRIPE_WEBHOOK_SECRET?.trim();

  if (!stripe || !webhookSecret) {
    return NextResponse.json({ error: "Webhook not configured." }, { status: 503 });
  }

  const signature = request.headers.get("stripe-signature");
  if (!signature) {
    return NextResponse.json({ error: "Missing signature." }, { status: 400 });
  }

  const body = await request.text();
  let event: Stripe.Event;

  try {
    event = stripe.webhooks.constructEvent(body, signature, webhookSecret);
  } catch (err) {
    const message = err instanceof Error ? err.message : "Invalid signature";
    return NextResponse.json({ error: message }, { status: 400 });
  }

  try {
    switch (event.type) {
      case "checkout.session.completed": {
        const session = event.data.object as Stripe.Checkout.Session;
        // Only upgrade on confirmed payment — ignore sessions with pending/unpaid status.
        if (session.payment_status !== "paid") break;
        const userId =
          session.metadata?.supabase_user_id || session.client_reference_id;
        if (userId) {
          const customerId =
            typeof session.customer === "string" ? session.customer : null;
          await upgradeUserToPro(userId, customerId);
        }
        break;
      }
      case "customer.subscription.deleted": {
        const subscription = event.data.object as Stripe.Subscription;
        const userId = subscription.metadata?.supabase_user_id;
        if (userId) {
          await downgradeUserToFree(userId);
        }
        break;
      }
      default:
        break;
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : "Webhook handler failed";
    return NextResponse.json({ error: message }, { status: 500 });
  }

  return NextResponse.json({ received: true });
}
