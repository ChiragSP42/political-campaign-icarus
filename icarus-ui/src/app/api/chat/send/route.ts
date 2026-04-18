import { NextRequest, NextResponse } from "next/server";

const API = process.env.API_ENDPOINT!;

export async function POST(req: NextRequest) {
  try {
    const { email, query, conversation_history, chatId } = await req.json();
    const res = await fetch(`${API}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, query, conversation_history, chatId }),
      signal: AbortSignal.timeout(100000),
    });
    const data = await res.json();
    return NextResponse.json(data);
  } catch (err: any) {
    return NextResponse.json({ status: "FAILED", message: err.message }, { status: 500 });
  }
}
