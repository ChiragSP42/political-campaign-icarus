import { NextRequest, NextResponse } from "next/server";

const API = process.env.API_ENDPOINT!;

export async function POST(req: NextRequest) {
  try {
    const { email, answers } = await req.json();
    const res = await fetch(`${API}/save-questionnaire`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, answers }),
      signal: AbortSignal.timeout(60000),
    });
    const data = await res.json();
    return NextResponse.json(data);
  } catch (err: any) {
    return NextResponse.json({ success: false, message: err.message }, { status: 500 });
  }
}
