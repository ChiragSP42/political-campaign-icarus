import { NextRequest, NextResponse } from "next/server";

const API = process.env.API_ENDPOINT!;

export async function GET(req: NextRequest) {
  try {
    const email = req.nextUrl.searchParams.get("email")!;
    const res = await fetch(`${API}/check-response?email=${encodeURIComponent(email)}`, {
      headers: { "Content-Type": "application/json" },
      signal: AbortSignal.timeout(100000),
    });
    const data = await res.json();
    return NextResponse.json(data);
  } catch (err: any) {
    return NextResponse.json({ status: "FAILED", message: err.message }, { status: 500 });
  }
}
