import { NextRequest, NextResponse } from "next/server";

const API = process.env.API_ENDPOINT!;

export async function GET(req: NextRequest) {
  try {
    const email = req.nextUrl.searchParams.get("email")!;
    const res = await fetch(
      `${API}/sessions?userId=${encodeURIComponent(email)}`,
      {
        headers: { "Content-Type": "application/json" },
        signal: AbortSignal.timeout(100000),
      }
    );
    const data = await res.json();
    return NextResponse.json(data);
  } catch (err: any) {
    return NextResponse.json(
      { status: "FAILED", message: err.message },
      { status: 500 }
    );
  }
}

export async function DELETE(req: NextRequest) {
  try {
    const chatId = req.nextUrl.searchParams.get("chatId")!;
    const email = req.nextUrl.searchParams.get("email")!;
    const res = await fetch(
      `${API}/sessions?chatId=${encodeURIComponent(chatId)}&userId=${encodeURIComponent(email)}`,
      {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        signal: AbortSignal.timeout(100000),
      }
    );
    const data = await res.json();
    return NextResponse.json(data);
  } catch (err: any) {
    return NextResponse.json(
      { status: "FAILED", message: err.message },
      { status: 500 }
    );
  }
}
