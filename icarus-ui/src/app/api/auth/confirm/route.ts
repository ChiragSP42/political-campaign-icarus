import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  try {
    const { email, code } = await req.json();
    const { CognitoIdentityProviderClient, ConfirmSignUpCommand } = await import("@aws-sdk/client-cognito-identity-provider");

    const client = new CognitoIdentityProviderClient({
      region: process.env.COGNITO_REGION || "us-east-1",
      credentials: {
        accessKeyId: process.env.CUSTOM_ACCESS_KEY_ID!,
        secretAccessKey: process.env.CUSTOM_SECRET_ACCESS_KEY!,
      },
    });

    await client.send(new ConfirmSignUpCommand({
      ClientId: process.env.COGNITO_CLIENT_ID!,
      Username: email,
      ConfirmationCode: code,
    }));

    return NextResponse.json({ success: true, message: "Email confirmed! You can now sign in." });
  } catch (err: any) {
    const code = err?.name || "";
    const messages: Record<string, string> = {
      CodeMismatchException: "Invalid verification code.",
      UserNotFoundException: "User not found.",
    };
    return NextResponse.json({ success: false, message: messages[code] || err.message }, { status: 400 });
  }
}
