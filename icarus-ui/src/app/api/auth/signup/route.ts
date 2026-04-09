import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  try {
    const { email, password } = await req.json();
    const { CognitoIdentityProviderClient, SignUpCommand } = await import("@aws-sdk/client-cognito-identity-provider");

    const client = new CognitoIdentityProviderClient({
      region: process.env.COGNITO_REGION || "us-east-1",
      credentials: {
        accessKeyId: process.env.CUSTOM_ACCESS_KEY_ID!,
        secretAccessKey: process.env.CUSTOM_SECRET_ACCESS_KEY!,
      },
    });

    await client.send(new SignUpCommand({
      ClientId: process.env.COGNITO_CLIENT_ID!,
      Username: email,
      Password: password,
      UserAttributes: [{ Name: "email", Value: email }],
    }));

    return NextResponse.json({ success: true, message: "Check your email for a verification code." });
  } catch (err: any) {
    const code = err?.name || "";
    const messages: Record<string, string> = {
      UsernameExistsException: "Email already registered. Please sign in.",
      InvalidPasswordException: "Password must be 8+ characters with uppercase, lowercase, and numbers.",
      InvalidParameterException: "Invalid email format.",
    };
    return NextResponse.json({ success: false, message: messages[code] || err.message }, { status: 400 });
  }
}
