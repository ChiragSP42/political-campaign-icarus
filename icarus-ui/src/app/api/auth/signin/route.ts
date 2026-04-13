import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  try {
    const { email, password } = await req.json();

    const { CognitoIdentityProviderClient, InitiateAuthCommand } = await import("@aws-sdk/client-cognito-identity-provider");

    const client = new CognitoIdentityProviderClient({
      region: process.env.COGNITO_REGION || "us-east-1",
      credentials: {
        accessKeyId: process.env.CUSTOM_ACCESS_KEY_ID!,
        secretAccessKey: process.env.CUSTOM_SECRET_ACCESS_KEY!,
      },
    });

    const authResult = await client.send(new InitiateAuthCommand({
      ClientId: process.env.COGNITO_CLIENT_ID!,
      AuthFlow: "USER_PASSWORD_AUTH",
      AuthParameters: { USERNAME: email, PASSWORD: password },
    }));

    const accessToken = authResult.AuthenticationResult?.AccessToken;
    const idToken = authResult.AuthenticationResult?.IdToken;
    const expiresIn = authResult.AuthenticationResult?.ExpiresIn || 3600;

    if (!accessToken || !idToken) {
      return NextResponse.json({ success: false, message: "Authentication failed" }, { status: 401 });
    }

    // Check if insights already exist for this user
    const { DynamoDBClient } = await import("@aws-sdk/client-dynamodb");
    const { DynamoDBDocumentClient, GetCommand } = await import("@aws-sdk/lib-dynamodb");

    const ddbClient = new DynamoDBClient({
      region: process.env.CUSTOM_REGION || "us-east-1",
      credentials: {
        accessKeyId: process.env.CUSTOM_ACCESS_KEY_ID!,
        secretAccessKey: process.env.CUSTOM_SECRET_ACCESS_KEY!,
      },
    });
    const docClient = DynamoDBDocumentClient.from(ddbClient);

    let hasInsights = false;
    try {
      const result = await docClient.send(new GetCommand({
        TableName: process.env.MAIN_TABLE_NAME!,
        Key: { userId: `USER#${email}`, SK: "INSIGHTS" },
      }));
      hasInsights = !!(result.Item && result.Item.insights);
    } catch (e) {
      console.error("Error checking insights:", e);
    }

    // Set tokens in httpOnly cookies
    const res = NextResponse.json({ success: true, email, questionnaireCompleted: hasInsights });

    res.cookies.set("icarus_access_token", accessToken, {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "strict",
      path: "/",
      maxAge: expiresIn,
    });

    res.cookies.set("icarus_id_token", idToken, {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "strict",
      path: "/",
      maxAge: expiresIn,
    });

    // Store email in a separate cookie for session lookup (not sensitive)
    res.cookies.set("icarus_email", email, {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "strict",
      path: "/",
      maxAge: expiresIn,
    });

    return res;
  } catch (err: any) {
    const code = err?.name || err?.__type || "";
    const messages: Record<string, string> = {
      NotAuthorizedException: "Invalid email or password.",
      UserNotFoundException: "User not found. Please sign up first.",
      UserNotConfirmedException: "Please confirm your email first.",
    };
    return NextResponse.json({ success: false, message: messages[code] || `Sign in failed: ${err.message}` }, { status: 401 });
  }
}
