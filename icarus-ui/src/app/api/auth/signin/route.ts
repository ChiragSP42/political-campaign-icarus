import { NextRequest, NextResponse } from "next/server";

const API = process.env.API_ENDPOINT!;

export async function POST(req: NextRequest) {
  try {
    const { email, password } = await req.json();

    // Use your existing Cognito auth via the API Gateway or direct SDK
    // For now, proxy to your existing Lambda-backed endpoint if you have one,
    // or call Cognito directly. Below is a direct Cognito call:
    const { CognitoIdentityProviderClient, InitiateAuthCommand } = await import("@aws-sdk/client-cognito-identity-provider");

    const client = new CognitoIdentityProviderClient({
      region: process.env.COGNITO_REGION || "us-east-1",
      credentials: {
        accessKeyId: process.env.CUSTOM_ACCESS_KEY_ID!,
        secretAccessKey: process.env.CUSTOM_SECRET_ACCESS_KEY!,
      },
    });

    await client.send(new InitiateAuthCommand({
      ClientId: process.env.COGNITO_CLIENT_ID!,
      AuthFlow: "USER_PASSWORD_AUTH",
      AuthParameters: { USERNAME: email, PASSWORD: password },
    }));

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

    return NextResponse.json({ success: true, email, questionnaireCompleted: hasInsights });
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
