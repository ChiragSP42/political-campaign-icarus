import { NextRequest, NextResponse } from "next/server";

export async function GET(req: NextRequest) {
  const accessToken = req.cookies.get("winflip_access_token")?.value;
  const email = req.cookies.get("winflip_email")?.value;

  if (!accessToken || !email) {
    return NextResponse.json({ authenticated: false }, { status: 401 });
  }

  try {
    // Validate the token with Cognito
    const { CognitoIdentityProviderClient, GetUserCommand } = await import("@aws-sdk/client-cognito-identity-provider");

    const client = new CognitoIdentityProviderClient({
      region: process.env.COGNITO_REGION || "us-east-1",
    });

    await client.send(new GetUserCommand({ AccessToken: accessToken }));

    // Token is valid — check if insights exist
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
    } catch {
      // If DDB check fails, still let them in — they're authenticated
    }

    return NextResponse.json({ authenticated: true, email, questionnaireCompleted: hasInsights });
  } catch (err: any) {
    // Token is expired or invalid — clear cookies
    const res = NextResponse.json({ authenticated: false }, { status: 401 });
    res.cookies.delete("winflip_access_token");
    res.cookies.delete("winflip_id_token");
    res.cookies.delete("winflip_email");
    return res;
  }
}
