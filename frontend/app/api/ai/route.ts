// app/api/route.ts
import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  try {
    // Parse the request body to get question and conversation_id
    const { question, conversation_id } = await request.json();

    // Prepare the payload to send to the external API
    const payload = {
      question,
      conversation_id,
    };

    // Send a POST request to the external API
    const apiResponse = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    // Check if the response is OK
    if (!apiResponse.ok) {
      return NextResponse.json(
        { error: 'Failed to fetch data from external API' },
        { status: apiResponse.status }
      );
    }

    // Get the JSON data from the external API
    const data = await apiResponse.json();
    console.log("data: ", data);

    // Return the response from the external API
    return NextResponse.json(data);
  } catch (error) {
    // Handle any errors
    return NextResponse.json(
      { error: 'An error occurred while processing your request' },
      { status: 500 }
    );
  }
}
