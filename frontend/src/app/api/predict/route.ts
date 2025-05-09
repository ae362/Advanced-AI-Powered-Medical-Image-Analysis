import { NextResponse } from 'next/server'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api'

export async function POST(request: Request) {
  try {
    const formData = await request.formData()
    
    const response = await fetch(`${API_URL}/analyses/predict/`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      const errorText = await response.text()
      console.error('API Error:', errorText)
      return NextResponse.json(
        { error: 'Analysis failed' },
        { status: response.status }
      )
    }
    
    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error('Analysis error:', error)
    return NextResponse.json(
      { error: 'An error occurred during analysis' },
      { status: 500 }
    )
  }
}

