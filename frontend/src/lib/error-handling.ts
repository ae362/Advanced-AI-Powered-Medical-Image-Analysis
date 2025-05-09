import { toast } from "@/components/ui/use-toast"

/**
 * Custom error class for application-specific errors
 */
export class AppError extends Error {
  constructor(message: string, public code: string) {
    super(message)
    this.name = 'AppError'
  }
}

/**
 * Global error handler that provides consistent error handling across the application
 */
export function handleError(error: unknown) {
  console.error('An error occurred:', error)

  if (error instanceof AppError) {
    if (error.code === 'ENDPOINT_NOT_FOUND') {
      toast({
        variant: "destructive",
        title: "API Configuration Error",
        description: "Could not connect to the training endpoint. Please check your API configuration.",
      })
      return
    }
    toast({
      variant: "destructive",
      title: "Error",
      description: error.message,
    })
  } else if (error instanceof SyntaxError) {
    toast({
      variant: "destructive",
      title: "Invalid Response",
      description: "Received invalid response from server. Please check API configuration.",
    })
  } else if (error instanceof Error) {
    toast({
      variant: "destructive",
      title: "Unexpected Error",
      description: "An unexpected error occurred. Please try again.",
    })
  } else {
    toast({
      variant: "destructive",
      title: "Unknown Error",
      description: "An unknown error occurred. Please try again.",
    })
  }
}

/**
 * Utility function to retry failed operations
 */
export async function retryOperation<T>(
  operation: () => Promise<T>,
  maxRetries: number = 3,
  delay: number = 1000
): Promise<T> {
  let lastError: unknown

  for (let i = 0; i < maxRetries; i++) {
    try {
      return await operation()
    } catch (error) {
      lastError = error
      await new Promise(resolve => setTimeout(resolve, delay))
    }
  }

  throw lastError
}

