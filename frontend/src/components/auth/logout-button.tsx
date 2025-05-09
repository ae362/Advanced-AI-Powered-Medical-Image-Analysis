'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { Button } from "@/components/ui/button"
import { toast } from "@/components/ui/use-toast"
import { Loader2 } from 'lucide-react'
import { removeTokens, getRefreshToken } from '@/lib/auth'

export function LogoutButton() {
  const [isLoading, setIsLoading] = useState(false)
  const router = useRouter()

  const handleLogout = async () => {
    setIsLoading(true)
    try {
      const response = await fetch('http://localhost:8000/api/logout/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('accessToken')}`,
        },
      })

      if (response.ok) {
        localStorage.removeItem('accessToken')
        localStorage.removeItem('refreshToken')
        toast({
          title: "Logged out",
          description: "You have been successfully logged out.",
        })
        router.push('/login')
      } else {
        throw new Error('Logout failed')
      }
    } catch (error) {
      console.error('Logout error:', error)
      // Even if the server-side logout fails, clear local storage
      localStorage.removeItem('accessToken')
      localStorage.removeItem('refreshToken')
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to log out properly, but your session has been cleared.",
      })
      router.push('/login')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <Button onClick={handleLogout} disabled={isLoading}>
      {isLoading ? (
        <>
          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          Logging out...
        </>
      ) : (
        'Logout'
      )}
    </Button>
  )
}

