'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Users, FileText } from 'lucide-react'

async function fetchDashboardData() {
  const patientsResponse = await fetch('http://localhost:8000/api/patients/')
  const analysesResponse = await fetch('http://localhost:8000/api/analyses/')
  
  const patients = await patientsResponse.json()
  const analyses = await analysesResponse.json()

  return {
    patientCount: patients.length,
    analysisCount: analyses.length
  }
}

export function Dashboard() {
  const [dashboardData, setDashboardData] = useState({ patientCount: 0, analysisCount: 0 })

  useEffect(() => {
    fetchDashboardData().then(setDashboardData)
  }, [])

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">
            Total Patients
          </CardTitle>
          <Users className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{dashboardData.patientCount}</div>
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">
            Total Analyses
          </CardTitle>
          <FileText className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{dashboardData.analysisCount}</div>
        </CardContent>
      </Card>
    </div>
  )
}

