'use client'

import { useState, useEffect } from 'react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { PatientList } from "@/components/patients/patient-list"
import { AnalysisHistory } from "@/components/analysis/analysis-history"
import { AnalysisForm } from "@/components/analysis/analysis-form"
import { ThemeToggle } from "@/components/theme-toggle"
import { AddPatientDialog } from "@/components/patients/add-patient-dialog"
import { SettingsTab } from "@/components/settings/settings-tab"
import { Dashboard } from "@/components/dashboard/Dashboard"
import { ModelTraining } from "@/components/training/model-training"
import { MedicalReport } from "@/components/reports/medical-report"
import { type Patient, type Analysis, type Disease } from "@/types/medical"
import { toast } from "@/components/ui/use-toast"
import { getPatients, getPatientAnalyses, deletePatient, deleteAnalysis, runAnalysis } from "@/lib/api"
import { LogoutButton } from "@/components/auth/logout-button"
import { Button } from "@/components/ui/button"
import { Loader2 } from 'lucide-react'
import { ComparisonView } from "@/components/analysis/comparison-view"
import { useRouter } from 'next/navigation'

type AnalysisFormData = {
  type: 'brain_tumor' | 'cancer';
  patientId: string;
  image: File;
}

export default function Home() {
  const router = useRouter()
  const [patients, setPatients] = useState<Patient[]>([])
  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null)
  const [analyses, setAnalyses] = useState<Analysis[]>([])
  const [activeView, setActiveView] = useState<'dashboard' | 'list' | 'history' | 'new' | 'settings' | 'report' | 'training' | 'comparison'>('dashboard')
  const [isLoading, setIsLoading] = useState(true)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [diseases, setDiseases] = useState<Disease[]>([
    {
      id: "1",
      name: "Brain Tumor Detection",
      key: "brain_tumor",
      description: "Analysis for detecting brain tumors in MRI scans",
      is_active: true,
      created_at: new Date().toISOString()
    },
    {
      id: "2",
      name: "Cancer Detection",
      key: "cancer",
      description: "General cancer detection and analysis",
      is_active: true,
      created_at: new Date().toISOString()
    }
  ])

  useEffect(() => {
    fetchPatients()
  }, [])

  useEffect(() => {
    if (selectedPatient) {
      fetchPatientAnalyses(selectedPatient.id)
    }
  }, [selectedPatient])

  useEffect(() => {
    
  }, [router])

  async function fetchPatients() {
    try {
      setIsLoading(true)
      const data = await getPatients()
      setPatients(data)
    } catch (error) {
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to fetch patients",
      })
    } finally {
      setIsLoading(false)
    }
  }

  async function fetchPatientAnalyses(patientId: string) {
    try {
      const data = await getPatientAnalyses(patientId)
      setAnalyses(data)
    } catch (error) {
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to fetch analyses",
      })
    }
  }

  const handlePatientSelect = (patient: Patient) => {
    setSelectedPatient(patient)
    setActiveView('history')
  }

  const handleNewAnalysis = (patient: Patient) => {
    setSelectedPatient(patient)
    setActiveView('new')
  }

  const handleAnalysisSubmit = async (data: AnalysisFormData) => {
    try {
      setIsAnalyzing(true)
      const formData = new FormData()
      formData.append('image', data.image)
      formData.append('model_type', data.type)
      formData.append('patient_id', data.patientId)
    
      const analysis = await runAnalysis(formData)
    
      toast({
        title: "Analysis Complete",
        description: "The analysis has been completed successfully.",
      })
    
      if (selectedPatient) {
        await fetchPatientAnalyses(selectedPatient.id)
      }
      setActiveView('history')
    
    } catch (error) {
      console.error('Error:', error)
      toast({
        variant: "destructive",
        title: "Error",
        description: "An error occurred during analysis. Please try again.",
      })
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleDeletePatient = async (patientId: string) => {
    try {
      await deletePatient(patientId)
      toast({
        title: "Patient Deleted",
        description: "The patient has been successfully deleted.",
      })
      await fetchPatients()
      if (selectedPatient && selectedPatient.id === patientId) {
        setSelectedPatient(null)
        setActiveView('list')
      }
    } catch (error) {
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to delete patient",
      })
    }
  }

  const handleDeleteAnalysis = async (analysisId: string) => {
    try {
      await deleteAnalysis(analysisId)
      toast({
        title: "Analysis Deleted",
        description: "The analysis has been successfully deleted.",
      })
      if (selectedPatient) {
        await fetchPatientAnalyses(selectedPatient.id)
      }
    } catch (error) {
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to delete analysis",
      })
    }
  }

  const handleAddDisease = (disease: Omit<Disease, "id" | "is_active" | "created_at">) => {
    const newDisease: Disease = {
      ...disease,
      id: (diseases.length + 1).toString(),
      is_active: true,
      created_at: new Date().toISOString()
    }
    const updatedDiseases = [...diseases, newDisease]
    setDiseases(updatedDiseases)
    localStorage.setItem('diseases', JSON.stringify(updatedDiseases))
    toast({
      title: "Disease Added",
      description: `${disease.name} has been added successfully.`,
    })
  }

  return (
    <div className="min-h-screen">
      <header className="border-b bg-primary/10">
        <div className="container py-4 flex items-center justify-between">
          <div className="flex-1" />
          <h1 className="text-4xl font-bold text-center text-primary flex-1">Medical Analysis System</h1>
          <div className="flex items-center gap-4 flex-1 justify-end">
            <AddPatientDialog onPatientAdded={fetchPatients} />
            <LogoutButton />
            <ThemeToggle />
          </div>
        </div>
      </header>

      <main className="container py-8">
        <Tabs value={activeView} onValueChange={(value: string) => setActiveView(value as any)}>
          <div className="flex items-center justify-between mb-8">
            <TabsList>
              <TabsTrigger value="dashboard">Dashboard</TabsTrigger>
              <TabsTrigger value="list">Patient List</TabsTrigger>
              {selectedPatient && (
                <>
                  <TabsTrigger value="history">Analysis History</TabsTrigger>
                  <TabsTrigger value="new">New Analysis</TabsTrigger>
                  {analyses.length > 0 && (
                    <>
                      <TabsTrigger value="report">Medical Report</TabsTrigger>
                      <TabsTrigger value="comparison">Comparison</TabsTrigger>
                    </>
                  )}
                </>
              )}
              <TabsTrigger value="training">Model Training</TabsTrigger>
              <TabsTrigger value="settings">Settings</TabsTrigger>
            </TabsList>
          </div>

          <TabsContent value="dashboard" className="m-0">
            <Dashboard />
          </TabsContent>

          <TabsContent value="list" className="m-0">
            <PatientList
              patients={patients}
              onViewPatient={handlePatientSelect}
              onNewAnalysis={handleNewAnalysis}
              onDeletePatient={handleDeletePatient}
              isLoading={isLoading}
            />
          </TabsContent>

          <TabsContent value="history" className="m-0">
            {selectedPatient && (
              <AnalysisHistory
                patient={selectedPatient}
                analyses={analyses}
                onDeleteAnalysis={handleDeleteAnalysis}
              />
            )}
          </TabsContent>

          <TabsContent value="new" className="m-0">
            {selectedPatient && (
              <>
                <AnalysisForm
                  patient={selectedPatient}
                  onSubmit={handleAnalysisSubmit}
                  diseases={diseases}
                />
                {isAnalyzing && (
                  <div className="flex justify-center items-center mt-4">
                    <Button disabled>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Analyzing...
                    </Button>
                  </div>
                )}
              </>
            )}
          </TabsContent>

          <TabsContent value="training" className="m-0">
            <ModelTraining />
          </TabsContent>

          <TabsContent value="settings" className="m-0">
            <SettingsTab 
              diseases={diseases}
              onAddDisease={handleAddDisease}
            />
          </TabsContent>

          <TabsContent value="report" className="m-0">
            {selectedPatient && analyses.length > 0 && (
              <MedicalReport
                patient={selectedPatient}
                analyses={analyses}
              />
            )}
          </TabsContent>

         
        </Tabs>
      </main>
    </div>
  )
}

