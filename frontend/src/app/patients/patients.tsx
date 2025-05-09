'use client'

import { useState } from 'react'
import { PatientList } from '@/components/patients/PatientList'
import { PatientDetails } from '@/components/patients/PatientDetails'
import { Button } from '@/components/ui/button'
import { type Patient, type Analysis } from '@/types/medical'

// Mock data - replace with actual API calls
const mockPatients: Patient[] = [
  {
    id: '1',
    name: 'John Doe',
    dateOfBirth: '1980-05-15',
    gender: 'male',
    contactDetails: '+1 234 567 8900',
    medicalHistory: 'No significant medical history.',
    createdAt: '2023-01-15T09:00:00Z'
  },
  // Add more mock patients as needed
]

const mockAnalyses: Analysis[] = [
  {
    id: '1',
    patientId: '1',
    type: 'brain_tumor',
    prediction: 'negative',
    confidence: 0.95,
    visualization: '',
    createdAt: '2023-03-10T11:20:00Z',
    modelAccuracy: 0.92
  },
  // Add more mock analyses as needed
]

export default function PatientsPage() {
  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null)

  const handleViewPatient = (patient: Patient) => {
    setSelectedPatient(patient)
  }

  const handleNewAnalysis = (patient: Patient) => {
    // Handle new analysis logic
    console.log('New analysis for patient:', patient)
  }

  return (
    <div className="container mx-auto py-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Patients</h1>
        {selectedPatient && (
          <Button variant="outline" onClick={() => setSelectedPatient(null)}>
            Back to List
          </Button>
        )}
      </div>

      {selectedPatient ? (
        <PatientDetails 
          patient={selectedPatient}
          analyses={mockAnalyses.filter(a => a.patientId === selectedPatient.id)}
        />
      ) : (
        <PatientList
          patients={mockPatients}
          onViewPatient={handleViewPatient}
          onNewAnalysis={handleNewAnalysis}
        />
      )}
    </div>
  )
}

