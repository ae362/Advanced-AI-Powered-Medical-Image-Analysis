'use client'

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Patient, Analysis } from "@/types/medical"
import { formatDate } from "@/lib/utils"

interface PatientDetailsProps {
  patient: Patient
  analyses: Analysis[]
}

export function PatientDetails({ patient, analyses }: PatientDetailsProps) {
  return (
    <Tabs defaultValue="info" className="w-full">
      <TabsList>
        <TabsTrigger value="info">Patient Information</TabsTrigger>
        <TabsTrigger value="history">Medical History</TabsTrigger>
        <TabsTrigger value="analyses">Analyses</TabsTrigger>
      </TabsList>
      
      <TabsContent value="info">
        <Card>
          <CardHeader>
            <CardTitle>Patient Information</CardTitle>
            <CardDescription>View and manage patient details</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-2">
              <Label>Name</Label>
              <div className="text-sm">{patient.name}</div>
            </div>
            <div className="grid gap-2">
              <Label>Date of Birth</Label>
              <div className="text-sm">{formatDate(patient.dateOfBirth)}</div>
            </div>
            <div className="grid gap-2">
              <Label>Gender</Label>
              <div className="text-sm capitalize">{patient.gender}</div>
            </div>
            <div className="grid gap-2">
              <Label>Contact Details</Label>
              <div className="text-sm">{patient.contactDetails}</div>
            </div>
          </CardContent>
        </Card>
      </TabsContent>

      <TabsContent value="history">
        <Card>
          <CardHeader>
            <CardTitle>Medical History</CardTitle>
            <CardDescription>Patient's medical history and notes</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="whitespace-pre-wrap text-sm">
              {patient.medicalHistory}
            </div>
          </CardContent>
        </Card>
      </TabsContent>

      <TabsContent value="analyses">
        <Card>
          <CardHeader>
            <CardTitle>Analysis History</CardTitle>
            <CardDescription>Previous medical analyses and results</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {analyses.map((analysis) => (
                <Card key={analysis.id}>
                  <CardHeader>
                    <CardTitle className="text-base">
                      {analysis.type === 'brain_tumor' ? 'Brain Tumor Analysis' : 'Cancer Analysis'}
                    </CardTitle>
                    <CardDescription>{formatDate(analysis.createdAt)}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid gap-2 text-sm">
                      <div className="flex justify-between">
                        <span className="font-medium">Result:</span>
                        <span className={analysis.prediction === 'positive' ? 'text-red-500' : 'text-green-500'}>
                          {analysis.prediction}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="font-medium">Confidence:</span>
                        <span>{(analysis.confidence * 100).toFixed(2)}%</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </CardContent>
        </Card>
      </TabsContent>
    </Tabs>
  )
}

