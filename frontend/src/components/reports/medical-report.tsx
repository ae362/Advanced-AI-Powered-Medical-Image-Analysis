'use client'

import { useState, useRef } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { FileDown, Eye } from 'lucide-react'
import { Patient, Analysis } from "@/types/medical"
import { formatDate } from "@/lib/utils"
import html2canvas from 'html2canvas'
import jsPDF from 'jspdf'
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { cn } from "@/lib/utils"

interface MedicalReportProps {
  patient: Patient
  analyses: Analysis[]
}

export function MedicalReport({ patient, analyses }: MedicalReportProps) {
  const [doctorNotes, setDoctorNotes] = useState('')
  const [showPreview, setShowPreview] = useState(false)
  const [isGenerating, setIsGenerating] = useState(false)
  const reportRef = useRef<HTMLDivElement>(null)

  if (!analyses.length) {
    return (
      <Card>
        <CardContent className="pt-6">
          <p className="text-center text-muted-foreground">
            No analysis history available for this patient.
          </p>
        </CardContent>
      </Card>
    )
  }

  const generatePDF = async () => {
    if (!reportRef.current) return
    setIsGenerating(true)

    try {
      const canvas = await html2canvas(reportRef.current, {
        scale: 2,
        useCORS: true,
        logging: false,
        backgroundColor: '#000000'
      })

      const imgData = canvas.toDataURL('image/jpeg', 1.0)
      const pdf = new jsPDF('p', 'mm', 'a4')
      const pdfWidth = pdf.internal.pageSize.getWidth()
      const pdfHeight = pdf.internal.pageSize.getHeight()
      const imgWidth = canvas.width
      const imgHeight = canvas.height
      const ratio = Math.min(pdfWidth / imgWidth, pdfHeight / imgHeight)
      const imgX = (pdfWidth - imgWidth * ratio) / 2
      const imgY = 0

      pdf.addImage(imgData, 'JPEG', imgX, imgY, imgWidth * ratio, imgHeight * ratio)
      pdf.save(`medical-report-${patient.id}-${formatDate(new Date())}.pdf`)
    } catch (error) {
      console.error('Error generating PDF:', error)
    } finally {
      setIsGenerating(false)
    }
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Doctor's Notes</CardTitle>
        </CardHeader>
        <CardContent>
          <Textarea
            placeholder="Enter your medical observations and recommendations..."
            value={doctorNotes}
            onChange={(e) => setDoctorNotes(e.target.value)}
            className="min-h-[200px]"
          />
        </CardContent>
      </Card>

      <div className="flex gap-4">
        <Button onClick={generatePDF} disabled={isGenerating}>
          <FileDown className="mr-2 h-4 w-4" />
          {isGenerating ? 'Generating PDF...' : 'Download Report'}
        </Button>

        <Button variant="outline" onClick={() => setShowPreview(!showPreview)}>
          <Eye className="mr-2 h-4 w-4" />
          {showPreview ? 'Hide Preview' : 'Show Preview'}
        </Button>
      </div>

      <div ref={reportRef} className={cn(showPreview ? 'block' : 'hidden', "bg-black text-white")}>
        <div className="max-w-4xl mx-auto p-8">
          <div className="space-y-8">
            <div>
              <h1 className="text-3xl font-bold mb-2">Medical Report</h1>
              <p className="text-xl text-muted-foreground">Patient ID: {patient.id}</p>
            </div>

            <div className="space-y-6">
              {analyses.map((analysis, index) => (
                <div key={analysis.id} className="space-y-4">
                  <h3 className="text-lg font-semibold">
                    Analysis {index + 1} - {formatDate(analysis.date)}
                  </h3>
                  
                  <div className="grid gap-4 rounded-lg bg-gray-900 p-4">
                    <div className="flex justify-between items-center">
                      <span className="font-medium">Type:</span>
                      <Badge variant={analysis.type === 'brain_tumor' ? 'default' : 'secondary'}>
                        {analysis.type === 'brain_tumor' ? 'Brain Tumor' : 'Cancer'}
                      </Badge>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="font-medium">Diagnosis:</span>
                      <Badge 
                        variant={analysis.prediction === 'positive' ? 'destructive' : 'outline'}
                        className="capitalize font-medium"
                      >
                        {analysis.prediction === 'positive' ? 'Abnormal' : 'Normal'}
                      </Badge>
                    </div>
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="font-medium">Confidence:</span>
                        <span>{(analysis.confidence * 100).toFixed(2)}%</span>
                      </div>
                      <Progress 
                        value={analysis.confidence * 100} 
                        className="h-2"
                        indicatorClassName={
                          analysis.prediction === 'positive' 
                            ? "bg-destructive" 
                            : "bg-green-600"
                        }
                      />
                    </div>
                  </div>

                  <div className="relative bg-black rounded-lg overflow-hidden">
                    <div className="aspect-[16/9] relative">
                      <img
                        src={`data:image/png;base64,${analysis.visualization}`}
                        alt={`Analysis ${index + 1} Visualization`}
                        className="absolute inset-0 w-full h-full object-contain"
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {doctorNotes && (
              <div className="space-y-4">
                <h2 className="text-2xl font-semibold">Doctor's Notes</h2>
                <div className="p-4 bg-gray-900 rounded-lg">
                  <div className="whitespace-pre-wrap break-words max-w-full">
                    {doctorNotes}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

