'use client'

import { useState } from 'react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import * as z from 'zod'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Patient, Disease, AnalysisFormData } from "@/types/medical"
import { Upload } from 'lucide-react'
import { toast } from '@/components/ui/use-toast'

const formSchema = z.object({
  patientId: z.string(),
  type: z.enum(['brain_tumor', 'cancer','breast_cancer']),
  image: z.instanceof(File, { message: "Please select an image file" })
})

interface AnalysisFormProps {
  patient: Patient
  onSubmit: (data: AnalysisFormData) => Promise<void>
  diseases: Disease[]
}

export function AnalysisForm({ patient, onSubmit, diseases }: AnalysisFormProps) {
  const [preview, setPreview] = useState<string | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)

  // Ensure we always have the default options available
  const defaultDiseases: Disease[] = [
    { id: "default-1", name: "Brain Tumor Detection", key: "brain_tumor", description: "Analysis for detecting brain tumors in MRI scans" },
    { id: "default-2", name: "Cancer Detection", key: "cancer", description: "General cancer detection and analysis" }
  ]

  // Combine default diseases with provided diseases, ensuring no duplicates
  const allDiseases = [...defaultDiseases, ...diseases].reduce((acc, disease) => {
    if (!acc.some(d => d.key === disease.key)) {
      acc.push(disease)
    }
    return acc
  }, [] as Disease[])

  const form = useForm<AnalysisFormData>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      patientId: patient.id,
      type: 'brain_tumor',
    },
  })

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      form.setValue('image', file)
      const reader = new FileReader()
      reader.onloadend = () => {
        setPreview(reader.result as string)
      }
      reader.readAsDataURL(file)
    }
  }

  const handleSubmit = async (data: AnalysisFormData) => {
    try {
      setIsSubmitting(true)
      await onSubmit(data)
    } catch (error) {
      console.error('Form submission error:', error)
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to submit analysis. Please try again.",
      })
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>New Analysis</CardTitle>
        <CardDescription>
          Patient: {patient.name}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Form {...form}>
          <form onSubmit={form.handleSubmit(handleSubmit)} className="space-y-6">
            <FormField
              control={form.control}
              name="type"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Analysis Type</FormLabel>
                  <Select 
                    value={field.value} 
                    onValueChange={field.onChange}
                  >
                    <FormControl>
                      <SelectTrigger className="w-full">
                        <SelectValue placeholder="Select analysis type" />
                      </SelectTrigger>
                    </FormControl>
                    <SelectContent>
                      {allDiseases.map((disease) => (
                        <SelectItem 
                          key={disease.id} 
                          value={disease.key}
                        >
                          {disease.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="image"
              render={({ field: { onChange, ...field } }) => (
                <FormItem>
                  <FormLabel>Medical Image</FormLabel>
                  <FormControl>
                    <div className="space-y-4">
                      <Input
                        type="file"
                        accept="image/*"
                        onChange={(e) => {
                          handleImageChange(e)
                          onChange(e.target.files?.[0])
                        }}
                        className="hidden"
                        id="image-upload"
                      />
                      <div 
                        className={`
                          border-2 border-dashed rounded-lg p-6 text-center cursor-pointer
                          hover:bg-muted/50 transition-colors
                          ${preview ? 'border-primary' : 'border-muted-foreground/25'}
                        `}
                        onClick={() => document.getElementById('image-upload')?.click()}
                      >
                        {preview ? (
                          <img
                            src={preview}
                            alt="Preview"
                            className="max-h-[300px] mx-auto rounded-lg"
                          />
                        ) : (
                          <div className="flex flex-col items-center gap-2">
                            <Upload className="h-8 w-8 text-muted-foreground" />
                            <p className="text-sm text-muted-foreground">
                              Click or drag and drop to upload an image
                            </p>
                          </div>
                        )}
                      </div>
                    </div>
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            <Button 
              type="submit" 
              className="w-full"
              disabled={isSubmitting}
            >
              {isSubmitting ? 'Analyzing...' : 'Start Analysis'}
            </Button>
          </form>
        </Form>
      </CardContent>
    </Card>
  )
}

