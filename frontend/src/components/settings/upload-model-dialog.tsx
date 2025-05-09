'use client'

import { useState } from "react"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { toast } from "@/components/ui/use-toast"
import { Upload } from 'lucide-react'
import { Disease } from "@/types/medical"

interface UploadModelDialogProps {
  disease: Disease
  onModelUploaded: () => void
}

export function UploadModelDialog({ disease, onModelUploaded }: UploadModelDialogProps) {
  const [open, setOpen] = useState(false)
  const [file, setFile] = useState<File | null>(null)
  const [isUploading, setIsUploading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!file) {
      toast({
        variant: "destructive",
        title: "Error",
        description: "Please select a model file",
      })
      return
    }

    setIsUploading(true)
    try {
      const formData = new FormData()
      formData.append('model_file', file)

      const response = await fetch(`/api/diseases/${disease.id}/upload-model/`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Failed to upload model')
      }

      setOpen(false)
      setFile(null)
      onModelUploaded()
      toast({
        title: "Model Uploaded",
        description: "The model has been uploaded and the disease is now active.",
      })
    } catch (error) {
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to upload model. Please try again.",
      })
    } finally {
      setIsUploading(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm">
          <Upload className="mr-2 h-4 w-4" />
          Upload Model
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Upload Model for {disease.name}</DialogTitle>
          <DialogDescription>
            Upload a trained model file for this disease. The model should be compatible with your system.
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit}>
          <div className="grid gap-4 py-4">
            <div className="grid gap-2">
              <Label htmlFor="model">Model File</Label>
              <Input
                id="model"
                type="file"
                accept=".h5,.keras,.pkl"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
              />
              <p className="text-sm text-muted-foreground">
                Supported formats: .h5, .keras, .pkl
              </p>
            </div>
          </div>
          <DialogFooter>
            <Button type="submit" disabled={isUploading || !file}>
              {isUploading ? "Uploading..." : "Upload Model"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}