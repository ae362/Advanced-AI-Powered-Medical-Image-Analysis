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
import { Textarea } from "@/components/ui/textarea"
import { toast } from "@/components/ui/use-toast"
import { PlusIcon, Upload } from 'lucide-react'
import { Disease } from "@/types/medical"

interface AddDiseaseDialogProps {
  onDiseaseAdded: (disease: Omit<Disease, "id" | "is_active" | "created_at">) => Promise<void>
}

export function AddDiseaseDialog({ onDiseaseAdded }: AddDiseaseDialogProps) {
  const [open, setOpen] = useState(false)
  const [name, setName] = useState("")
  const [key, setKey] = useState("")
  const [description, setDescription] = useState("")
  const [isSubmitting, setIsSubmitting] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!name.trim() || !key.trim()) {
      toast({
        variant: "destructive",
        title: "Error",
        description: "Disease name and key are required",
      })
      return
    }

    setIsSubmitting(true)
    try {
      await onDiseaseAdded({ name, key, description })
      setOpen(false)
      setName("")
      setKey("")
      setDescription("")
      toast({
        title: "Disease Added",
        description: "The new disease has been added successfully. Upload a model to activate it.",
      })
    } catch (error) {
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to add disease. Please try again.",
      })
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="outline">
          <PlusIcon className="mr-2 h-4 w-4" />
          Add Medical Disease
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Add New Disease</DialogTitle>
          <DialogDescription>
            Add a new disease type for analysis. You'll need to upload a trained model after adding the disease.
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit}>
          <div className="grid gap-4 py-4">
            <div className="grid gap-2">
              <Label htmlFor="name">Disease Name</Label>
              <Input
                id="name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g., Lung Cancer"
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="key">Disease Key</Label>
              <Input
                id="key"
                value={key}
                onChange={(e) => setKey(e.target.value.toLowerCase().replace(/\s+/g, '_'))}
                placeholder="e.g., lung_cancer"
              />
              <p className="text-sm text-muted-foreground">
                This will be used as the identifier for the disease
              </p>
            </div>
            <div className="grid gap-2">
              <Label htmlFor="description">Description</Label>
              <Textarea
                id="description"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Describe the disease and its characteristics..."
              />
            </div>
          </div>
          <DialogFooter>
            <Button type="submit" disabled={isSubmitting}>
              {isSubmitting ? "Adding..." : "Add Disease"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}