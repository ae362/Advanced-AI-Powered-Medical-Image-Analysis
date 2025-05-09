import { useState, useRef, useEffect } from 'react'
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Slider } from "@/components/ui/slider"
import { getGradCAMVisualization } from "@/lib/api"
import { toast } from "@/components/ui/use-toast"
import { ZoomIn, ZoomOut, Move } from 'lucide-react'

interface GradCAMVisualizationProps {
  analysisId: string
  originalImage: string
}

export function GradCAMVisualization({ analysisId, originalImage }: GradCAMVisualizationProps) {
  const [gradCAMImage, setGradCAMImage] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [opacity, setOpacity] = useState(0.5)
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = useState(false)
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 })
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    handleGenerateGradCAM()
  }, [analysisId])

  const handleGenerateGradCAM = async () => {
    setIsLoading(true)
    try {
      const result = await getGradCAMVisualization(analysisId)
      setGradCAMImage(result)
    } catch (error) {
      console.error('Error generating Grad-CAM:', error)
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to generate Grad-CAM visualization. Please try again.",
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleZoomIn = () => setZoom(prev => Math.min(prev + 0.1, 3))
  const handleZoomOut = () => setZoom(prev => Math.max(prev - 0.1, 0.5))

  const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    setIsDragging(true)
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y })
  }

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!isDragging) return
    setPan({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y
    })
  }

  const handleMouseUp = () => {
    setIsDragging(false)
  }

  return (
    <Card className="w-full max-w-4xl mx-auto">
      <CardHeader>
        <CardTitle>Grad-CAM Visualization</CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="flex justify-center items-center h-64">
            <span className="loading loading-spinner loading-lg"></span>
          </div>
        ) : gradCAMImage ? (
          <div className="space-y-4">
            <div 
              ref={containerRef}
              className="relative w-full h-64 overflow-hidden border rounded-lg"
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
            >
              <img
                src={originalImage}
                alt="Original"
                className="absolute top-0 left-0 w-full h-full object-contain"
                style={{
                  transform: `scale(${zoom}) translate(${pan.x}px, ${pan.y}px)`,
                  transition: 'transform 0.1s ease-out',
                }}
              />
              <img
                src={`data:image/png;base64,${gradCAMImage}`}
                alt="Grad-CAM Visualization"
                className="absolute top-0 left-0 w-full h-full object-contain"
                style={{
                  opacity: opacity,
                  transform: `scale(${zoom}) translate(${pan.x}px, ${pan.y}px)`,
                  transition: 'transform 0.1s ease-out',
                }}
              />
            </div>
            <div className="flex items-center justify-between">
              <div className="space-x-2">
                <Button variant="outline" size="icon" onClick={handleZoomIn}>
                  <ZoomIn className="h-4 w-4" />
                </Button>
                <Button variant="outline" size="icon" onClick={handleZoomOut}>
                  <ZoomOut className="h-4 w-4" />
                </Button>
              </div>
              <div className="flex items-center space-x-2">
                <span>Opacity:</span>
                <Slider
                  value={[opacity]}
                  min={0}
                  max={1}
                  step={0.01}
                  onValueChange={(value) => setOpacity(value[0])}
                  className="w-32"
                />
              </div>
            </div>
          </div>
        ) : (
          <Button onClick={handleGenerateGradCAM}>Generate Grad-CAM</Button>
        )}
      </CardContent>
    </Card>
  )
}

