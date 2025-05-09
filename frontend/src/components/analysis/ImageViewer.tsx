import React, { useState } from 'react';
import { Image, Slider, Button, Space } from 'antd';
import { ZoomInOutlined, ZoomOutOutlined, RotateLeftOutlined, RotateRightOutlined } from '@ant-design/icons';

interface ImageViewerProps {
  src: string;
}

const ImageViewer: React.FC<ImageViewerProps> = ({ src }) => {
  const [zoom, setZoom] = useState(100);
  const [rotation, setRotation] = useState(0);

  const handleZoomChange = (value: number) => {
    setZoom(value);
  };

  const handleRotate = (direction: 'left' | 'right') => {
    setRotation((prev) => (direction === 'left' ? prev - 90 : prev + 90));
  };

  return (
    <div>
      <Space direction="vertical" style={{ width: '100%' }}>
        <Image
          src={src}
          style={{
            transform: `scale(${zoom / 100}) rotate(${rotation}deg)`,
            transition: 'transform 0.3s ease',
          }}
          preview={false}
        />
        <Space>
          <Button icon={<ZoomOutOutlined />} onClick={() => handleZoomChange(Math.max(zoom - 10, 10))} />
          <Slider
            min={10}
            max={200}
            value={zoom}
            onChange={handleZoomChange}
            style={{ width: 200 }}
          />
          <Button icon={<ZoomInOutlined />} onClick={() => handleZoomChange(Math.min(zoom + 10, 200))} />
          <Button icon={<RotateLeftOutlined />} onClick={() => handleRotate('left')} />
          <Button icon={<RotateRightOutlined />} onClick={() => handleRotate('right')} />
        </Space>
      </Space>
    </div>
  );
};

export default ImageViewer;

