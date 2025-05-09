import React from 'react';
import { Card, Col, Row, Statistic } from 'antd';
import { UserOutlined, FileImageOutlined, FileTextOutlined } from '@ant-design/icons';

const SummaryPanel: React.FC = () => {
  return (
    <Row gutter={16}>
      <Col span={8}>
        <Card>
          <Statistic
            title="Total Patients"
            value={1128}
            prefix={<UserOutlined />}
          />
        </Card>
      </Col>
      <Col span={8}>
        <Card>
          <Statistic
            title="Pending Analyses"
            value={43}
            prefix={<FileImageOutlined />}
          />
        </Card>
      </Col>
      <Col span={8}>
        <Card>
          <Statistic
            title="Recent Reports"
            value={28}
            prefix={<FileTextOutlined />}
          />
        </Card>
      </Col>
    </Row>
  );
};

export default SummaryPanel;

