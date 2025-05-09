import React from 'react';
import MainLayout from '../components/layout/MainLayout';
import UserPreferences from '../components/settings/UserPreferences';
import AdminSettings from '../components/settings/AdminSettings';
import { Typography, Tabs } from 'antd';

const { Title } = Typography;
const { TabPane } = Tabs;

const SettingsPage: React.FC = () => {
  return (
    <MainLayout>
      <Title level={2}>Settings</Title>
      <Tabs defaultActiveKey="1">
        <TabPane tab="User Preferences" key="1">
          <UserPreferences />
        </TabPane>
        <TabPane tab="Admin Settings" key="2">
          <AdminSettings />
        </TabPane>
      </Tabs>
    </MainLayout>
  );
};

export default SettingsPage;

