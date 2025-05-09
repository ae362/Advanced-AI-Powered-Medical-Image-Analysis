import React from 'react';
import Link from 'next/link';
import { useRouter } from 'next/router';
import { Layout, Menu, Button, Dropdown, Avatar } from 'antd';
import {
  HomeOutlined,
  TeamOutlined,
  FileImageOutlined,
  FileTextOutlined,
  SettingOutlined,
  QuestionCircleOutlined,
  UserOutlined,
  LogoutOutlined,
  PlusOutlined,
  UploadOutlined,
  SearchOutlined,
  PrinterOutlined
} from '@ant-design/icons';

const { Header, Sider, Content } = Layout;

interface MainLayoutProps {
  children: React.ReactNode;
}

const MainLayout: React.FC<MainLayoutProps> = ({ children }) => {
  const router = useRouter();

  const userMenu = (
    <Menu>
      <Menu.Item key="profile" icon={<UserOutlined />}>
        Profile
      </Menu.Item>
      <Menu.Item key="logout" icon={<LogoutOutlined />}>
        Logout
      </Menu.Item>
    </Menu>
  );

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header className="flex items-center justify-between bg-white">
        <div className="flex items-center">
          <img src="/images/logo.png" alt="Logo" className="h-8 mr-4" />
          <h1 className="text-xl font-bold">Medical Image Analysis</h1>
        </div>
        <Menu mode="horizontal" selectedKeys={[router.pathname]}>
          <Menu.Item key="/" icon={<HomeOutlined />}>
            <Link href="/">Home</Link>
          </Menu.Item>
          <Menu.Item key="/patients" icon={<TeamOutlined />}>
            <Link href="/patients">Patients</Link>
          </Menu.Item>
          <Menu.Item key="/analysis" icon={<FileImageOutlined />}>
            <Link href="/analysis">Image Analysis</Link>
          </Menu.Item>
          <Menu.Item key="/reports" icon={<FileTextOutlined />}>
            <Link href="/reports">Reports</Link>
          </Menu.Item>
          <Menu.Item key="/settings" icon={<SettingOutlined />}>
            <Link href="/settings">Settings</Link>
          </Menu.Item>
          <Menu.Item key="/help" icon={<QuestionCircleOutlined />}>
            <Link href="/help">Help/Support</Link>
          </Menu.Item>
        </Menu>
        <Dropdown overlay={userMenu} trigger={['click']}>
          <Avatar icon={<UserOutlined />} />
        </Dropdown>
      </Header>
      <Layout>
        <Sider width={200} className="bg-white">
          <Menu
            mode="inline"
            defaultSelectedKeys={['1']}
            defaultOpenKeys={['sub1']}
            style={{ height: '100%', borderRight: 0 }}
          >
            <Menu.Item key="add_patient" icon={<PlusOutlined />}>
              Add New Patient
            </Menu.Item>
            <Menu.Item key="upload_images" icon={<UploadOutlined />}>
              Upload Images
            </Menu.Item>
            <Menu.Item key="analyze_images" icon={<SearchOutlined />}>
              Analyze Images
            </Menu.Item>
            <Menu.Item key="print_report" icon={<PrinterOutlined />}>
              Print Report
            </Menu.Item>
          </Menu>
        </Sider>
        <Layout style={{ padding: '24px' }}>
          <Content className="bg-white p-6 rounded-lg">
            {children}
          </Content>
        </Layout>
      </Layout>
    </Layout>
  );
};

export default MainLayout;

