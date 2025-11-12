# DataBoard - _Data Analysis and Visualization Web Application_

## Overview
This interactive web application enables users to seamlessly upload, analyze, visualize, and forecast datasets in a single interface. The platform provides end-to-end data handling (from login and dataset upload to dynamic reporting and personalized dashboards).

## Key Features
- User Authentication
    - Secure registration and login system using Streamlit session state.
    - User credentials are stored persistently — registration is required only once.
    - The profile section allows users to update their profile picture and name.

- Dataset Management
    - Users can upload multiple datasets (CSV format).
    - The app automatically detects common column names to compare data across files.

- Sector Analysis
    - An interactive feature allowing users to select any column from their dataset.
    - Generates dynamic visualizations (bar, line, scatter, etc.) based on selected data.

- Market Trends & Forecasting
    - Automatically analyzes market trends within the uploaded dataset.
    - Uses time-series forecasting to predict values for the next 15 days.

- Custom Reports
    - Users can choose specific columns to order or filter their dataset.
    - Generates a custom report instantly.
    - Option to download reports as CSV for offline use.

- Interactive Dashboards
    - Users can create personalized dashboards by selecting X-axis, Y-axis, and chart type.
    - Supports bar, line, area, pie, and scatter visualizations.

- Profile Management
    - Dedicated profile section for users to:
    - Upload or change profile picture.
    - Edit displayed username.
    - Includes a Logout button for secure exit and session reset.

## Tech Stack
- Frontend & Backend: Streamlit
- Data Handling: Pandas, NumPy
- Visualization: Matplotlib, Plotly, Seaborn
- Data Export: CSV
- Session Management: Streamlit Session State

## How It Works
- Register or log in to access your workspace.
- Upload one or more datasets.
- Perform sector analysis or compare common columns.
- Generate insights and forecast trends.
- Create custom dashboards and reports.
- Download results and manage your profile.
