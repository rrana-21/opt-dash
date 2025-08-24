import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import calendar

# Set page config
st.set_page_config(
    page_title="Clarus Vision Analytics",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Clarus optometry styling with blue-teal color scheme
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        color: #0891b2;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .clarus-brand {
        font-size: 1.4rem;
        color: #0e7490;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f9ff;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #0891b2;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .optometry-specific {
        background: linear-gradient(135deg, #0891b2 0%, #06b6d4 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1.5rem 0;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .sidebar .sidebar-content {
        background-color: #f0f9ff;
    }
    .upload-section {
        border: 3px dashed #0891b2;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        background-color: #f0f9ff;
        margin-bottom: 1rem;
    }
    .insight-box {
        background-color: #f0fdff;
        border-left: 5px solid #0891b2;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .conversion-highlight {
        background-color: #ecfdf5;
        border-left: 5px solid #10b981;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .retention-highlight {
        background-color: #fef3c7;
        border-left: 5px solid #f59e0b;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .stMetric > div > div > div > div {
        font-size: 2rem !important;
        font-weight: bold !important;
    }
    .stMetric > div > div > div > div:first-child {
        font-size: 1.2rem !important;
        color: #0891b2 !important;
        font-weight: 600 !important;
    }
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #f9fafb;
        padding: 4px;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        padding: 12px 20px;
        background-color: white;
        border: 1px solid #e5e7eb;
        color: #6b7280;
        font-weight: 500;
        font-size: 14px;
        border-radius: 6px;
        margin: 0 1px;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0891b2 !important;
        color: white !important;
        border-color: #0891b2 !important;
        font-weight: 600 !important;
    }
    .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
        background-color: #f3f4f6;
        border-color: #d1d5db;
    }
    /* Mobile responsive improvements */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        .metric-card {
            padding: 1rem;
        }
        .optometry-specific {
            padding: 1rem;
        }
        .js-plotly-plot .gtitle {
            font-size: 12px !important;
        }
    }
    .js-plotly-plot {
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .js-plotly-plot .plotly-graph-div {
        overflow: visible !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(file_path_or_buffer):
    """Load and preprocess the optometry data"""
    try:
        # Handle both file paths and uploaded files
        if isinstance(file_path_or_buffer, str):
            if file_path_or_buffer.endswith('.xlsx'):
                df = pd.read_excel(file_path_or_buffer)
            else:
                df = pd.read_csv(file_path_or_buffer)
        else:
            if file_path_or_buffer.name.endswith('.xlsx'):
                df = pd.read_excel(file_path_or_buffer)
            else:
                df = pd.read_csv(file_path_or_buffer)
        
        # Convert Date to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Create additional date columns for easier filtering
        df['Month'] = df['Date'].dt.to_period('M')
        df['Year'] = df['Date'].dt.year
        df['Day_of_Week'] = df['Date'].dt.day_name()
        df['Month_Name'] = df['Date'].dt.strftime('%B %Y')
        df['Quarter'] = df['Date'].dt.to_period('Q')
        
        # Calculate revenue
        df['Revenue'] = df['TotalPrice']
        
        # Create service type categories based on common optometry services
        df['Is_Eye_Exam'] = df['ServiceType'].str.contains('Exam|Eye Exam|Comprehensive|Annual', case=False, na=False)
        df['Is_Frame_Sale'] = df['ServiceType'].str.contains('Frame|Glasses|Eyewear', case=False, na=False)
        df['Is_Contact_Sale'] = df['ServiceType'].str.contains('Contact', case=False, na=False)
        df['Is_Lens_Sale'] = df['ServiceType'].str.contains('Lens', case=False, na=False) & ~df['Is_Contact_Sale']
        df['Is_Specialty_Service'] = df['ServiceType'].str.contains('Fitting|Dry Eye|Glaucoma|Specialty', case=False, na=False)
        
        # Create eyewear purchase flag (frames or contacts)
        df['Is_Eyewear_Purchase'] = df['Is_Frame_Sale'] | df['Is_Contact_Sale'] | df['Is_Lens_Sale']
        
        # Create frame brand categories
        if 'FrameBrand' in df.columns:
            # Common premium vs budget brands (you can customize this list)
            premium_brands = ['Ray-Ban', 'Oakley', 'Gucci', 'Prada', 'Tom Ford', 'Silhouette', 'Lindberg']
            df['Is_Premium_Frame'] = df['FrameBrand'].isin(premium_brands)
            df['Frame_Category'] = df['FrameBrand'].apply(
                lambda x: 'Premium' if x in premium_brands 
                else 'Budget' if pd.notna(x) and x != ''
                else 'No Frame'
            )
        
        # Create lens type categories
        if 'LensType' in df.columns:
            progressive_types = ['Progressive', 'Multifocal', 'Varifocal']
            specialty_types = ['Anti-Reflective', 'Blue Light', 'Photochromic', 'Transitions']
            df['Is_Progressive_Lens'] = df['LensType'].isin(progressive_types)
            df['Is_Specialty_Lens'] = df['LensType'].isin(specialty_types)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please ensure your file has the required columns: TransactionID, Date, PatientID, ServiceType, FrameBrand, LensType, Quantity, UnitPrice, InsuranceUsed, TotalPrice")
        return None

def analyze_exam_to_purchase_conversion(df):
    """Analyze exam-to-purchase conversion rate"""
    # Get patients who had eye exams
    exam_patients = df[df['Is_Eye_Exam']]['PatientID'].unique()
    
    # Get patients who made eyewear purchases
    purchase_patients = df[df['Is_Eyewear_Purchase']]['PatientID'].unique()
    
    # Calculate conversion rate
    converted_patients = set(exam_patients) & set(purchase_patients)
    conversion_rate = (len(converted_patients) / len(exam_patients)) * 100 if len(exam_patients) > 0 else 0
    
    # Get exam and purchase details
    exam_revenue = df[df['Is_Eye_Exam']]['TotalPrice'].sum()
    eyewear_revenue = df[df['Is_Eyewear_Purchase']]['TotalPrice'].sum()
    
    return {
        'total_exam_patients': len(exam_patients),
        'converted_patients': len(converted_patients),
        'conversion_rate': conversion_rate,
        'exam_revenue': exam_revenue,
        'eyewear_revenue': eyewear_revenue,
        'total_revenue': exam_revenue + eyewear_revenue
    }

def create_conversion_analysis(df):
    """Create exam-to-purchase conversion visualization"""
    conversion_data = analyze_exam_to_purchase_conversion(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Conversion funnel
        stages = ['Eye Exam Patients', 'Eyewear Purchasers']
        values = [conversion_data['total_exam_patients'], conversion_data['converted_patients']]
        
        fig_funnel = go.Figure(go.Funnel(
            y=stages,
            x=values,
            textinfo="value+percent initial",
            marker={"color": ["#0891b2", "#06b6d4"]},
            connector={"line": {"color": "#67e8f9", "dash": "dot", "width": 3}}
        ))
        
        fig_funnel.update_layout(
            title="Exam-to-Purchase Conversion Funnel",
            title_font_size=14,
            title_x=0.5,
            title_font_color='#0891b2',
            height=300,
            margin=dict(t=50, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_funnel, use_container_width=True)
    
    with col2:
        # Revenue breakdown
        revenue_data = pd.DataFrame({
            'Service': ['Eye Exams', 'Eyewear Sales'],
            'Revenue': [conversion_data['exam_revenue'], conversion_data['eyewear_revenue']]
        })
        
        fig_revenue = px.pie(
            revenue_data,
            values='Revenue',
            names='Service',
            title="Revenue Split: Exams vs Eyewear",
            color_discrete_sequence=['#0891b2', '#06b6d4']
        )
        fig_revenue.update_traces(
            textposition="auto",
            textinfo="percent+label",
            textfont_size=12,
            hovertemplate='<b>%{label}</b><br>Revenue: $%{value:,.2f}<br>Percentage: %{percent}<extra></extra>'
        )
        fig_revenue.update_layout(
            title_font_size=14,
            title_x=0.5,
            title_font_color='#0891b2',
            height=300,
            margin=dict(t=50, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_revenue, use_container_width=True)
    


def create_top_selling_analysis(df):
    """Create top-selling frames and lens options analysis"""
    col1, col2 = st.columns(2)
    
    with col1:
        # Top frame brands - check if we have any frame sales first
        frame_sales_exist = False
        if 'FrameBrand' in df.columns and df['FrameBrand'].notna().any():
            # Look for any eyewear-related transactions
            eyewear_df = df[df['FrameBrand'].notna() & (df['FrameBrand'] != '')]
            if len(eyewear_df) > 0:
                frame_sales = eyewear_df.groupby('FrameBrand').agg({
                    'TotalPrice': 'sum',
                    'TransactionID': 'count',
                    'UnitPrice': 'mean'
                }).reset_index().sort_values('TotalPrice', ascending=False).head(10)
                
                if len(frame_sales) > 0:
                    frame_sales_exist = True
                    fig_frames = px.bar(
                        frame_sales,
                        x='TotalPrice',
                        y='FrameBrand',
                        orientation='h',
                        title="Top Frame Brands by Revenue",
                        color='UnitPrice',
                        color_continuous_scale=['#a7f3d0', '#0891b2'],
                        text='TotalPrice'
                    )
                    fig_frames.update_traces(
                        texttemplate='$%{text:,.0f}',
                        textposition='outside',
                        hovertemplate='<b>%{y}</b><br>Revenue: $%{x:,.2f}<br>Units Sold: %{customdata}<br>Avg Price: $%{marker.color:.2f}<extra></extra>',
                        customdata=frame_sales['TransactionID']
                    )
                    fig_frames.update_layout(
                        title_font_size=14,
                        title_x=0.5,
                        title_font_color='#0891b2',
                        height=400,
                        margin=dict(t=50, b=50, l=120, r=80),
                        showlegend=False
                    )
                    st.plotly_chart(fig_frames, use_container_width=True)
        
        if not frame_sales_exist:
            st.info("No frame brand data available for analysis. Frame sales may be recorded differently in your data.")
    
    with col2:
        # Top lens types - check if we have any lens data
        lens_sales_exist = False
        if 'LensType' in df.columns and df['LensType'].notna().any():
            lens_df = df[df['LensType'].notna() & (df['LensType'] != '')]
            if len(lens_df) > 0:
                lens_sales = lens_df.groupby('LensType').agg({
                    'TotalPrice': 'sum',
                    'TransactionID': 'count',
                    'UnitPrice': 'mean'
                }).reset_index().sort_values('TotalPrice', ascending=False).head(10)
                
                if len(lens_sales) > 0:
                    lens_sales_exist = True
                    fig_lenses = px.bar(
                        lens_sales,
                        x='TotalPrice',
                        y='LensType',
                        orientation='h',
                        title="Top Lens Types by Revenue",
                        color='UnitPrice',
                        color_continuous_scale=['#fef3c7', '#f59e0b'],
                        text='TotalPrice'
                    )
                    fig_lenses.update_traces(
                        texttemplate='$%{text:,.0f}',
                        textposition='outside',
                        hovertemplate='<b>%{y}</b><br>Revenue: $%{x:,.2f}<br>Units Sold: %{customdata}<br>Avg Price: $%{marker.color:.2f}<extra></extra>',
                        customdata=lens_sales['TransactionID']
                    )
                    fig_lenses.update_layout(
                        title_font_size=14,
                        title_x=0.5,
                        title_font_color='#f59e0b',
                        height=400,
                        margin=dict(t=50, b=50, l=120, r=80),
                        showlegend=False
                    )
                    st.plotly_chart(fig_lenses, use_container_width=True)
        
        if not lens_sales_exist:
            st.info("No lens type data available for analysis. Lens sales may be recorded differently in your data.")
    
    # Service type analysis as alternative
    st.subheader("Product and Service Performance")
    service_analysis = df.groupby('ServiceType').agg({
        'TotalPrice': ['sum', 'mean'],
        'TransactionID': 'count',
        'UnitPrice': 'mean'
    }).round(2)
    service_analysis.columns = ['Total Revenue', 'Avg Revenue', 'Transaction Count', 'Avg Unit Price']
    service_analysis = service_analysis.sort_values('Total Revenue', ascending=False)
    
    # Display the service analysis table
    st.dataframe(
        service_analysis.style.format({
            'Total Revenue': '${:,.2f}',
            'Avg Revenue': '${:.2f}',
            'Transaction Count': '{:,}',
            'Avg Unit Price': '${:.2f}'
        }),
        use_container_width=True
    )
    
    # Service type revenue chart
    service_revenue_data = service_analysis.reset_index().head(10)
    fig_services = px.bar(
        service_revenue_data,
        x='ServiceType',
        y='Total Revenue',
        title="Revenue by Service Type",
        color='Total Revenue',
        color_continuous_scale=['#a7f3d0', '#0891b2'],
        text='Total Revenue'
    )
    fig_services.update_traces(
        texttemplate='$%{text:,.0f}',
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Revenue: $%{y:,.2f}<extra></extra>'
    )
    fig_services.update_layout(
        title_font_size=14,
        title_x=0.5,
        title_font_color='#0891b2',
        height=400,
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig_services, use_container_width=True)

def analyze_patient_retention(df):
    """Analyze patient recall and retention metrics"""
    # Calculate time between visits for each patient
    patient_visits = df.groupby('PatientID').agg({
        'Date': ['count', 'min', 'max'],
        'TotalPrice': 'sum'
    }).reset_index()
    
    patient_visits.columns = ['PatientID', 'Visit_Count', 'First_Visit', 'Last_Visit', 'Total_Spent']
    
    # Calculate days between visits
    patient_visits['Days_Since_First'] = (patient_visits['Last_Visit'] - patient_visits['First_Visit']).dt.days
    
    # Categorize patients
    patient_visits['Patient_Type'] = patient_visits.apply(lambda row: 
        'New Patient' if row['Visit_Count'] == 1
        else 'Returning Patient (< 1 year)' if row['Days_Since_First'] < 365 and row['Visit_Count'] > 1
        else 'Annual Return Patient' if 365 <= row['Days_Since_First'] <= 400 and row['Visit_Count'] > 1
        else 'Overdue Patient' if row['Days_Since_First'] > 400
        else 'Regular Patient', axis=1
    )
    
    # Calculate retention metrics
    total_patients = len(patient_visits)
    returning_patients = len(patient_visits[patient_visits['Visit_Count'] > 1])
    retention_rate = (returning_patients / total_patients) * 100 if total_patients > 0 else 0
    
    # Patients due for annual exam (assuming 12-month cycle)
    annual_due = len(patient_visits[patient_visits['Days_Since_First'] >= 365])
    annual_completed = len(patient_visits[patient_visits['Patient_Type'] == 'Annual Return Patient'])
    annual_recall_rate = (annual_completed / annual_due) * 100 if annual_due > 0 else 0
    
    return patient_visits, {
        'total_patients': total_patients,
        'returning_patients': returning_patients,
        'retention_rate': retention_rate,
        'annual_recall_rate': annual_recall_rate,
        'avg_visits_per_patient': patient_visits['Visit_Count'].mean()
    }

def create_retention_analysis(df):
    """Create patient retention and recall visualization"""
    retention_data, summary = analyze_patient_retention(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Patient type distribution
        patient_distribution = retention_data['Patient_Type'].value_counts().reset_index()
        patient_distribution.columns = ['Patient_Type', 'Count']
        
        fig_patient_types = px.bar(
            patient_distribution,
            x='Patient_Type',
            y='Count',
            title="Patient Visit Patterns",
            color='Count',
            color_continuous_scale=['#a7f3d0', '#0891b2'],
            text='Count'
        )
        fig_patient_types.update_traces(
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Patients: %{y}<extra></extra>'
        )
        fig_patient_types.update_layout(
            title_font_size=14,
            title_x=0.5,
            title_font_color='#0891b2',
            height=350,
            showlegend=False,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_patient_types, use_container_width=True)
    
    with col2:
        # Visit frequency distribution
        visit_counts = retention_data['Visit_Count'].value_counts().sort_index().head(10).reset_index()
        visit_counts.columns = ['Visit_Count', 'Patient_Count']
        
        fig_visits = px.bar(
            visit_counts,
            x='Visit_Count',
            y='Patient_Count',
            title="Patient Visit Frequency",
            color='Patient_Count',
            color_continuous_scale=['#fef3c7', '#f59e0b'],
            text='Patient_Count'
        )
        fig_visits.update_traces(
            textposition='outside',
            hovertemplate='<b>%{x} visits</b><br>Patients: %{y}<extra></extra>'
        )
        fig_visits.update_layout(
            title_font_size=14,
            title_x=0.5,
            title_font_color='#f59e0b',
            height=350,
            showlegend=False
        )
        st.plotly_chart(fig_visits, use_container_width=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Patient Retention Rate", f"{summary['retention_rate']:.1f}%")
    with col2:
        st.metric("Annual Recall Rate", f"{summary['annual_recall_rate']:.1f}%")
    with col3:
        st.metric("Avg Visits/Patient", f"{summary['avg_visits_per_patient']:.1f}")
    with col4:
        st.metric("Total Active Patients", f"{summary['total_patients']:,}")
    

def create_insurance_analysis(df):
    """Create comprehensive insurance vs out-of-pocket analysis"""
    col1, col2 = st.columns(2)
    
    with col1:
        # Insurance usage overview
        insurance_breakdown = df.groupby('InsuranceUsed').agg({
            'TotalPrice': 'sum',
            'TransactionID': 'count'
        }).reset_index()
        
        fig_insurance = px.pie(
            insurance_breakdown,
            values='TotalPrice',
            names='InsuranceUsed',
            title="Revenue: Insurance vs Out-of-Pocket",
            color_discrete_sequence=['#0891b2', '#06b6d4']
        )
        fig_insurance.update_traces(
            textposition="auto",
            textinfo="percent+label",
            textfont_size=12,
            hovertemplate='<b>%{label}</b><br>Revenue: $%{value:,.2f}<br>Transactions: %{customdata}<extra></extra>',
            customdata=insurance_breakdown['TransactionID']
        )
        fig_insurance.update_layout(
            title_font_size=14,
            title_x=0.5,
            title_font_color='#0891b2',
            height=350
        )
        st.plotly_chart(fig_insurance, use_container_width=True)
    
    with col2:
        # Average transaction value by payment type
        avg_values = df.groupby('InsuranceUsed')['TotalPrice'].mean().reset_index()
        
        fig_avg = px.bar(
            avg_values,
            x='InsuranceUsed',
            y='TotalPrice',
            title="Average Transaction Value",
            color='TotalPrice',
            color_continuous_scale=['#a7f3d0', '#0891b2'],
            text='TotalPrice'
        )
        fig_avg.update_traces(
            texttemplate='$%{text:.0f}',
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Avg Value: $%{y:.2f}<extra></extra>'
        )
        fig_avg.update_layout(
            title_font_size=14,
            title_x=0.5,
            title_font_color='#0891b2',
            height=350,
            showlegend=False
        )
        st.plotly_chart(fig_avg, use_container_width=True)
    
    # Insurance usage by service type
    st.subheader("Insurance Usage by Service Type")
    service_insurance = df.groupby(['ServiceType', 'InsuranceUsed']).agg({
        'TotalPrice': 'sum',
        'TransactionID': 'count'
    }).reset_index()
    
    fig_service_insurance = px.bar(
        service_insurance,
        x='ServiceType',
        y='TotalPrice',
        color='InsuranceUsed',
        title="Revenue by Service Type and Payment Method",
        color_discrete_map={'Yes': '#0891b2', 'No': '#06b6d4'},
        barmode='group'
    )
    fig_service_insurance.update_traces(
        hovertemplate='<b>%{x}</b><br>%{fullData.name}<br>Revenue: $%{y:,.2f}<extra></extra>'
    )
    fig_service_insurance.update_layout(
        title_font_size=14,
        title_x=0.5,
        title_font_color='#0891b2',
        height=350,
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig_service_insurance, use_container_width=True)

def create_service_mix_analysis(df):
    """Create service mix analysis"""
    # Service type breakdown
    service_analysis = df.groupby('ServiceType').agg({
        'TotalPrice': ['sum', 'mean'],
        'TransactionID': 'count',
        'PatientID': 'nunique'
    }).round(2)
    service_analysis.columns = ['Total_Revenue', 'Avg_Revenue', 'Transaction_Count', 'Unique_Patients']
    service_analysis = service_analysis.sort_values('Total_Revenue', ascending=False).reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue by service type
        fig_service_rev = px.bar(
            service_analysis.head(10),
            x='ServiceType',
            y='Total_Revenue',
            title="Revenue by Service Type",
            color='Total_Revenue',
            color_continuous_scale=['#a7f3d0', '#0891b2'],
            text='Total_Revenue'
        )
        fig_service_rev.update_traces(
            texttemplate='$%{text:,.0f}',
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Revenue: $%{y:,.2f}<br>Transactions: %{customdata}<extra></extra>',
            customdata=service_analysis.head(10)['Transaction_Count']
        )
        fig_service_rev.update_layout(
            title_font_size=14,
            title_x=0.5,
            title_font_color='#0891b2',
            height=400,
            showlegend=False,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_service_rev, use_container_width=True)
    
    with col2:
        # Service volume distribution
        fig_service_vol = px.pie(
            service_analysis.head(8),
            values='Transaction_Count',
            names='ServiceType',
            title="Transaction Volume by Service",
            color_discrete_sequence=px.colors.sequential.Teal
        )
        fig_service_vol.update_traces(
            textposition="auto",
            textinfo="percent",
            textfont_size=10,
            hovertemplate='<b>%{label}</b><br>Transactions: %{value}<br>Percentage: %{percent}<extra></extra>'
        )
        fig_service_vol.update_layout(
            title_font_size=14,
            title_x=0.5,
            title_font_color='#0891b2',
            height=400
        )
        st.plotly_chart(fig_service_vol, use_container_width=True)
    
    # Detailed service performance table
    st.subheader("Detailed Service Performance")
    st.dataframe(
        service_analysis.style.format({
            'Total_Revenue': '${:,.2f}',
            'Avg_Revenue': '${:.2f}',
            'Transaction_Count': '{:,}',
            'Unique_Patients': '{:,}'
        }),
        use_container_width=True
    )

def create_appointment_utilization_analysis(df):
    """Create appointment utilization and scheduling analysis"""
    # Day of week analysis
    dow_analysis = df.groupby('Day_of_Week').agg({
        'TransactionID': 'count',
        'TotalPrice': 'sum',
        'PatientID': 'nunique'
    }).reset_index()
    
    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_analysis['Day_of_Week'] = pd.Categorical(dow_analysis['Day_of_Week'], categories=day_order, ordered=True)
    dow_analysis = dow_analysis.sort_values('Day_of_Week')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Appointments by day of week
        fig_dow_appts = px.bar(
            dow_analysis,
            x='Day_of_Week',
            y='TransactionID',
            title="Appointments by Day of Week",
            color='TransactionID',
            color_continuous_scale=['#a7f3d0', '#0891b2'],
            text='TransactionID'
        )
        fig_dow_appts.update_traces(
            texttemplate='%{text}',
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Appointments: %{y}<br>Revenue: $%{customdata:,.2f}<extra></extra>',
            customdata=dow_analysis['TotalPrice']
        )
        fig_dow_appts.update_layout(
            title_font_size=14,
            title_x=0.5,
            title_font_color='#0891b2',
            height=350,
            showlegend=False,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_dow_appts, use_container_width=True)
    
    with col2:
        # Revenue efficiency by day
        dow_analysis['Revenue_per_Appointment'] = dow_analysis['TotalPrice'] / dow_analysis['TransactionID']
        
        fig_efficiency = px.bar(
            dow_analysis,
            x='Day_of_Week',
            y='Revenue_per_Appointment',
            title="Revenue per Appointment by Day",
            color='Revenue_per_Appointment',
            color_continuous_scale=['#fef3c7', '#f59e0b'],
            text='Revenue_per_Appointment'
        )
        fig_efficiency.update_traces(
            texttemplate='$%{text:.0f}',
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Revenue/Appointment: $%{y:.2f}<extra></extra>'
        )
        fig_efficiency.update_layout(
            title_font_size=14,
            title_x=0.5,
            title_font_color='#f59e0b',
            height=350,
            showlegend=False,
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_efficiency, use_container_width=True)
    
    # Monthly appointment trends
    st.subheader("Monthly Appointment Trends")
    monthly_trends = df.groupby(df['Date'].dt.to_period('M')).agg({
        'TransactionID': 'count',
        'TotalPrice': 'sum',
        'PatientID': 'nunique'
    }).reset_index()
    monthly_trends['Month'] = monthly_trends['Date'].dt.strftime('%Y-%m')
    
    fig_monthly = px.line(
        monthly_trends,
        x='Month',
        y='TransactionID',
        title="Monthly Appointment Volume Trend",
        markers=True,
        line_shape='spline'
    )
    fig_monthly.update_traces(
        line_color='#0891b2',
        line_width=4,
        marker_size=8,
        hovertemplate='<b>%{x}</b><br>Appointments: %{y}<br>Revenue: $%{customdata:,.2f}<extra></extra>',
        customdata=monthly_trends['TotalPrice']
    )
    fig_monthly.update_layout(
        title_font_size=16,
        title_x=0.5,
        title_font_color='#0891b2',
        height=400,
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig_monthly, use_container_width=True)
    
    # Utilization metrics
    col1, col2, col3, col4 = st.columns(4)
    total_appointments = df['TransactionID'].nunique()
    avg_daily_appointments = total_appointments / df['Date'].nunique() if df['Date'].nunique() > 0 else 0
    peak_day = dow_analysis.loc[dow_analysis['TransactionID'].idxmax(), 'Day_of_Week']
    avg_revenue_per_appt = df['TotalPrice'].sum() / total_appointments if total_appointments > 0 else 0
    
    with col1:
        st.metric("Total Appointments", f"{total_appointments:,}")
    with col2:
        st.metric("Avg Daily Appointments", f"{avg_daily_appointments:.1f}")
    with col3:
        st.metric("Peak Day", f"{peak_day}")
    with col4:
        st.metric("Avg Revenue/Appointment", f"${avg_revenue_per_appt:.2f}")

def create_daily_revenue_trend(df):
    """Create daily revenue trend line chart"""
    daily_revenue = df.groupby('Date')['Revenue'].sum().reset_index()
    
    fig = px.line(
        daily_revenue,
        x='Date',
        y='Revenue',
        title='Daily Practice Revenue Trend',
        labels={'Revenue': 'Total Revenue ($)', 'Date': 'Date'},
        line_shape='spline'
    )
    
    fig.update_traces(
        line_color='#0891b2',
        line_width=4,
        hovertemplate='<b>Date:</b> %{x}<br><b>Revenue:</b> $%{y:,.2f}<extra></extra>',
        mode='lines+markers',
        marker=dict(size=6, color='#0891b2')
    )
    
    fig.update_layout(
        title_font_size=16,
        title_x=0.5,
        title_y=0.95,
        title_font_color='#0891b2',
        height=400,
        margin=dict(t=80, b=50, l=80, r=50),
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(tickformat='$,.0f'),
        font_size=11
    )
    
    return fig

def create_optometry_specific_metrics(df):
    """Create optometry practice specific metrics"""
    st.markdown("""
    <div class="optometry-specific">
        <h3>Vision Care Performance Indicators</h3>
        <p>Essential metrics for tracking your practice's optical performance and patient care efficiency</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate KPIs
    conversion_data = analyze_exam_to_purchase_conversion(df)
    retention_data, retention_summary = analyze_patient_retention(df)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Exam-to-Purchase Conversion Rate
        st.metric(
            "Exam-to-Purchase Conversion", 
            f"{conversion_data['conversion_rate']:.1f}%",
            help="Percentage of exam patients who purchase eyewear"
        )
    
    with col2:
        # Average Frame Sale Value
        frame_avg = df[df['Is_Frame_Sale']]['TotalPrice'].mean() if len(df[df['Is_Frame_Sale']]) > 0 else 0
        st.metric(
            "Avg Frame Sale", 
            f"${frame_avg:.2f}",
            help="Average revenue per frame sale"
        )
    
    with col3:
        # Patient Retention Rate
        st.metric(
            "Patient Retention Rate", 
            f"{retention_summary['retention_rate']:.1f}%",
            help="Percentage of patients who return for additional visits"
        )
    
    with col4:
        # Insurance Utilization
        insurance_pct = (len(df[df['InsuranceUsed'] == 'Yes']) / len(df)) * 100 if len(df) > 0 else 0
        st.metric(
            "Insurance Utilization", 
            f"{insurance_pct:.1f}%",
            help="Percentage of transactions using insurance"
        )

def main():
    """Main Clarus optometry dashboard function"""
    # Header
    st.markdown('<h1 class="main-header">Clarus Vision Analytics</h1>', unsafe_allow_html=True)
    
    # Welcome message
    st.markdown("""
    <div style="background-color: #f0f9ff; padding: 1rem; border-radius: 10px; border-left: 5px solid #0891b2; margin-bottom: 2rem;">
        <h4>Welcome to Clarus Vision Analytics</h4>
        <p>Transform your optometry practice data into actionable insights with comprehensive analytics designed for exam-to-purchase conversion, patient retention, and optical retail performance.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - Data options
    st.sidebar.header("Data Source")
    
    data_option = st.sidebar.radio(
        "Choose your data:",
        ["View Sample Data", "Upload Your Data"],
        help="Select how you'd like to explore the Clarus Vision Analytics platform"
    )
    
    if data_option == "Upload Your Data":
        st.sidebar.markdown("**Upload your optometry data:**")
        uploaded_file = st.sidebar.file_uploader(
            "Choose CSV or Excel file",
            type=['csv', 'xlsx'],
            help="Supports CSV and Excel (.xlsx) files"
        )
        st.sidebar.markdown("*Your data stays secure and private.*")
    else:
        uploaded_file = None
    
    # Load data based on user choice
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.sidebar.success("Your data loaded successfully!")
        st.sidebar.info(f"Analyzing {len(df):,} transactions")
    else:
        # Try to load sample data
        sample_data_loaded = False
        
        # Try CSV first
        try:
            df = load_data('synthetic_optometry_data.csv')
            if data_option == "View Sample Data":
                st.sidebar.success("Sample data loaded (CSV)")
            sample_data_loaded = True
        except:
            # Try Excel if CSV fails
            try:
                df = load_data('synthetic_optometry_data.xlsx')
                if data_option == "View Sample Data":
                    st.sidebar.success("Sample data loaded (Excel)")
                sample_data_loaded = True
            except:
                pass
        
        # If neither format works, show error
        if not sample_data_loaded:
            st.error("No sample data available. Please upload your optometry practice data to get started.")
            st.info("""
            **Required columns:** TransactionID, Date, PatientID, ServiceType, FrameBrand, LensType, Quantity, UnitPrice, InsuranceUsed, TotalPrice
            
            **Sample ServiceTypes:** Eye Exam, Frame Sale, Contact Lens Sale, Contact Lens Fitting, Specialty Exam
            """)
            return
    
    if df is None:
        return
    
    # Display data info
    st.sidebar.markdown("---")
    st.sidebar.subheader("Practice Overview")
    st.sidebar.write(f"**Total Transactions:** {len(df):,}")
    st.sidebar.write(f"**Date Range:** {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    st.sidebar.write(f"**Total Revenue:** ${df['Revenue'].sum():,.2f}")
    st.sidebar.write(f"**Unique Patients:** {df['PatientID'].nunique():,}")
    
    # Sidebar filters
    st.sidebar.header("Filter Your Data")
    
    # Month filter
    months_available = sorted(df['Month'].unique())
    month_names = [str(month) for month in months_available]
    
    selected_month_idx = st.sidebar.selectbox(
        "Select Month",
        range(len(months_available)),
        format_func=lambda x: month_names[x],
        index=len(months_available)-1
    )
    selected_month = months_available[selected_month_idx]
    
    # Filter data by selected month
    filtered_df = df[df['Month'] == selected_month]
    
    # Date range filter within the month
    min_date = filtered_df['Date'].min()
    max_date = filtered_df['Date'].max()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['Date'] >= pd.to_datetime(date_range[0])) & 
            (filtered_df['Date'] <= pd.to_datetime(date_range[1]))
        ]
    
    # Key Metrics Dashboard
    st.header("Practice Performance Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_revenue = filtered_df['Revenue'].sum()
        st.metric("Total Revenue", f"${total_revenue:,.2f}")
    
    with col2:
        total_transactions = filtered_df['TransactionID'].nunique()
        st.metric("Total Transactions", f"{total_transactions:,}")
    
    with col3:
        avg_transaction = filtered_df['Revenue'].mean()
        st.metric("Avg Transaction Value", f"${avg_transaction:.2f}")
    
    with col4:
        unique_patients = filtered_df['PatientID'].nunique()
        st.metric("Unique Patients", f"{unique_patients:,}")
    
    # Optometry specific metrics
    create_optometry_specific_metrics(filtered_df)
    
    # Create tabs for different analyses
    tabs = st.tabs([
        "Revenue Trends", 
        "Exam-to-Purchase Conversion", 
        "Top Selling Products", 
        "Patient Retention", 
        "Insurance Analysis",
        "Service Mix",
        "Scheduling & Utilization"
    ])
    
    with tabs[0]:
        st.header("Daily Revenue Trend Analysis")
        st.plotly_chart(create_daily_revenue_trend(filtered_df), use_container_width=True)
    
    with tabs[1]:
        st.header("Exam-to-Purchase Conversion Analysis")
        create_conversion_analysis(filtered_df)
    
    with tabs[2]:
        st.header("Top-Selling Frames & Lens Options")
        create_top_selling_analysis(filtered_df)
    
    with tabs[3]:
        st.header("Patient Recall & Retention Metrics")
        create_retention_analysis(filtered_df)
    
    with tabs[4]:
        st.header("Insurance vs Out-of-Pocket Analysis")
        create_insurance_analysis(filtered_df)
    
    with tabs[5]:
        st.header("Service Mix Analysis")
        create_service_mix_analysis(filtered_df)
    
    with tabs[6]:
        st.header("Appointment Scheduling & Utilization")
        create_appointment_utilization_analysis(filtered_df)
    
    # Raw Data Section
    with st.expander("View Detailed Transaction Data"):
        st.subheader("Recent Practice Transactions")
        display_df = filtered_df.sort_values('Date', ascending=False)
        
        # Show formatted transaction data
        columns_to_show = ['Date', 'PatientID', 'ServiceType', 'FrameBrand', 'LensType', 'Quantity', 'UnitPrice', 'TotalPrice', 'InsuranceUsed']
        formatted_display = display_df[columns_to_show].head(100)
        
        st.dataframe(
            formatted_display.style.format({
                'UnitPrice': '${:.2f}',
                'TotalPrice': '${:.2f}',
                'Date': lambda x: x.strftime('%Y-%m-%d')
            }),
            use_container_width=True,
            height=400
        )
        
        if len(display_df) > 100:
            st.info(f"Showing first 100 of {len(display_df):,} transactions. Download complete report below.")
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Vision Analytics Report (CSV)",
            data=csv,
            file_name=f"clarus_vision_analytics_{selected_month}.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown("**© 2025 Clarus** | Empowering Data-Driven Insights")

if __name__ == "__main__":
    main()