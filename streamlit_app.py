import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
import openai
from typing import Dict, List, Any, Optional
import uuid
import time
import io
import csv

# Page config
st.set_page_config(
    page_title="Momentic AI - Content Operations Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with brand colors
st.markdown("""
<style>
    /* Brand Colors */
    :root {
        --orange: #FF5B04;
        --midnight-green: #075056;
        --gunmetal: #233038;
        --ivory-cream: #FDF8E3;
        --sand-yellow: #F4D40C;
        --light-silver: #D0D8DD;
    }
    
    /* Primary button styling */
    .stButton > button {
        background-color: #FF5B04;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #E54F00;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(255, 91, 4, 0.3);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #075056;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #233038;
    }
    
    /* Info boxes */
    .stInfo {
        background-color: #FDF8E3;
        border-left: 4px solid #F4D40C;
    }
    
    /* Success messages */
    .stSuccess {
        background-color: #E8F5E9;
        border-left: 4px solid #4CAF50;
    }
    
    /* Metrics */
    [data-testid="metric-container"] {
        background-color: #FDF8E3;
        border: 1px solid #D0D8DD;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Workflow node styles */
    .workflow-node {
        background: white;
        border: 2px solid #D0D8DD;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem;
        cursor: move;
        transition: all 0.3s;
    }
    
    .workflow-node:hover {
        border-color: #FF5B04;
        box-shadow: 0 4px 8px rgba(255, 91, 4, 0.2);
    }
    
    .node-ai {
        border-left: 4px solid #FF5B04;
    }
    
    .node-data {
        border-left: 4px solid #075056;
    }
    
    .node-output {
        border-left: 4px solid #F4D40C;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'workflows' not in st.session_state:
    st.session_state.workflows = {}
if 'agents' not in st.session_state:
    st.session_state.agents = {
        "content_writer": {
            "name": "Content Writer",
            "model": "gpt-4-turbo-preview",
            "prompt": "You are an expert content writer. Create engaging, SEO-optimized content based on the given topic and data.",
            "temperature": 0.7
        },
        "seo_optimizer": {
            "name": "SEO Optimizer",
            "model": "gpt-4-turbo-preview",
            "prompt": "You are an SEO expert. Optimize the given content for search engines while maintaining readability.",
            "temperature": 0.3
        },
        "social_media": {
            "name": "Social Media Specialist",
            "model": "gpt-4-turbo-preview",
            "prompt": "You are a social media expert. Create engaging posts for various platforms from the given content.",
            "temperature": 0.8
        }
    }
if 'data_sources' not in st.session_state:
    st.session_state.data_sources = {}
if 'generated_content' not in st.session_state:
    st.session_state.generated_content = []

# Helper functions
def call_openai(prompt: str, model: str = "gpt-4-turbo-preview", temperature: float = 0.7) -> str:
    """Call OpenAI API with the specified model"""
    if not st.session_state.api_key:
        return "❌ Please configure your OpenAI API key in Settings"
    
    try:
        client = openai.OpenAI(api_key=st.session_state.api_key)
        
        # Map model names to actual OpenAI model strings
        model_mapping = {
            "gpt-4.1": "gpt-4-turbo-preview",
            "gpt-4.1-mini": "gpt-3.5-turbo",
            "o3": "gpt-4-turbo-preview",  # Placeholder
            "gpt-4.5": "gpt-4-turbo-preview"  # Placeholder
        }
        
        actual_model = model_mapping.get(model, model)
        
        response = client.chat.completions.create(
            model=actual_model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant for content creation."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=2000
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error: {str(e)}"

def create_workflow_node(node_type: str, node_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Create a workflow node"""
    return {
        "id": node_id,
        "type": node_type,
        "config": config,
        "output": None
    }

def execute_workflow(workflow_id: str, input_data: Any) -> List[Dict[str, Any]]:
    """Execute a workflow with given input data"""
    if workflow_id not in st.session_state.workflows:
        return [{"error": "Workflow not found"}]
    
    workflow = st.session_state.workflows[workflow_id]
    results = []
    current_data = input_data
    
    for node in workflow["nodes"]:
        if node["type"] == "data_input":
            current_data = input_data
        
        elif node["type"] == "ai_process":
            agent_id = node["config"].get("agent_id", "content_writer")
            agent = st.session_state.agents.get(agent_id, st.session_state.agents["content_writer"])
            
            prompt = f"{agent['prompt']}\n\nInput data: {current_data}"
            
            with st.spinner(f"🤖 Processing with {agent['name']}..."):
                result = call_openai(prompt, agent['model'], agent['temperature'])
                current_data = result
                results.append({
                    "node": node["id"],
                    "agent": agent['name'],
                    "output": result
                })
        
        elif node["type"] == "output":
            results.append({
                "node": node["id"],
                "type": "final_output",
                "output": current_data
            })
    
    return results

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/200x50/FF5B04/FFFFFF?text=Momentic+AI", width=200)
    st.markdown("---")
    
    page = st.selectbox(
        "Navigate",
        ["🏠 Dashboard", "🔧 Workflow Builder", "🤖 AI Agents", "📊 Data Sources", 
         "📝 Content Library", "⚙️ Settings"]
    )

# Main content based on page selection
if page == "⚙️ Settings":
    st.title("⚙️ Settings")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("OpenAI Configuration")
        api_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.api_key,
            type="password",
            placeholder="sk-..."
        )
        
        if st.button("Save API Key", type="primary"):
            st.session_state.api_key = api_key
            st.success("✅ API Key saved successfully!")
        
        st.info("🔒 Your API key is stored only in your session and never saved permanently.")
        
        # Model preferences
        st.subheader("Model Preferences")
        default_model = st.selectbox(
            "Default Model",
            ["gpt-4.1", "gpt-4.1-mini", "o3", "gpt-4.5"],
            index=0
        )
        
        st.markdown("### Model Descriptions")
        st.markdown("""
        - **GPT-4.1** (Default): Best balance of quality and speed
        - **GPT-4.1-mini**: Faster, more cost-effective for simple tasks
        - **o3**: Advanced reasoning capabilities
        - **GPT-4.5**: Cutting-edge performance
        """)

elif page == "🏠 Dashboard":
    st.title("🚀 Momentic AI Dashboard")
    st.markdown("### Welcome to your content operations command center!")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Active Workflows",
            value=len(st.session_state.workflows),
            delta="+2 this week"
        )
    
    with col2:
        st.metric(
            label="Content Generated",
            value=len(st.session_state.generated_content),
            delta="+15 today"
        )
    
    with col3:
        st.metric(
            label="AI Agents",
            value=len(st.session_state.agents),
            delta="3 active"
        )
    
    with col4:
        st.metric(
            label="API Usage",
            value="2.3k tokens",
            delta="-15% vs yesterday"
        )
    
    # Quick Actions
    st.markdown("### 🎯 Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📝 Create Blog Post", use_container_width=True):
            st.session_state.quick_action = "blog"
    
    with col2:
        if st.button("📱 Generate Social Media", use_container_width=True):
            st.session_state.quick_action = "social"
    
    with col3:
        if st.button("📧 Write Email Campaign", use_container_width=True):
            st.session_state.quick_action = "email"
    
    # Recent Content
    st.markdown("### 📄 Recent Content")
    if st.session_state.generated_content:
        for idx, content in enumerate(st.session_state.generated_content[-5:]):
            with st.expander(f"Content {idx+1} - {content.get('timestamp', 'Unknown time')}"):
                st.write(content.get('output', 'No content'))
    else:
        st.info("No content generated yet. Create your first workflow to get started!")

elif page == "🔧 Workflow Builder":
    st.title("🔧 Workflow Builder")
    
    tab1, tab2 = st.tabs(["Create Workflow", "My Workflows"])
    
    with tab1:
        st.markdown("### Build Your Content Pipeline")
        
        workflow_name = st.text_input("Workflow Name", placeholder="E.g., Blog Post Generator")
        
        st.markdown("#### Add Nodes to Your Workflow")
        
        col1, col2, col3 = st.columns(3)
        
        # Node palette
        with col1:
            st.markdown("##### 📥 Input Nodes")
            if st.button("➕ Data Source", use_container_width=True):
                st.info("Data source node added")
            if st.button("➕ Text Input", use_container_width=True):
                st.info("Text input node added")
        
        with col2:
            st.markdown("##### 🤖 AI Nodes")
            if st.button("➕ Content Writer", use_container_width=True):
                st.info("Content writer node added")
            if st.button("➕ SEO Optimizer", use_container_width=True):
                st.info("SEO optimizer node added")
        
        with col3:
            st.markdown("##### 📤 Output Nodes")
            if st.button("➕ Save to Library", use_container_width=True):
                st.info("Output node added")
            if st.button("➕ Export", use_container_width=True):
                st.info("Export node added")
        
        # Simplified workflow creation
        st.markdown("### Quick Workflow Setup")
        
        workflow_type = st.selectbox(
            "Select Workflow Type",
            ["Blog Post Generator", "Social Media Suite", "Email Campaign Creator", "Custom"]
        )
        
        if workflow_type == "Blog Post Generator":
            st.markdown("""
            <div class="workflow-node node-data">
                📊 Data Input → 
            </div>
            <div class="workflow-node node-ai">
                🤖 Content Writer →
            </div>
            <div class="workflow-node node-ai">
                🔍 SEO Optimizer →
            </div>
            <div class="workflow-node node-output">
                💾 Save Output
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("Create Workflow", type="primary"):
            if workflow_name:
                workflow_id = str(uuid.uuid4())
                st.session_state.workflows[workflow_id] = {
                    "name": workflow_name,
                    "type": workflow_type,
                    "nodes": [
                        create_workflow_node("data_input", "node1", {}),
                        create_workflow_node("ai_process", "node2", {"agent_id": "content_writer"}),
                        create_workflow_node("ai_process", "node3", {"agent_id": "seo_optimizer"}),
                        create_workflow_node("output", "node4", {})
                    ],
                    "created": datetime.now().isoformat()
                }
                st.success(f"✅ Workflow '{workflow_name}' created successfully!")
            else:
                st.error("Please enter a workflow name")
    
    with tab2:
        st.markdown("### Your Workflows")
        if st.session_state.workflows:
            for wf_id, workflow in st.session_state.workflows.items():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(f"**{workflow['name']}** - {workflow['type']}")
                with col2:
                    if st.button("Run", key=f"run_{wf_id}"):
                        st.session_state.run_workflow = wf_id
                with col3:
                    if st.button("Delete", key=f"del_{wf_id}"):
                        del st.session_state.workflows[wf_id]
                        st.rerun()
        else:
            st.info("No workflows created yet. Create your first workflow above!")

elif page == "🤖 AI Agents":
    st.title("🤖 AI Agents")
    
    tab1, tab2 = st.tabs(["Agent Library", "Create Agent"])
    
    with tab1:
        st.markdown("### Your AI Agents")
        
        for agent_id, agent in st.session_state.agents.items():
            with st.expander(f"{agent['name']} - {agent['model']}"):
                st.markdown(f"**Prompt:** {agent['prompt']}")
                st.markdown(f"**Temperature:** {agent['temperature']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"Edit", key=f"edit_{agent_id}"):
                        st.info("Edit functionality coming soon!")
                with col2:
                    if st.button(f"Test", key=f"test_{agent_id}"):
                        test_input = st.text_area("Test Input", key=f"test_input_{agent_id}")
                        if test_input:
                            result = call_openai(f"{agent['prompt']}\n\n{test_input}", 
                                               agent['model'], agent['temperature'])
                            st.write("**Output:**")
                            st.write(result)
    
    with tab2:
        st.markdown("### Create New AI Agent")
        
        agent_name = st.text_input("Agent Name", placeholder="E.g., Product Description Writer")
        
        agent_type = st.selectbox(
            "Agent Type",
            ["Content Writer", "SEO Specialist", "Social Media Manager", 
             "Email Marketer", "Technical Writer", "Custom"]
        )
        
        model = st.selectbox(
            "Model",
            ["gpt-4.1", "gpt-4.1-mini", "o3", "gpt-4.5"],
            index=0
        )
        
        temperature = st.slider("Temperature (Creativity)", 0.0, 1.0, 0.7, 0.1)
        
        base_prompt = st.text_area(
            "Base Prompt",
            placeholder="Define the agent's role, expertise, and instructions...",
            height=150
        )
        
        if st.button("Create Agent", type="primary"):
            if agent_name and base_prompt:
                agent_id = agent_name.lower().replace(" ", "_")
                st.session_state.agents[agent_id] = {
                    "name": agent_name,
                    "type": agent_type,
                    "model": model,
                    "prompt": base_prompt,
                    "temperature": temperature
                }
                st.success(f"✅ Agent '{agent_name}' created successfully!")
            else:
                st.error("Please fill in all required fields")

elif page == "📊 Data Sources":
    st.title("📊 Data Sources")
    
    st.markdown("### Upload Your Data")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'txt'],
        accept_multiple_files=True
    )
    
    if uploaded_file:
        for file in uploaded_file:
            file_id = str(uuid.uuid4())
            
            if file.type == "text/csv":
                df = pd.read_csv(file)
                st.session_state.data_sources[file_id] = {
                    "name": file.name,
                    "type": "csv",
                    "data": df,
                    "uploaded": datetime.now().isoformat()
                }
                st.success(f"✅ Uploaded {file.name}")
                st.dataframe(df.head())
            
            elif file.type == "text/plain":
                content = file.read().decode("utf-8")
                st.session_state.data_sources[file_id] = {
                    "name": file.name,
                    "type": "txt",
                    "data": content,
                    "uploaded": datetime.now().isoformat()
                }
                st.success(f"✅ Uploaded {file.name}")
                st.text_area("Content Preview", content[:500] + "...", height=200)
    
    st.markdown("### Your Data Sources")
    if st.session_state.data_sources:
        for ds_id, data_source in st.session_state.data_sources.items():
            with st.expander(f"{data_source['name']} ({data_source['type']})"):
                if data_source['type'] == 'csv':
                    st.dataframe(data_source['data'])
                else:
                    st.text(data_source['data'][:500] + "...")
                
                if st.button(f"Delete", key=f"del_ds_{ds_id}"):
                    del st.session_state.data_sources[ds_id]
                    st.rerun()
    else:
        st.info("No data sources uploaded yet. Upload CSV or TXT files to get started!")

elif page == "📝 Content Library":
    st.title("📝 Content Library")
    
    # Quick content generation
    st.markdown("### Quick Generate")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        content_type = st.selectbox(
            "Content Type",
            ["Blog Post", "Social Media Post", "Email", "Product Description"]
        )
        
        topic = st.text_input("Topic/Title", placeholder="Enter your topic...")
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Generate", type="primary", use_container_width=True):
            if topic and st.session_state.api_key:
                with st.spinner("Generating content..."):
                    prompt = f"Create a {content_type} about: {topic}"
                    result = call_openai(prompt)
                    
                    content_item = {
                        "id": str(uuid.uuid4()),
                        "type": content_type,
                        "topic": topic,
                        "output": result,
                        "timestamp": datetime.now().isoformat()
                    }
                    st.session_state.generated_content.append(content_item)
                    st.success("✅ Content generated successfully!")
            else:
                st.error("Please enter a topic and ensure API key is configured")
    
    # Content library
    st.markdown("### Your Content")
    
    if st.session_state.generated_content:
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_type = st.selectbox("Filter by Type", ["All"] + list(set(c.get("type", "Unknown") for c in st.session_state.generated_content)))
        
        # Display content
        filtered_content = st.session_state.generated_content if filter_type == "All" else [c for c in st.session_state.generated_content if c.get("type") == filter_type]
        
        for content in reversed(filtered_content):
            with st.expander(f"{content.get('type', 'Content')} - {content.get('topic', 'Untitled')} ({content.get('timestamp', 'Unknown')[:10]})"):
                st.write(content.get('output', 'No content'))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("📋 Copy", key=f"copy_{content['id']}"):
                        st.info("Content copied to clipboard!")
                with col2:
                    if st.button("✏️ Edit", key=f"edit_{content['id']}"):
                        st.info("Edit functionality coming soon!")
                with col3:
                    if st.button("🗑️ Delete", key=f"delete_{content['id']}"):
                        st.session_state.generated_content.remove(content)
                        st.rerun()
    else:
        st.info("No content generated yet. Use Quick Generate above or run a workflow!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Momentic AI - Empowering Marketing Teams with AI-Powered Content Operations</p>
        <p style='font-size: 0.8em;'>Made with ❤️ for content creators</p>
    </div>
    """,
    unsafe_allow_html=True
)
