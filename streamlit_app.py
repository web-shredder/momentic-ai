import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
import openai
from typing import Dict, List, Any, Optional, Tuple
import uuid
import time
import io
import csv
import streamlit.components.v1 as components
from dataclasses import dataclass
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict
import re

# Page config
st.set_page_config(
    page_title="Momentic AI - Content Operations Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Model mappings
MODEL_MAP = {
    "4.1": "gpt-4.1-2025-04-14",
    "4o": "gpt-4o-2024-08-06", 
    "o3": "o3-2025-04-16"
}

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
    
    /* Node styles */
    .node-container {
        background: white;
        border: 2px solid #D0D8DD;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        position: relative;
        cursor: move;
    }
    
    .node-data { border-left: 4px solid #075056; }
    .node-ai { border-left: 4px solid #FF5B04; }
    .node-output { border-left: 4px solid #F4D40C; }
    .node-logic { border-left: 4px solid #8B5CF6; }
    .node-human { border-left: 4px solid #10B981; }
    
    /* Workflow canvas */
    .workflow-canvas {
        background: #f8f9fa;
        border: 2px dashed #D0D8DD;
        border-radius: 8px;
        padding: 2rem;
        min-height: 500px;
        position: relative;
        overflow: auto;
    }
    
    /* Execution status */
    .execution-success { color: #10B981; }
    .execution-error { color: #EF4444; }
    .execution-pending { color: #F59E0B; }
    
    /* Analytics cards */
    .metric-card {
        background: #FDF8E3;
        border: 1px solid #D0D8DD;
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Progress bar */
    .progress-bar {
        background: #E5E7EB;
        height: 8px;
        border-radius: 4px;
        overflow: hidden;
    }
    
    .progress-fill {
        background: #FF5B04;
        height: 100%;
        transition: width 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'workflows' not in st.session_state:
    st.session_state.workflows = {}
if 'current_workflow' not in st.session_state:
    st.session_state.current_workflow = {
        'id': str(uuid.uuid4()),
        'name': '',
        'nodes': [],
        'connections': [],
        'variables': {}
    }
if 'workflow_templates' not in st.session_state:
    st.session_state.workflow_templates = {
        'blog_post': {
            'name': 'Blog Post Generator',
            'nodes': [
                {'id': 'n1', 'type': 'input', 'subtype': 'data_source', 'x': 100, 'y': 200},
                {'id': 'n2', 'type': 'ai', 'subtype': 'content_writer', 'x': 300, 'y': 200},
                {'id': 'n3', 'type': 'ai', 'subtype': 'seo_optimizer', 'x': 500, 'y': 200},
                {'id': 'n4', 'type': 'output', 'subtype': 'save', 'x': 700, 'y': 200}
            ],
            'connections': [
                {'from': 'n1', 'to': 'n2'},
                {'from': 'n2', 'to': 'n3'},
                {'from': 'n3', 'to': 'n4'}
            ]
        },
        'social_media': {
            'name': 'Social Media Suite',
            'nodes': [
                {'id': 'n1', 'type': 'input', 'subtype': 'text_input', 'x': 100, 'y': 100},
                {'id': 'n2', 'type': 'ai', 'subtype': 'content_writer', 'x': 300, 'y': 100},
                {'id': 'n3', 'type': 'logic', 'subtype': 'split', 'x': 500, 'y': 100},
                {'id': 'n4', 'type': 'transform', 'subtype': 'twitter', 'x': 700, 'y': 50},
                {'id': 'n5', 'type': 'transform', 'subtype': 'linkedin', 'x': 700, 'y': 150},
                {'id': 'n6', 'type': 'output', 'subtype': 'save_multi', 'x': 900, 'y': 100}
            ],
            'connections': [
                {'from': 'n1', 'to': 'n2'},
                {'from': 'n2', 'to': 'n3'},
                {'from': 'n3', 'to': 'n4'},
                {'from': 'n3', 'to': 'n5'},
                {'from': 'n4', 'to': 'n6'},
                {'from': 'n5', 'to': 'n6'}
            ]
        }
    }
if 'agents' not in st.session_state:
    st.session_state.agents = {
        "content_writer": {
            "name": "Content Writer",
            "model": "4.1",
            "prompt": "You are an expert content writer. Create engaging, SEO-optimized content based on the given topic and data.",
            "temperature": 0.7,
            "max_tokens": 2000
        },
        "seo_optimizer": {
            "name": "SEO Optimizer", 
            "model": "4o",
            "prompt": "You are an SEO expert. Optimize the given content for search engines while maintaining readability. Include relevant keywords and meta descriptions.",
            "temperature": 0.3,
            "max_tokens": 2000
        },
        "social_media_expert": {
            "name": "Social Media Expert",
            "model": "4.1",
            "prompt": "You are a social media expert. Create engaging posts optimized for different platforms.",
            "temperature": 0.8,
            "max_tokens": 500
        }
    }
if 'data_sources' not in st.session_state:
    st.session_state.data_sources = {}
if 'knowledge_bases' not in st.session_state:
    st.session_state.knowledge_bases = {}
if 'generated_content' not in st.session_state:
    st.session_state.generated_content = []
if 'execution_history' not in st.session_state:
    st.session_state.execution_history = []
if 'user_role' not in st.session_state:
    st.session_state.user_role = 'admin'  # admin, editor, viewer
if 'ab_tests' not in st.session_state:
    st.session_state.ab_tests = {}

# Data classes for structured data
@dataclass
class WorkflowNode:
    id: str
    type: str
    subtype: str
    config: Dict[str, Any]
    x: float = 0
    y: float = 0
    
@dataclass
class Connection:
    from_node: str
    to_node: str
    condition: Optional[str] = None

@dataclass
class ExecutionResult:
    workflow_id: str
    timestamp: datetime
    status: str  # success, error, cancelled
    duration: float
    token_usage: int
    cost: float
    results: List[Dict[str, Any]]

# Helper functions
def call_openai(prompt: str, model: str = "4.1", temperature: float = 0.7, max_tokens: int = 2000) -> Tuple[str, int, float]:
    """Call OpenAI API with the specified model and return response, tokens, and cost"""
    if not st.session_state.api_key:
        return "❌ Please configure your OpenAI API key in Settings", 0, 0.0
    
    try:
        client = openai.OpenAI(api_key=st.session_state.api_key)
        
        # Use the actual model names from MODEL_MAP
        actual_model = MODEL_MAP.get(model, model)
        
        response = client.chat.completions.create(
            model=actual_model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant for content creation."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Calculate tokens and cost
        tokens = response.usage.total_tokens
        # Approximate cost calculation (adjust based on actual pricing)
        cost_per_1k = {"4.1": 0.01, "4o": 0.005, "o3": 0.015}
        cost = (tokens / 1000) * cost_per_1k.get(model, 0.01)
        
        return response.choices[0].message.content, tokens, cost
    except Exception as e:
        return f"❌ Error: {str(e)}", 0, 0.0

def execute_node(node: Dict, input_data: Any, variables: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """Execute a single node and return output and updated variables"""
    node_type = node['type']
    subtype = node['subtype']
    config = node.get('config', {})
    
    if node_type == 'input':
        if subtype == 'data_source':
            ds_id = config.get('data_source_id')
            if ds_id and ds_id in st.session_state.data_sources:
                return st.session_state.data_sources[ds_id]['data'], variables
        elif subtype == 'text_input':
            return config.get('text', ''), variables
        elif subtype == 'variable':
            var_name = config.get('variable_name', 'default')
            return variables.get(var_name, ''), variables
            
    elif node_type == 'ai':
        agent_id = config.get('agent_id', 'content_writer')
        agent = st.session_state.agents.get(agent_id)
        
        if agent:
            # Include knowledge base if specified
            kb_id = config.get('knowledge_base_id')
            kb_context = ""
            if kb_id and kb_id in st.session_state.knowledge_bases:
                kb_context = f"\n\nContext from knowledge base:\n{st.session_state.knowledge_bases[kb_id]['content']}"
            
            # Variable substitution in prompt
            prompt = agent['prompt']
            for var_name, var_value in variables.items():
                prompt = prompt.replace(f"{{{var_name}}}", str(var_value))
            
            full_prompt = f"{prompt}{kb_context}\n\nInput: {input_data}"
            result, tokens, cost = call_openai(
                full_prompt, 
                agent['model'], 
                agent['temperature'],
                agent.get('max_tokens', 2000)
            )
            
            # Track usage
            st.session_state.execution_history.append({
                'timestamp': datetime.now(),
                'agent': agent['name'],
                'tokens': tokens,
                'cost': cost
            })
            
            return result, variables
            
    elif node_type == 'transform':
        if subtype == 'summarize':
            prompt = f"Summarize the following content in {config.get('length', '3')} sentences:\n\n{input_data}"
            result, _, _ = call_openai(prompt, temperature=0.3)
            return result, variables
        elif subtype == 'translate':
            target_lang = config.get('language', 'Spanish')
            prompt = f"Translate the following to {target_lang}:\n\n{input_data}"
            result, _, _ = call_openai(prompt, temperature=0.3)
            return result, variables
        elif subtype == 'tone':
            tone = config.get('tone', 'professional')
            prompt = f"Rewrite the following content in a {tone} tone:\n\n{input_data}"
            result, _, _ = call_openai(prompt, temperature=0.7)
            return result, variables
        elif subtype == 'twitter':
            prompt = f"Convert this into a Twitter thread (max 5 tweets, 280 chars each):\n\n{input_data}"
            result, _, _ = call_openai(prompt, temperature=0.8)
            return result, variables
        elif subtype == 'linkedin':
            prompt = f"Convert this into a LinkedIn post (professional, engaging, with hashtags):\n\n{input_data}"
            result, _, _ = call_openai(prompt, temperature=0.7)
            return result, variables
            
    elif node_type == 'logic':
        if subtype == 'condition':
            condition = config.get('condition', '')
            # Simple condition evaluation (can be enhanced)
            if 'contains' in condition:
                keyword = config.get('keyword', '')
                if keyword.lower() in str(input_data).lower():
                    return input_data, {**variables, 'condition_met': True}
                else:
                    return None, {**variables, 'condition_met': False}
        elif subtype == 'loop':
            # For batch processing
            if isinstance(input_data, list):
                results = []
                for item in input_data:
                    # Process each item (simplified - in real implementation would process through connected nodes)
                    results.append(item)
                return results, variables
            return input_data, variables
        elif subtype == 'merge':
            # Merge multiple inputs (simplified)
            return input_data, variables
        elif subtype == 'split':
            # Split for parallel processing
            return input_data, variables
            
    elif node_type == 'human':
        if subtype == 'review':
            # In a real implementation, this would pause for human input
            return f"[Awaiting human review]\n{input_data}", {**variables, 'human_approved': True}
        elif subtype == 'approval':
            return f"[Awaiting approval]\n{input_data}", variables
            
    elif node_type == 'output':
        if subtype == 'save':
            # Save to content library
            content_item = {
                'id': str(uuid.uuid4()),
                'content': input_data,
                'timestamp': datetime.now().isoformat(),
                'workflow': st.session_state.current_workflow.get('name', 'Unnamed')
            }
            st.session_state.generated_content.append(content_item)
            return input_data, variables
        elif subtype == 'export_csv':
            # Export functionality would go here
            return input_data, variables
        elif subtype == 'webhook':
            # Webhook functionality would go here
            return input_data, variables
    
    return input_data, variables

def execute_workflow(workflow: Dict, initial_input: Any = None) -> ExecutionResult:
    """Execute an entire workflow and return results"""
    start_time = time.time()
    results = []
    variables = workflow.get('variables', {})
    total_tokens = 0
    total_cost = 0.0
    status = 'success'
    
    try:
        # Build adjacency list for node connections
        connections = defaultdict(list)
        for conn in workflow.get('connections', []):
            connections[conn['from']].append(conn['to'])
        
        # Find start nodes (nodes with no incoming connections)
        all_nodes = {node['id'] for node in workflow['nodes']}
        has_incoming = {conn['to'] for conn in workflow.get('connections', [])}
        start_nodes = all_nodes - has_incoming
        
        # Execute nodes in order (simplified - real implementation would handle parallel execution)
        executed = set()
        to_execute = list(start_nodes)
        node_outputs = {}
        
        while to_execute:
            node_id = to_execute.pop(0)
            if node_id in executed:
                continue
                
            # Find the node
            node = next((n for n in workflow['nodes'] if n['id'] == node_id), None)
            if not node:
                continue
            
            # Get input from previous node(s)
            if node_id in start_nodes:
                input_data = initial_input
            else:
                # Get input from connected nodes
                input_data = None
                for conn in workflow.get('connections', []):
                    if conn['to'] == node_id and conn['from'] in node_outputs:
                        input_data = node_outputs[conn['from']]
                        break
            
            # Execute the node
            output, variables = execute_node(node, input_data, variables)
            node_outputs[node_id] = output
            executed.add(node_id)
            
            # Add connected nodes to execution queue
            for next_node in connections[node_id]:
                if next_node not in executed:
                    to_execute.append(next_node)
            
            results.append({
                'node_id': node_id,
                'node_type': node['type'],
                'node_subtype': node['subtype'],
                'output': output
            })
            
    except Exception as e:
        status = 'error'
        results.append({'error': str(e)})
    
    duration = time.time() - start_time
    
    return ExecutionResult(
        workflow_id=workflow.get('id', 'unknown'),
        timestamp=datetime.now(),
        status=status,
        duration=duration,
        token_usage=total_tokens,
        cost=total_cost,
        results=results
    )

def render_workflow_canvas():
    """Render the visual workflow canvas using HTML/JavaScript"""
    canvas_html = """
    <div id="workflow-canvas" style="width: 100%; height: 600px; border: 2px solid #D0D8DD; border-radius: 8px; background: #f8f9fa; position: relative; overflow: auto;">
        <div style="position: absolute; top: 10px; right: 10px; z-index: 100;">
            <button onclick="zoomIn()" style="padding: 5px 10px; margin: 2px;">➕</button>
            <button onclick="zoomOut()" style="padding: 5px 10px; margin: 2px;">➖</button>
            <button onclick="fitToScreen()" style="padding: 5px 10px; margin: 2px;">⬜</button>
        </div>
        <svg id="connections" style="position: absolute; width: 100%; height: 100%; pointer-events: none;">
            <!-- Connections will be drawn here -->
        </svg>
        <div id="nodes-container" style="position: relative; width: 100%; height: 100%;">
            <!-- Nodes will be rendered here -->
        </div>
    </div>
    
    <script>
    let scale = 1;
    let selectedNode = null;
    
    function zoomIn() {
        scale *= 1.2;
        updateZoom();
    }
    
    function zoomOut() {
        scale /= 1.2;
        updateZoom();
    }
    
    function fitToScreen() {
        scale = 1;
        updateZoom();
    }
    
    function updateZoom() {
        const container = document.getElementById('nodes-container');
        container.style.transform = `scale(${scale})`;
        container.style.transformOrigin = 'top left';
    }
    
    function selectNode(nodeId) {
        selectedNode = nodeId;
        // Send selection to Streamlit
        window.parent.postMessage({type: 'nodeSelected', nodeId: nodeId}, '*');
    }
    
    // Render nodes
    const nodes = %s;
    const connections = %s;
    
    nodes.forEach(node => {
        const nodeEl = document.createElement('div');
        nodeEl.id = `node-${node.id}`;
        nodeEl.className = `node-container node-${node.type}`;
        nodeEl.style.position = 'absolute';
        nodeEl.style.left = `${node.x || 100}px`;
        nodeEl.style.top = `${node.y || 100}px`;
        nodeEl.style.width = '150px';
        nodeEl.style.cursor = 'move';
        nodeEl.innerHTML = `
            <div style="font-weight: bold;">${node.type}</div>
            <div style="font-size: 0.9em; color: #666;">${node.subtype}</div>
        `;
        nodeEl.onclick = () => selectNode(node.id);
        
        // Make draggable
        let isDragging = false;
        let startX, startY, initialX, initialY;
        
        nodeEl.onmousedown = (e) => {
            isDragging = true;
            startX = e.clientX;
            startY = e.clientY;
            initialX = nodeEl.offsetLeft;
            initialY = nodeEl.offsetTop;
            e.preventDefault();
        };
        
        document.onmousemove = (e) => {
            if (!isDragging) return;
            const dx = e.clientX - startX;
            const dy = e.clientY - startY;
            nodeEl.style.left = `${initialX + dx / scale}px`;
            nodeEl.style.top = `${initialY + dy / scale}px`;
            updateConnections();
        };
        
        document.onmouseup = () => {
            isDragging = false;
        };
        
        document.getElementById('nodes-container').appendChild(nodeEl);
    });
    
    // Draw connections
    function updateConnections() {
        const svg = document.getElementById('connections');
        svg.innerHTML = '';
        
        connections.forEach(conn => {
            const fromNode = document.getElementById(`node-${conn.from}`);
            const toNode = document.getElementById(`node-${conn.to}`);
            
            if (fromNode && toNode) {
                const x1 = fromNode.offsetLeft + fromNode.offsetWidth;
                const y1 = fromNode.offsetTop + fromNode.offsetHeight / 2;
                const x2 = toNode.offsetLeft;
                const y2 = toNode.offsetTop + toNode.offsetHeight / 2;
                
                const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                const d = `M ${x1} ${y1} C ${x1 + 50} ${y1}, ${x2 - 50} ${y2}, ${x2} ${y2}`;
                path.setAttribute('d', d);
                path.setAttribute('stroke', '#FF5B04');
                path.setAttribute('stroke-width', '2');
                path.setAttribute('fill', 'none');
                path.setAttribute('marker-end', 'url(#arrowhead)');
                
                svg.appendChild(path);
            }
        });
    }
    
    // Add arrowhead marker
    const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    const marker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
    marker.setAttribute('id', 'arrowhead');
    marker.setAttribute('markerWidth', '10');
    marker.setAttribute('markerHeight', '10');
    marker.setAttribute('refX', '9');
    marker.setAttribute('refY', '3');
    marker.setAttribute('orient', 'auto');
    
    const arrow = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
    arrow.setAttribute('points', '0 0, 10 3, 0 6');
    arrow.setAttribute('fill', '#FF5B04');
    
    marker.appendChild(arrow);
    defs.appendChild(marker);
    document.getElementById('connections').appendChild(defs);
    
    updateConnections();
    </script>
    """ % (json.dumps(st.session_state.current_workflow['nodes']), json.dumps(st.session_state.current_workflow['connections']))
    
    components.html(canvas_html, height=650)

def add_node_to_workflow(node_type: str, subtype: str, x: float = None, y: float = None) -> str:
    """Add a node to the current workflow with position"""
    node_id = f"node_{str(uuid.uuid4())[:8]}"
    
    # Calculate position if not provided
    if x is None or y is None:
        existing_nodes = st.session_state.current_workflow['nodes']
        if existing_nodes:
            # Place new node to the right of the last node
            last_node = existing_nodes[-1]
            x = last_node.get('x', 100) + 200
            y = last_node.get('y', 100)
        else:
            x, y = 100, 100
    
    node = {
        'id': node_id,
        'type': node_type,
        'subtype': subtype,
        'config': {},
        'x': x,
        'y': y
    }
    
    st.session_state.current_workflow['nodes'].append(node)
    return node_id

def add_connection(from_node: str, to_node: str, condition: str = None):
    """Add a connection between two nodes"""
    connection = {
        'from': from_node,
        'to': to_node
    }
    if condition:
        connection['condition'] = condition
    
    st.session_state.current_workflow['connections'].append(connection)

def render_analytics_dashboard():
    """Render analytics dashboard with execution metrics"""
    if not st.session_state.execution_history:
        st.info("No execution data available yet. Run some workflows to see analytics!")
        return
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(st.session_state.execution_history)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_executions = len(df)
        st.metric("Total Executions", total_executions)
    
    with col2:
        total_tokens = df['tokens'].sum() if 'tokens' in df.columns else 0
        st.metric("Total Tokens Used", f"{total_tokens:,}")
    
    with col3:
        total_cost = df['cost'].sum() if 'cost' in df.columns else 0
        st.metric("Total Cost", f"${total_cost:.2f}")
    
    with col4:
        avg_tokens = df['tokens'].mean() if 'tokens' in df.columns and len(df) > 0 else 0
        st.metric("Avg Tokens/Execution", f"{avg_tokens:.0f}")
    
    # Token usage over time
    if 'timestamp' in df.columns and 'tokens' in df.columns:
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        daily_tokens = df.groupby('date')['tokens'].sum().reset_index()
        
        fig = px.line(daily_tokens, x='date', y='tokens', title='Token Usage Over Time',
                     labels={'tokens': 'Tokens Used', 'date': 'Date'})
        fig.update_traces(line_color='#FF5B04')
        st.plotly_chart(fig, use_container_width=True)
    
    # Agent usage distribution
    if 'agent' in df.columns:
        agent_usage = df['agent'].value_counts()
        fig = px.pie(values=agent_usage.values, names=agent_usage.index, 
                    title='Agent Usage Distribution',
                    color_discrete_sequence=['#FF5B04', '#075056', '#F4D40C', '#233038'])
        st.plotly_chart(fig, use_container_width=True)

# Sidebar
with st.sidebar:
    st.markdown("# 🚀 Momentic AI")
    st.markdown("---")
    
    # User role indicator
    st.markdown(f"**Role:** {st.session_state.user_role.title()}")
    
    page = st.selectbox(
        "Navigate",
        ["🏠 Dashboard", "🔧 Workflow Builder", "🤖 AI Agents", "📊 Data Sources", 
         "📚 Knowledge Base", "📝 Content Library", "📈 Analytics", "🧪 A/B Testing", 
         "👥 Team", "⚙️ Settings"]
    )

# Main content based on page selection
if page == "⚙️ Settings":
    st.title("⚙️ Settings")
    
    tabs = st.tabs(["API Configuration", "Model Preferences", "User Settings"])
    
    with tabs[0]:
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
        
    with tabs[1]:
        st.subheader("Model Preferences")
        
        default_model = st.selectbox(
            "Default Model",
            options=list(MODEL_MAP.keys()),
            index=0
        )
        
        st.markdown("### Available Models")
        for model_key, model_name in MODEL_MAP.items():
            st.markdown(f"- **{model_key}**: `{model_name}`")
            
        # Model performance comparison
        st.markdown("### Model Comparison")
        model_data = {
            'Model': ['4.1', '4o', 'o3'],
            'Speed': [8, 10, 6],
            'Quality': [10, 8, 10],
            'Cost': [7, 9, 5],
            'Context Window': [128000, 128000, 200000]
        }
        
        df_models = pd.DataFrame(model_data)
        st.dataframe(df_models, use_container_width=True)
        
    with tabs[2]:
        st.subheader("User Preferences")
        
        theme = st.selectbox("Theme", ["Light", "Dark", "Auto"])
        notifications = st.checkbox("Enable notifications", value=True)
        auto_save = st.checkbox("Auto-save workflows", value=True)
        
        if st.button("Save Preferences"):
            st.success("✅ Preferences saved!")

elif page == "🏠 Dashboard":
    st.title("🚀 Momentic AI Dashboard")
    st.markdown("### Your Content Command Center")
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        workflows_count = len(st.session_state.workflows)
        st.markdown(f"""
        <div class="metric-card">
            <h3>{workflows_count}</h3>
            <p>Active Workflows</p>
            <small>↑ 2 this week</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        content_count = len(st.session_state.generated_content)
        st.markdown(f"""
        <div class="metric-card">
            <h3>{content_count}</h3>
            <p>Content Pieces</p>
            <small>↑ 15 today</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        agents_count = len(st.session_state.agents)
        st.markdown(f"""
        <div class="metric-card">
            <h3>{agents_count}</h3>
            <p>AI Agents</p>
            <small>3 active</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total_tokens = sum(h.get('tokens', 0) for h in st.session_state.execution_history)
        st.markdown(f"""
        <div class="metric-card">
            <h3>{total_tokens:,}</h3>
            <p>Tokens Used</p>
            <small>↓ 15% vs yesterday</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick actions
    st.markdown("### 🎯 Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📝 New Blog Post", use_container_width=True):
            # Load blog post template
            template = st.session_state.workflow_templates['blog_post']
            st.session_state.current_workflow = {
                'id': str(uuid.uuid4()),
                'name': template['name'],
                'nodes': template['nodes'].copy(),
                'connections': template['connections'].copy(),
                'variables': {}
            }
            st.switch_page("pages/workflow_builder.py")
    
    with col2:
        if st.button("📱 Social Media Suite", use_container_width=True):
            template = st.session_state.workflow_templates['social_media']
            st.session_state.current_workflow = {
                'id': str(uuid.uuid4()),
                'name': template['name'],
                'nodes': template['nodes'].copy(),
                'connections': template['connections'].copy(),
                'variables': {}
            }
            st.switch_page("pages/workflow_builder.py")
    
    with col3:
        if st.button("📧 Email Campaign", use_container_width=True):
            st.info("Email campaign template coming soon!")
    
    with col4:
        if st.button("🔍 SEO Content", use_container_width=True):
            st.info("SEO content template coming soon!")
    
    # Recent activity
    st.markdown("### 📊 Recent Activity")
    
    if st.session_state.execution_history:
        recent_executions = st.session_state.execution_history[-5:]
        for exec in reversed(recent_executions):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"**{exec.get('agent', 'Unknown')}** - {exec['timestamp'].strftime('%Y-%m-%d %H:%M')}")
            with col2:
                st.write(f"🪙 {exec.get('tokens', 0)} tokens")
            with col3:
                st.write(f"💰 ${exec.get('cost', 0):.3f}")
    else:
        st.info("No recent activity. Start creating content!")

elif page == "🔧 Workflow Builder":
    st.title("🔧 Workflow Builder")
    
    # Workflow name and controls
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        workflow_name = st.text_input(
            "Workflow Name", 
            value=st.session_state.current_workflow.get('name', ''),
            placeholder="E.g., Blog Post Generator"
        )
        st.session_state.current_workflow['name'] = workflow_name
    
    with col2:
        if st.button("💾 Save Workflow"):
            if workflow_name and st.session_state.current_workflow['nodes']:
                workflow_id = st.session_state.current_workflow.get('id', str(uuid.uuid4()))
                st.session_state.workflows[workflow_id] = {
                    **st.session_state.current_workflow,
                    'saved_at': datetime.now().isoformat()
                }
                st.success("✅ Workflow saved!")
            else:
                st.error("Please name your workflow and add nodes")
    
    with col3:
        if st.button("🗑️ Clear Workflow"):
            st.session_state.current_workflow = {
                'id': str(uuid.uuid4()),
                'name': '',
                'nodes': [],
                'connections': [],
                'variables': {}
            }
            st.rerun()
    
    # Main workflow area
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Visual Builder", "📝 Node List", "🔧 Variables", "📚 Templates"])
    
    with tab1:
        st.markdown("### Workflow Canvas")
        
        # Node palette
        with st.expander("🎨 Node Palette", expanded=True):
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.markdown("**📥 Input**")
                if st.button("Data Source", use_container_width=True, key="add_data_source"):
                    add_node_to_workflow('input', 'data_source')
                    st.rerun()
                if st.button("Text Input", use_container_width=True, key="add_text_input"):
                    add_node_to_workflow('input', 'text_input')
                    st.rerun()
                if st.button("Variable", use_container_width=True, key="add_variable"):
                    add_node_to_workflow('input', 'variable')
                    st.rerun()
            
            with col2:
                st.markdown("**🤖 AI**")
                if st.button("AI Agent", use_container_width=True, key="add_ai_agent"):
                    add_node_to_workflow('ai', 'agent')
                    st.rerun()
            
            with col3:
                st.markdown("**🔄 Transform**")
                if st.button("Summarize", use_container_width=True, key="add_summarize"):
                    add_node_to_workflow('transform', 'summarize')
                    st.rerun()
                if st.button("Translate", use_container_width=True, key="add_translate"):
                    add_node_to_workflow('transform', 'translate')
                    st.rerun()
                if st.button("Change Tone", use_container_width=True, key="add_tone"):
                    add_node_to_workflow('transform', 'tone')
                    st.rerun()
            
            with col4:
                st.markdown("**🧠 Logic**")
                if st.button("Condition", use_container_width=True, key="add_condition"):
                    add_node_to_workflow('logic', 'condition')
                    st.rerun()
                if st.button("Loop", use_container_width=True, key="add_loop"):
                    add_node_to_workflow('logic', 'loop')
                    st.rerun()
                if st.button("Split", use_container_width=True, key="add_split"):
                    add_node_to_workflow('logic', 'split')
                    st.rerun()
                if st.button("Merge", use_container_width=True, key="add_merge"):
                    add_node_to_workflow('logic', 'merge')
                    st.rerun()
            
            with col5:
                st.markdown("**📤 Output**")
                if st.button("Save", use_container_width=True, key="add_save"):
                    add_node_to_workflow('output', 'save')
                    st.rerun()
                if st.button("Export CSV", use_container_width=True, key="add_export_csv"):
                    add_node_to_workflow('output', 'export_csv')
                    st.rerun()
                if st.button("Webhook", use_container_width=True, key="add_webhook"):
                    add_node_to_workflow('output', 'webhook')
                    st.rerun()
        
        # Visual canvas
        render_workflow_canvas()
        
        # Execution controls
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            if st.button("▶️ Execute", type="primary", use_container_width=True):
                if st.session_state.current_workflow['nodes']:
                    with st.spinner("Executing workflow..."):
                        result = execute_workflow(st.session_state.current_workflow)
                        
                        if result.status == 'success':
                            st.success(f"✅ Workflow executed successfully in {result.duration:.2f}s")
                            
                            # Show results
                            with st.expander("📊 Execution Results", expanded=True):
                                for res in result.results:
                                    if 'error' not in res:
                                        st.markdown(f"**{res['node_type']} - {res['node_subtype']}**")
                                        if res['output']:
                                            st.write(res['output'][:500] + "..." if len(str(res['output'])) > 500 else res['output'])
                                        st.markdown("---")
                        else:
                            st.error("❌ Workflow execution failed")
                else:
                    st.warning("Add nodes to your workflow first")
        
        with col2:
            if st.button("🐛 Debug", use_container_width=True):
                st.info("Debug mode coming soon!")
        
    with tab2:
        st.markdown("### Node Configuration")
        
        if st.session_state.current_workflow['nodes']:
            for idx, node in enumerate(st.session_state.current_workflow['nodes']):
                with st.expander(f"**{idx+1}. {node['type'].title()} - {node['subtype'].replace('_', ' ').title()}**", expanded=True):
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        # Node-specific configuration
                        if node['type'] == 'input' and node['subtype'] == 'data_source':
                            data_sources = list(st.session_state.data_sources.keys())
                            if data_sources:
                                selected_ds = st.selectbox(
                                    "Select Data Source",
                                    options=[''] + data_sources,
                                    index=0 if not node['config'].get('data_source_id') else data_sources.index(node['config'].get('data_source_id')) + 1,
                                    key=f"ds_{node['id']}"
                                )
                                if selected_ds:
                                    node['config']['data_source_id'] = selected_ds
                            else:
                                st.info("No data sources available. Upload files in Data Sources.")
                        
                        elif node['type'] == 'input' and node['subtype'] == 'text_input':
                            text = st.text_area(
                                "Input Text",
                                value=node['config'].get('text', ''),
                                key=f"text_{node['id']}",
                                height=100
                            )
                            node['config']['text'] = text
                        
                        elif node['type'] == 'input' and node['subtype'] == 'variable':
                            var_name = st.text_input(
                                "Variable Name",
                                value=node['config'].get('variable_name', ''),
                                key=f"var_{node['id']}"
                            )
                            node['config']['variable_name'] = var_name
                        
                        elif node['type'] == 'ai':
                            col_a, col_b = st.columns(2)
                            with col_a:
                                agents = list(st.session_state.agents.keys())
                                selected_agent = st.selectbox(
                                    "AI Agent",
                                    options=agents,
                                    index=0 if not node['config'].get('agent_id') else agents.index(node['config'].get('agent_id')),
                                    key=f"agent_{node['id']}"
                                )
                                node['config']['agent_id'] = selected_agent
                            
                            with col_b:
                                kb_options = ['None'] + list(st.session_state.knowledge_bases.keys())
                                kb_idx = 0
                                if node['config'].get('knowledge_base_id') in st.session_state.knowledge_bases:
                                    kb_idx = kb_options.index(node['config'].get('knowledge_base_id'))
                                
                                selected_kb = st.selectbox(
                                    "Knowledge Base",
                                    options=kb_options,
                                    index=kb_idx,
                                    key=f"kb_{node['id']}"
                                )
                                if selected_kb != 'None':
                                    node['config']['knowledge_base_id'] = selected_kb
                                elif 'knowledge_base_id' in node['config']:
                                    del node['config']['knowledge_base_id']
                        
                        elif node['type'] == 'transform':
                            if node['subtype'] == 'summarize':
                                length = st.selectbox(
                                    "Summary Length",
                                    options=['1', '3', '5', '10'],
                                    index=1,
                                    key=f"summ_{node['id']}"
                                )
                                node['config']['length'] = length
                            elif node['subtype'] == 'translate':
                                language = st.selectbox(
                                    "Target Language",
                                    options=['Spanish', 'French', 'German', 'Italian', 'Portuguese', 'Chinese', 'Japanese'],
                                    key=f"lang_{node['id']}"
                                )
                                node['config']['language'] = language
                            elif node['subtype'] == 'tone':
                                tone = st.selectbox(
                                    "Target Tone",
                                    options=['professional', 'casual', 'friendly', 'formal', 'humorous', 'serious'],
                                    key=f"tone_{node['id']}"
                                )
                                node['config']['tone'] = tone
                        
                        elif node['type'] == 'logic' and node['subtype'] == 'condition':
                            condition = st.selectbox(
                                "Condition Type",
                                options=['contains', 'equals', 'greater_than', 'less_than'],
                                key=f"cond_{node['id']}"
                            )
                            node['config']['condition'] = condition
                            
                            if condition == 'contains':
                                keyword = st.text_input(
                                    "Keyword",
                                    value=node['config'].get('keyword', ''),
                                    key=f"keyword_{node['id']}"
                                )
                                node['config']['keyword'] = keyword
                        
                        # Connection configuration
                        if idx > 0:
                            st.markdown("**Connect from:**")
                            available_nodes = [n for i, n in enumerate(st.session_state.current_workflow['nodes']) if i < idx]
                            for prev_node in available_nodes:
                                if st.checkbox(
                                    f"{prev_node['type']} - {prev_node['subtype']}",
                                    key=f"conn_{prev_node['id']}_{node['id']}",
                                    value=any(c['from'] == prev_node['id'] and c['to'] == node['id'] 
                                            for c in st.session_state.current_workflow['connections'])
                                ):
                                    # Add connection if not exists
                                    if not any(c['from'] == prev_node['id'] and c['to'] == node['id'] 
                                              for c in st.session_state.current_workflow['connections']):
                                        add_connection(prev_node['id'], node['id'])
                                else:
                                    # Remove connection if exists
                                    st.session_state.current_workflow['connections'] = [
                                        c for c in st.session_state.current_workflow['connections']
                                        if not (c['from'] == prev_node['id'] and c['to'] == node['id'])
                                    ]
                    
                    with col2:
                        if st.button("🗑️", key=f"del_{node['id']}"):
                            # Remove node and its connections
                            st.session_state.current_workflow['nodes'] = [
                                n for n in st.session_state.current_workflow['nodes'] if n['id'] != node['id']
                            ]
                            st.session_state.current_workflow['connections'] = [
                                c for c in st.session_state.current_workflow['connections']
                                if c['from'] != node['id'] and c['to'] != node['id']
                            ]
                            st.rerun()
        else:
            st.info("No nodes in workflow. Add nodes from the Visual Builder tab.")
    
    with tab3:
        st.markdown("### Workflow Variables")
        st.info("Variables allow you to pass data between nodes and create dynamic workflows.")
        
        # Add variable
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            new_var_name = st.text_input("Variable Name", key="new_var_name")
        with col2:
            new_var_value = st.text_input("Default Value", key="new_var_value")
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Add Variable"):
                if new_var_name:
                    st.session_state.current_workflow['variables'][new_var_name] = new_var_value
                    st.rerun()
        
        # Display variables
        if st.session_state.current_workflow['variables']:
            st.markdown("**Current Variables:**")
            for var_name, var_value in st.session_state.current_workflow['variables'].items():
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    st.text_input("Name", value=var_name, disabled=True, key=f"vname_{var_name}")
                with col2:
                    new_value = st.text_input("Value", value=var_value, key=f"vval_{var_name}")
                    st.session_state.current_workflow['variables'][var_name] = new_value
                with col3:
                    if st.button("🗑️", key=f"vdel_{var_name}"):
                        del st.session_state.current_workflow['variables'][var_name]
                        st.rerun()
    
    with tab4:
        st.markdown("### Workflow Templates")
        st.info("Start with a pre-built template and customize it to your needs.")
        
        for template_id, template in st.session_state.workflow_templates.items():
            with st.expander(template['name']):
                st.markdown(f"**Nodes:** {len(template['nodes'])}")
                st.markdown(f"**Connections:** {len(template['connections'])}")
                
                if st.button(f"Use Template", key=f"template_{template_id}"):
                    st.session_state.current_workflow = {
                        'id': str(uuid.uuid4()),
                        'name': template['name'],
                        'nodes': template['nodes'].copy(),
                        'connections': template['connections'].copy(),
                        'variables': {}
                    }
                    st.success(f"✅ Loaded template: {template['name']}")
                    st.rerun()

elif page == "🤖 AI Agents":
    st.title("🤖 AI Agents")
    
    tabs = st.tabs(["Agent Library", "Create Agent", "A/B Testing", "Performance"])
    
    with tabs[0]:
        st.markdown("### Your AI Agents")
        
        for agent_id, agent in st.session_state.agents.items():
            with st.expander(f"{agent['name']} ({MODEL_MAP.get(agent['model'], agent['model'])})"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Model:** {agent['model']} - `{MODEL_MAP.get(agent['model'], 'Unknown')}`")
                    st.markdown(f"**Temperature:** {agent['temperature']}")
                    st.markdown(f"**Max Tokens:** {agent.get('max_tokens', 2000)}")
                    st.markdown("**System Prompt:**")
                    st.code(agent['prompt'], language='text')
                
                with col2:
                    if st.button("Test", key=f"test_{agent_id}"):
                        st.session_state[f"testing_{agent_id}"] = True
                    
                    if st.button("Edit", key=f"edit_{agent_id}"):
                        st.session_state[f"editing_{agent_id}"] = True
                    
                    if st.session_state.user_role == 'admin':
                        if st.button("Delete", key=f"delete_{agent_id}"):
                            del st.session_state.agents[agent_id]
                            st.rerun()
                
                # Test interface
                if st.session_state.get(f"testing_{agent_id}", False):
                    st.markdown("---")
                    test_input = st.text_area("Test Input", key=f"test_input_{agent_id}")
                    if st.button("Run Test", key=f"run_test_{agent_id}"):
                        with st.spinner("Testing agent..."):
                            result, tokens, cost = call_openai(
                                f"{agent['prompt']}\n\n{test_input}",
                                agent['model'],
                                agent['temperature'],
                                agent.get('max_tokens', 2000)
                            )
                            st.markdown("**Output:**")
                            st.write(result)
                            st.info(f"Tokens: {tokens} | Cost: ${cost:.4f}")
    
    with tabs[1]:
        st.markdown("### Create New Agent")
        
        col1, col2 = st.columns(2)
        
        with col1:
            agent_name = st.text_input("Agent Name", placeholder="E.g., Product Description Writer")
            agent_type = st.selectbox(
                "Agent Type",
                ["Content Writer", "SEO Specialist", "Social Media Manager", 
                 "Email Marketer", "Technical Writer", "Translator", "Editor", "Custom"]
            )
            model = st.selectbox(
                "Model",
                options=list(MODEL_MAP.keys()),
                format_func=lambda x: f"{x} ({MODEL_MAP[x]})"
            )
        
        with col2:
            temperature = st.slider("Temperature (Creativity)", 0.0, 1.0, 0.7, 0.1)
            max_tokens = st.number_input("Max Tokens", min_value=100, max_value=4000, value=2000, step=100)
        
        # Prompt templates based on agent type
        prompt_templates = {
            "Content Writer": "You are an expert content writer. Create engaging, well-structured content that resonates with the target audience. Focus on clarity, value, and readability.",
            "SEO Specialist": "You are an SEO expert. Optimize content for search engines while maintaining natural readability. Include relevant keywords, meta descriptions, and structured data suggestions.",
            "Social Media Manager": "You are a social media expert. Create engaging, platform-specific content that drives engagement. Use appropriate hashtags, emojis, and calls-to-action.",
            "Email Marketer": "You are an email marketing specialist. Write compelling email content with strong subject lines, clear CTAs, and personalized messaging.",
            "Technical Writer": "You are a technical writing expert. Create clear, accurate documentation that explains complex concepts in simple terms.",
            "Translator": "You are a professional translator. Translate content accurately while preserving tone, context, and cultural nuances.",
            "Editor": "You are a professional editor. Review and improve content for grammar, style, clarity, and impact."
        }
        
        default_prompt = prompt_templates.get(agent_type, "You are a helpful AI assistant.")
        
        system_prompt = st.text_area(
            "System Prompt",
            value=default_prompt,
            height=200,
            help="This prompt defines the agent's role and behavior"
        )
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            frequency_penalty = st.slider("Frequency Penalty", 0.0, 2.0, 0.0, 0.1)
            presence_penalty = st.slider("Presence Penalty", 0.0, 2.0, 0.0, 0.1)
            top_p = st.slider("Top P", 0.0, 1.0, 1.0, 0.05)
        
        if st.button("Create Agent", type="primary"):
            if agent_name and system_prompt:
                agent_id = agent_name.lower().replace(" ", "_")
                st.session_state.agents[agent_id] = {
                    'name': agent_name,
                    'type': agent_type,
                    'model': model,
                    'prompt': system_prompt,
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    'frequency_penalty': frequency_penalty,
                    'presence_penalty': presence_penalty,
                    'top_p': top_p
                }
                st.success(f"✅ Agent '{agent_name}' created successfully!")
                st.rerun()
            else:
                st.error("Please provide both agent name and system prompt")
    
    with tabs[2]:
        st.markdown("### A/B Testing for Prompts")
        st.info("Test different prompts to find the most effective one for your use case.")
        
        # Select agent for A/B testing
        agent_list = list(st.session_state.agents.keys())
        if agent_list:
            selected_agent = st.selectbox("Select Agent to Test", agent_list)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Version A (Current)**")
                current_prompt = st.session_state.agents[selected_agent]['prompt']
                st.text_area("Prompt A", value=current_prompt, height=150, disabled=True)
            
            with col2:
                st.markdown("**Version B (Test)**")
                test_prompt = st.text_area("Prompt B", height=150, placeholder="Enter alternative prompt...")
            
            test_input = st.text_area("Test Input", placeholder="Enter test content for both versions...")
            
            if st.button("Run A/B Test", type="primary"):
                if test_prompt and test_input:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        with st.spinner("Testing Version A..."):
                            result_a, tokens_a, cost_a = call_openai(
                                f"{current_prompt}\n\n{test_input}",
                                st.session_state.agents[selected_agent]['model'],
                                st.session_state.agents[selected_agent]['temperature']
                            )
                            st.markdown("**Output A:**")
                            st.write(result_a)
                            st.info(f"Tokens: {tokens_a} | Cost: ${cost_a:.4f}")
                    
                    with col2:
                        with st.spinner("Testing Version B..."):
                            result_b, tokens_b, cost_b = call_openai(
                                f"{test_prompt}\n\n{test_input}",
                                st.session_state.agents[selected_agent]['model'],
                                st.session_state.agents[selected_agent]['temperature']
                            )
                            st.markdown("**Output B:**")
                            st.write(result_b)
                            st.info(f"Tokens: {tokens_b} | Cost: ${cost_b:.4f}")
                    
                    # Save A/B test results
                    test_id = str(uuid.uuid4())
                    st.session_state.ab_tests[test_id] = {
                        'agent': selected_agent,
                        'prompt_a': current_prompt,
                        'prompt_b': test_prompt,
                        'result_a': result_a,
                        'result_b': result_b,
                        'tokens_a': tokens_a,
                        'tokens_b': tokens_b,
                        'cost_a': cost_a,
                        'cost_b': cost_b,
                        'timestamp': datetime.now()
                    }
                    
                    st.success("✅ A/B test completed! Which version performed better?")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Use Version A"):
                            st.info("Keeping current prompt")
                    with col2:
                        if st.button("Use Version B"):
                            st.session_state.agents[selected_agent]['prompt'] = test_prompt
                            st.success("✅ Agent updated with new prompt!")
                            st.rerun()
                    st.success("✅ Agent updated with new prompt!")
                            st.rerun()
    
    with tabs[3]:
        st.markdown("### Agent Performance Analytics")
        
        if st.session_state.execution_history:
            # Agent usage stats
            agent_stats = defaultdict(lambda: {'count': 0, 'tokens': 0, 'cost': 0})
            
            for exec in st.session_state.execution_history:
                agent = exec.get('agent', 'Unknown')
                agent_stats[agent]['count'] += 1
                agent_stats[agent]['tokens'] += exec.get('tokens', 0)
                agent_stats[agent]['cost'] += exec.get('cost', 0)
            
            # Display metrics
            for agent_name, stats in agent_stats.items():
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(f"{agent_name}", f"{stats['count']} uses")
                with col2:
                    st.metric("Total Tokens", f"{stats['tokens']:,}")
                with col3:
                    st.metric("Total Cost", f"${stats['cost']:.2f}")
                with col4:
                    avg_tokens = stats['tokens'] / stats['count'] if stats['count'] > 0 else 0
                    st.metric("Avg Tokens/Use", f"{avg_tokens:.0f}")
        else:
            st.info("No performance data available yet. Run some workflows to see analytics!")

elif page == "📊 Data Sources":
    st.title("📊 Data Sources")
    
    tabs = st.tabs(["Upload Data", "Manage Sources", "Data Preview"])
    
    with tabs[0]:
        st.markdown("### Upload Your Data")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['csv', 'txt', 'json', 'xlsx'],
            accept_multiple_files=True,
            help="Upload CSV for structured data, TXT for documents, JSON for configurations"
        )
        
        if uploaded_files:
            for file in uploaded_files:
                file_id = str(uuid.uuid4())[:8]
                
                try:
                    if file.type == "text/csv":
                        df = pd.read_csv(file)
                        st.session_state.data_sources[file_id] = {
                            'name': file.name,
                            'type': 'csv',
                            'data': df.to_dict('records'),
                            'preview': df,
                            'rows': len(df),
                            'columns': list(df.columns),
                            'uploaded': datetime.now().isoformat()
                        }
                        st.success(f"✅ Uploaded {file.name} ({len(df)} rows, {len(df.columns)} columns)")
                        
                        # Show preview
                        with st.expander("Preview Data"):
                            st.dataframe(df.head(10))
                    
                    elif file.type == "text/plain":
                        content = file.read().decode("utf-8")
                        st.session_state.data_sources[file_id] = {
                            'name': file.name,
                            'type': 'txt',
                            'data': content,
                            'preview': content[:1000],
                            'size': len(content),
                            'uploaded': datetime.now().isoformat()
                        }
                        st.success(f"✅ Uploaded {file.name} ({len(content)} characters)")
                        
                        with st.expander("Preview Content"):
                            st.text_area("Content", content[:1000] + "...", height=200)
                    
                    elif file.type == "application/json":
                        content = json.loads(file.read())
                        st.session_state.data_sources[file_id] = {
                            'name': file.name,
                            'type': 'json',
                            'data': content,
                            'preview': json.dumps(content, indent=2)[:1000],
                            'uploaded': datetime.now().isoformat()
                        }
                        st.success(f"✅ Uploaded {file.name}")
                        
                        with st.expander("Preview JSON"):
                            st.json(content)
                    
                except Exception as e:
                    st.error(f"Error uploading {file.name}: {str(e)}")
    
    with tabs[1]:
        st.markdown("### Your Data Sources")
        
        if st.session_state.data_sources:
            # Filter options
            col1, col2, col3 = st.columns(3)
            with col1:
                filter_type = st.selectbox("Filter by Type", ["All", "csv", "txt", "json"])
            with col2:
                search_term = st.text_input("Search by name", "")
            
            # Display data sources
            for ds_id, ds in st.session_state.data_sources.items():
                if filter_type != "All" and ds['type'] != filter_type:
                    continue
                if search_term and search_term.lower() not in ds['name'].lower():
                    continue
                
                with st.expander(f"{ds['name']} ({ds['type']}) - {ds['uploaded'][:10]}"):
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        if ds['type'] == 'csv':
                            st.write(f"**Rows:** {ds.get('rows', 'Unknown')}")
                            st.write(f"**Columns:** {', '.join(ds.get('columns', []))}")
                            if st.checkbox("Show data", key=f"show_{ds_id}"):
                                st.dataframe(ds['preview'])
                        elif ds['type'] == 'txt':
                            st.write(f"**Size:** {ds.get('size', 'Unknown')} characters")
                            if st.checkbox("Show content", key=f"show_{ds_id}"):
                                st.text_area("Content", ds['preview'], height=200)
                        elif ds['type'] == 'json':
                            if st.checkbox("Show JSON", key=f"show_{ds_id}"):
                                st.json(ds['data'])
                    
                    with col2:
                        if st.button("🗑️ Delete", key=f"del_{ds_id}"):
                            del st.session_state.data_sources[ds_id]
                            st.rerun()
        else:
            st.info("No data sources uploaded yet. Upload files in the Upload Data tab.")
    
    with tabs[2]:
        st.markdown("### Data Preview & Analysis")
        
        if st.session_state.data_sources:
            ds_names = {ds_id: ds['name'] for ds_id, ds in st.session_state.data_sources.items()}
            selected_ds_id = st.selectbox("Select Data Source", options=list(ds_names.keys()), format_func=lambda x: ds_names[x])
            
            if selected_ds_id:
                ds = st.session_state.data_sources[selected_ds_id]
                
                if ds['type'] == 'csv':
                    st.markdown("### Data Analysis")
                    df = ds['preview']
                    
                    # Basic statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Rows", len(df))
                    with col2:
                        st.metric("Total Columns", len(df.columns))
                    with col3:
                        st.metric("Memory Usage", f"{df.memory_usage().sum() / 1024:.2f} KB")
                    
                    # Column information
                    st.markdown("#### Column Information")
                    col_info = pd.DataFrame({
                        'Column': df.columns,
                        'Type': df.dtypes.astype(str),
                        'Non-Null Count': df.count(),
                        'Unique Values': df.nunique()
                    })
                    st.dataframe(col_info)
                    
                    # Data preview with search
                    st.markdown("#### Data Preview")
                    search_col, search_val = st.columns([1, 3])
                    with search_col:
                        search_column = st.selectbox("Search Column", ["All"] + list(df.columns))
                    with search_val:
                        search_value = st.text_input("Search Value")
                    
                    if search_value:
                        if search_column == "All":
                            mask = df.astype(str).apply(lambda x: x.str.contains(search_value, case=False)).any(axis=1)
                        else:
                            mask = df[search_column].astype(str).str.contains(search_value, case=False)
                        filtered_df = df[mask]
                        st.dataframe(filtered_df)
                    else:
                        st.dataframe(df)

elif page == "📚 Knowledge Base":
    st.title("📚 Knowledge Base")
    
    tabs = st.tabs(["Knowledge Bases", "Create New", "Templates"])
    
    with tabs[0]:
        if st.session_state.knowledge_bases:
            # Search and filter
            search = st.text_input("Search knowledge bases", "")
            
            for kb_id, kb in st.session_state.knowledge_bases.items():
                if search and search.lower() not in kb['name'].lower() and search.lower() not in kb['content'].lower():
                    continue
                
                with st.expander(f"{kb['name']} - {kb['type']}"):
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.markdown(f"**Type:** {kb['type']}")
                        st.markdown(f"**Created:** {kb.get('created', 'Unknown')}")
                        st.text_area("Content", kb['content'], height=200, disabled=True, key=f"kb_content_{kb_id}")
                        
                        # Usage stats
                        usage_count = sum(1 for wf in st.session_state.workflows.values() 
                                        for node in wf.get('nodes', []) 
                                        if node.get('config', {}).get('knowledge_base_id') == kb_id)
                        st.info(f"Used in {usage_count} workflow(s)")
                    
                    with col2:
                        if st.button("Edit", key=f"edit_kb_{kb_id}"):
                            st.session_state[f"editing_kb_{kb_id}"] = True
                        
                        if st.session_state.user_role == 'admin':
                            if st.button("Delete", key=f"del_kb_{kb_id}"):
                                del st.session_state.knowledge_bases[kb_id]
                                st.rerun()
                    
                    # Edit interface
                    if st.session_state.get(f"editing_kb_{kb_id}", False):
                        st.markdown("---")
                        new_content = st.text_area("Edit Content", value=kb['content'], height=300, key=f"edit_content_{kb_id}")
                        if st.button("Save Changes", key=f"save_kb_{kb_id}"):
                            kb['content'] = new_content
                            st.success("✅ Knowledge base updated!")
                            st.session_state[f"editing_kb_{kb_id}"] = False
                            st.rerun()
        else:
            st.info("No knowledge bases created yet. Create your first one in the Create New tab.")
    
    with tabs[1]:
        st.markdown("### Create New Knowledge Base")
        
        kb_name = st.text_input("Knowledge Base Name", placeholder="E.g., Brand Voice Guidelines")
        kb_type = st.selectbox(
            "Type",
            ["Brand Guidelines", "Style Guide", "Product Information", "Company Facts", 
             "Industry Knowledge", "Compliance Rules", "Custom"]
        )
        
        # Templates for different types
        kb_templates = {
            "Brand Guidelines": """Brand Name: [Your Brand]
Brand Voice: [Professional/Casual/Friendly/etc.]
Key Messages:
- 
- 
Tone Guidelines:
- 
Do's and Don'ts:
- Do: 
- Don't: """,
            "Style Guide": """Writing Style Guidelines:
- Point of View: [First/Second/Third person]
- Sentence Length: [Short/Medium/Long]
- Paragraph Structure: 
- Vocabulary Level: 
Formatting Rules:
- Headers: 
- Lists: 
- Citations: """,
            "Product Information": """Product Name: 
Product Category: 
Key Features:
- 
- 
Benefits:
- 
Target Audience: 
USP (Unique Selling Proposition): """,
            "Company Facts": """Company Name: 
Founded: 
Mission: 
Vision: 
Values:
- 
- 
Key Products/Services:
- 
- """,
            "Industry Knowledge": """Industry: 
Key Trends:
- 
- 
Common Terminology:
- 
Best Practices:
- 
Regulations:
- """
        }
        
        # Use template or custom
        use_template = st.checkbox("Use template", value=True)
        
        if use_template and kb_type in kb_templates:
            kb_content = st.text_area(
                "Content",
                value=kb_templates[kb_type],
                height=400,
                help="Fill in the template with your specific information"
            )
        else:
            kb_content = st.text_area(
                "Content",
                height=400,
                placeholder="Enter your knowledge base content..."
            )
        
        if st.button("Create Knowledge Base", type="primary"):
            if kb_name and kb_content:
                kb_id = kb_name.lower().replace(" ", "_")
                st.session_state.knowledge_bases[kb_id] = {
                    'name': kb_name,
                    'type': kb_type,
                    'content': kb_content,
                    'created': datetime.now().isoformat()
                }
                st.success(f"✅ Knowledge Base '{kb_name}' created!")
                st.rerun()
            else:
                st.error("Please provide both name and content")
    
    with tabs[2]:
        st.markdown("### Knowledge Base Templates")
        st.info("Quick-start templates for common knowledge base types")
        
        template_categories = {
            "Marketing": ["Brand Voice", "Content Guidelines", "SEO Best Practices"],
            "Product": ["Feature Documentation", "Benefits Framework", "Competitive Analysis"],
            "Sales": ["Value Propositions", "Objection Handling", "Case Studies"],
            "Support": ["FAQ Template", "Troubleshooting Guide", "Response Templates"]
        }
        
        for category, templates in template_categories.items():
            st.markdown(f"#### {category}")
            cols = st.columns(3)
            for idx, template in enumerate(templates):
                with cols[idx % 3]:
                    if st.button(template, key=f"template_{category}_{template}", use_container_width=True):
                        st.info(f"Template '{template}' selected. Customize it in the Create New tab!")

elif page == "📝 Content Library":
    st.title("📝 Content Library")
    
    tabs = st.tabs(["All Content", "Quick Generate", "Batch Process", "Export"])
    
    with tabs[0]:
        if st.session_state.generated_content:
            # Filters and search
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                filter_workflow = st.selectbox(
                    "Filter by Workflow",
                    ["All"] + list(set(c.get('workflow', 'Unknown') for c in st.session_state.generated_content))
                )
            with col2:
                filter_date = st.date_input("Filter by Date", value=None)
            with col3:
                sort_by = st.selectbox("Sort by", ["Newest", "Oldest", "Workflow"])
            with col4:
                search = st.text_input("Search content", "")
            
            # Filter content
            filtered_content = st.session_state.generated_content
            
            if filter_workflow != "All":
                filtered_content = [c for c in filtered_content if c.get('workflow') == filter_workflow]
            
            if filter_date:
                filtered_content = [c for c in filtered_content 
                                  if datetime.fromisoformat(c['timestamp']).date() == filter_date]
            
            if search:
                filtered_content = [c for c in filtered_content 
                                  if search.lower() in str(c.get('content', '')).lower()]
            
            # Sort content
            if sort_by == "Newest":
                filtered_content = sorted(filtered_content, key=lambda x: x['timestamp'], reverse=True)
            elif sort_by == "Oldest":
                filtered_content = sorted(filtered_content, key=lambda x: x['timestamp'])
            elif sort_by == "Workflow":
                filtered_content = sorted(filtered_content, key=lambda x: x.get('workflow', ''))
            
            # Display content
            for content in filtered_content:
                with st.expander(f"{content.get('workflow', 'Direct')} - {content['timestamp'][:16]}"):
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        content_text = content.get('content', content.get('output', 'No content'))
                        st.write(content_text)
                        
                        # Metadata
                        st.markdown("---")
                        metadata_cols = st.columns(4)
                        with metadata_cols[0]:
                            st.caption(f"ID: {content['id'][:8]}")
                        with metadata_cols[1]:
                            st.caption(f"Length: {len(str(content_text))} chars")
                        with metadata_cols[2]:
                            if 'version' in content:
                                st.caption(f"Version: {content['version']}")
                    
                    with col2:
                        if st.button("📋 Copy", key=f"copy_{content['id']}"):
                            st.toast("Content copied to clipboard!")
                        
                        if st.button("✏️ Edit", key=f"edit_content_{content['id']}"):
                            st.session_state[f"editing_{content['id']}"] = True
                        
                        if st.button("🔄 Version", key=f"version_{content['id']}"):
                            # Create new version
                            new_version = content.copy()
                            new_version['id'] = str(uuid.uuid4())
                            new_version['version'] = content.get('version', 1) + 1
                            new_version['parent_id'] = content['id']
                            new_version['timestamp'] = datetime.now().isoformat()
                            st.session_state.generated_content.append(new_version)
                            st.success("✅ New version created!")
                            st.rerun()
                        
                        if st.button("🗑️ Delete", key=f"del_{content['id']}"):
                            st.session_state.generated_content = [
                                c for c in st.session_state.generated_content if c['id'] != content['id']
                            ]
                            st.rerun()
                    
                    # Edit interface
                    if st.session_state.get(f"editing_{content['id']}", False):
                        st.markdown("---")
                        edited_content = st.text_area(
                            "Edit Content",
                            value=content_text,
                            height=300,
                            key=f"edit_area_{content['id']}"
                        )
                        if st.button("Save", key=f"save_edit_{content['id']}"):
                            content['content'] = edited_content
                            content['edited'] = datetime.now().isoformat()
                            st.session_state[f"editing_{content['id']}"] = False
                            st.success("✅ Content updated!")
                            st.rerun()
        else:
            st.info("No content generated yet. Use Quick Generate or run a workflow!")
    
    with tabs[1]:
        st.markdown("### Quick Content Generation")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            content_type = st.selectbox(
                "Content Type",
                ["Blog Post", "Social Media Post", "Product Description", "Email", 
                 "Landing Page Copy", "Ad Copy", "Press Release", "Newsletter"]
            )
            
            topic = st.text_input("Topic/Subject", placeholder="What's the content about?")
            
            # Additional parameters based on content type
            if content_type == "Blog Post":
                col_a, col_b = st.columns(2)
                with col_a:
                    word_count = st.selectbox("Word Count", ["500", "1000", "1500", "2000"])
                with col_b:
                    tone = st.selectbox("Tone", ["Professional", "Casual", "Educational", "Entertaining"])
            
            elif content_type == "Social Media Post":
                platform = st.selectbox("Platform", ["LinkedIn", "Twitter", "Facebook", "Instagram"])
                include_hashtags = st.checkbox("Include hashtags", value=True)
            
            elif content_type == "Email":
                email_type = st.selectbox("Email Type", ["Marketing", "Newsletter", "Welcome", "Follow-up"])
                
            # Knowledge base selection
            kb_options = ["None"] + list(st.session_state.knowledge_bases.keys())
            selected_kb = st.selectbox("Use Knowledge Base", kb_options)
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Generate", type="primary", use_container_width=True):
                if topic:
                    # Build prompt based on content type
                    prompt = f"Create a {content_type} about: {topic}"
                    
                    if content_type == "Blog Post":
                        prompt += f"\nWord count: approximately {word_count} words"
                        prompt += f"\nTone: {tone}"
                    elif content_type == "Social Media Post":
                        prompt += f"\nPlatform: {platform}"
                        if include_hashtags:
                            prompt += "\nInclude relevant hashtags"
                    
                    # Add knowledge base context
                    if selected_kb != "None":
                        kb = st.session_state.knowledge_bases[selected_kb]
                        prompt += f"\n\nUse this context:\n{kb['content']}"
                    
                    with st.spinner(f"Generating {content_type}..."):
                        result, tokens, cost = call_openai(prompt, model="4.1")
                        
                        # Save to content library
                        content_item = {
                            'id': str(uuid.uuid4()),
                            'type': content_type,
                            'topic': topic,
                            'content': result,
                            'workflow': 'Quick Generate',
                            'timestamp': datetime.now().isoformat(),
                            'tokens': tokens,
                            'cost': cost
                        }
                        st.session_state.generated_content.append(content_item)
                        
                        st.success("✅ Content generated successfully!")
                        
                        # Display result
                        st.markdown("### Generated Content")
                        st.write(result)
                        
                        # Quick actions
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if st.button("📋 Copy to Clipboard"):
                                st.toast("Content copied!")
                        with col2:
                            if st.button("💾 Save to Library"):
                                st.toast("Already saved to library!")
                        with col3:
                            if st.button("🔄 Regenerate"):
                                st.rerun()
                else:
                    st.error("Please enter a topic")
    
    with tabs[2]:
        st.markdown("### Batch Content Processing")
        st.info("Process multiple content pieces at once using your data sources")
        
        # Select data source
        if st.session_state.data_sources:
            csv_sources = {ds_id: ds for ds_id, ds in st.session_state.data_sources.items() if ds['type'] == 'csv'}
            
            if csv_sources:
                selected_source = st.selectbox(
                    "Select Data Source",
                    options=list(csv_sources.keys()),
                    format_func=lambda x: csv_sources[x]['name']
                )
                
                if selected_source:
                    df = csv_sources[selected_source]['preview']
                    
                    # Column selection
                    col1, col2 = st.columns(2)
                    with col1:
                        input_column = st.selectbox("Input Column", df.columns)
                    with col2:
                        id_column = st.selectbox("ID Column (optional)", ["None"] + list(df.columns))
                    
                    # Batch settings
                    batch_template = st.text_area(
                        "Content Template",
                        placeholder="Use {column_name} to reference data columns",
                        help="Example: Write a product description for {product_name} with features: {features}"
                    )
                    
                    # Select rows to process
                    max_rows = min(len(df), 100)
                    num_rows = st.slider("Number of rows to process", 1, max_rows, min(5, max_rows))
                    
                    # Preview
                    if batch_template:
                        st.markdown("### Preview (first row)")
                        first_row = df.iloc[0]
                        preview_text = batch_template
                        for col in df.columns:
                            preview_text = preview_text.replace(f"{{{col}}}", str(first_row[col]))
                        st.info(preview_text)
                    
                    if st.button("Start Batch Processing", type="primary"):
                        if batch_template:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            results = []
                            for idx, row in df.head(num_rows).iterrows():
                                # Update progress
                                progress = (idx + 1) / num_rows
                                progress_bar.progress(progress)
                                status_text.text(f"Processing row {idx + 1} of {num_rows}...")
                                
                                # Build prompt
                                prompt = batch_template
                                for col in df.columns:
                                    prompt = prompt.replace(f"{{{col}}}", str(row[col]))
                                
                                # Generate content
                                result, tokens, cost = call_openai(prompt, model="4.1", temperature=0.7)
                                
                                # Save result
                                content_item = {
                                    'id': str(uuid.uuid4()),
                                    'content': result,
                                    'workflow': 'Batch Process',
                                    'source_id': row[id_column] if id_column != "None" else idx,
                                    'timestamp': datetime.now().isoformat(),
                                    'tokens': tokens,
                                    'cost': cost
                                }
                                st.session_state.generated_content.append(content_item)
                                results.append(content_item)
                                
                                # Small delay to avoid rate limits
                                time.sleep(0.5)
                            
                            progress_bar.progress(1.0)
                            status_text.text("✅ Batch processing complete!")
                            
                            # Show summary
                            total_tokens = sum(r['tokens'] for r in results)
                            total_cost = sum(r['cost'] for r in results)
                            
                            st.success(f"""
                            Batch processing completed!
                            - Processed: {len(results)} items
                            - Total tokens: {total_tokens:,}
                            - Total cost: ${total_cost:.2f}
                            """)
                            
                            # Option to download results
                            if st.button("Download Results as CSV"):
                                results_df = pd.DataFrame([
                                    {
                                        'ID': r['source_id'],
                                        'Content': r['content'],
                                        'Tokens': r['tokens'],
                                        'Cost': r['cost']
                                    } for r in results
                                ])
                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    "Download CSV",
                                    csv,
                                    "batch_results.csv",
                                    "text/csv"
                                )
            else:
                st.warning("No CSV data sources available. Upload a CSV file in Data Sources.")
        else:
            st.warning("No data sources available. Upload files in Data Sources first.")
    
    with tabs[3]:
        st.markdown("### Export Content")
        
        if st.session_state.generated_content:
            # Export options
            export_format = st.selectbox("Export Format", ["CSV", "JSON", "TXT", "Markdown"])
            
            # Content selection
            col1, col2 = st.columns(2)
            with col1:
                date_range = st.date_input(
                    "Date Range",
                    value=(datetime.now().date(), datetime.now().date()),
                    key="export_date_range"
                )
            with col2:
                workflow_filter = st.multiselect(
                    "Workflows",
                    options=list(set(c.get('workflow', 'Unknown') for c in st.session_state.generated_content)),
                    default=[]
                )
            
            # Filter content for export
            export_content = st.session_state.generated_content
            
            if workflow_filter:
                export_content = [c for c in export_content if c.get('workflow') in workflow_filter]
            
            if len(date_range) == 2:
                start_date, end_date = date_range
                export_content = [
                    c for c in export_content
                    if start_date <= datetime.fromisoformat(c['timestamp']).date() <= end_date
                ]
            
            st.info(f"Ready to export {len(export_content)} content pieces")
            
            if st.button("Export", type="primary"):
                if export_format == "CSV":
                    df = pd.DataFrame([
                        {
                            'ID': c['id'],
                            'Workflow': c.get('workflow', 'Unknown'),
                            'Content': c.get('content', c.get('output', '')),
                            'Timestamp': c['timestamp'],
                            'Tokens': c.get('tokens', 0),
                            'Cost': c.get('cost', 0)
                        } for c in export_content
                    ])
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv,
                        f"content_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv"
                    )
                
                elif export_format == "JSON":
                    json_data = json.dumps(export_content, indent=2)
                    st.download_button(
                        "Download JSON",
                        json_data,
                        f"content_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        "application/json"
                    )
                
                elif export_format == "TXT":
                    txt_content = "\n\n---\n\n".join([
                        f"Workflow: {c.get('workflow', 'Unknown')}\n"
                        f"Date: {c['timestamp']}\n\n"
                        f"{c.get('content', c.get('output', ''))}"
                        for c in export_content
                    ])
                    st.download_button(
                        "Download TXT",
                        txt_content,
                        f"content_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        "text/plain"
                    )
                
                elif export_format == "Markdown":
                    md_content = "\n\n".join([
                        f"## {c.get('workflow', 'Unknown')} - {c['timestamp'][:10]}\n\n"
                        f"{c.get('content', c.get('output', ''))}"
                        for c in export_content
                    ])
                    st.download_button(
                        "Download Markdown",
                        md_content,
                        f"content_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        "text/markdown"
                    )
        else:
            st.info("No content to export yet.")

elif page == "📈 Analytics":
    st.title("📈 Analytics Dashboard")
    
    render_analytics_dashboard()
    
    # Additional analytics
    if st.session_state.execution_history:
        st.markdown("### Detailed Analytics")
        
        # Cost analysis
        st.markdown("#### Cost Analysis")
        df = pd.DataFrame(st.session_state.execution_history)
        
        if 'timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            daily_cost = df.groupby('date')['cost'].sum().reset_index()
            
            fig = px.bar(daily_cost, x='date', y='cost', title='Daily Cost Breakdown',
                        labels={'cost': 'Cost ($)', 'date': 'Date'})
            fig.update_traces(marker_color='#FF5B04')
            st.plotly_chart(fig, use_container_width=True)
        
        # Model usage
        st.markdown("#### Model Usage Distribution")
        model_usage = defaultdict(int)
        for exec in st.session_state.execution_history:
            if 'agent' in exec and exec['agent'] in st.session_state.agents:
                model = st.session_state.agents[exec['agent']].get('model', 'Unknown')
                model_usage[model] += 1
        
        if model_usage:
            fig = px.pie(
                values=list(model_usage.values()),
                names=list(model_usage.keys()),
                title='Model Usage Distribution',
                color_discrete_sequence=['#FF5B04', '#075056', '#F4D40C']
            )
            st.plotly_chart(fig, use_container_width=True)

elif page == "🧪 A/B Testing":
    st.title("🧪 A/B Testing Center")
    
    tabs = st.tabs(["New Test", "Active Tests", "Test Results"])
    
    with tabs[0]:
        st.markdown("### Create New A/B Test")
        
        test_name = st.text_input("Test Name", placeholder="E.g., Blog Intro Optimization")
        test_type = st.selectbox("Test Type", ["Prompt Variation", "Model Comparison", "Temperature Testing", "Workflow Comparison"])
        
        if test_type == "Prompt Variation":
            # Select agent
            agent_list = list(st.session_state.agents.keys())
            if agent_list:
                selected_agent = st.selectbox("Select Agent", agent_list)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Version A**")
                    prompt_a = st.text_area("Prompt A", height=150)
                
                with col2:
                    st.markdown("**Version B**")
                    prompt_b = st.text_area("Prompt B", height=150)
                
                # Test parameters
                num_tests = st.number_input("Number of test runs", min_value=5, max_value=50, value=10)
                test_input = st.text_area("Test Input", placeholder="Enter content to test with both versions...")
                
                if st.button("Start A/B Test", type="primary"):
                    if prompt_a and prompt_b and test_input:
                        # Run tests
                        results_a = []
                        results_b = []
                        
                        progress_bar = st.progress(0)
                        
                        for i in range(num_tests):
                            progress_bar.progress((i + 1) / (num_tests * 2))
                            
                            # Test Version A
                            result_a, tokens_a, cost_a = call_openai(
                                f"{prompt_a}\n\n{test_input}",
                                st.session_state.agents[selected_agent]['model'],
                                st.session_state.agents[selected_agent]['temperature']
                            )
                            results_a.append({'result': result_a, 'tokens': tokens_a, 'cost': cost_a})
                            
                            # Test Version B
                            result_b, tokens_b, cost_b = call_openai(
                                f"{prompt_b}\n\n{test_input}",
                                st.session_state.agents[selected_agent]['model'],
                                st.session_state.agents[selected_agent]['temperature']
                            )
                            results_b.append({'result': result_b, 'tokens': tokens_b, 'cost': cost_b})
                        
                        progress_bar.progress(1.0)
                        
                        # Analyze results
                        avg_tokens_a = sum(r['tokens'] for r in results_a) / len(results_a)
                        avg_tokens_b = sum(r['tokens'] for r in results_b) / len(results_b)
                        avg_cost_a = sum(r['cost'] for r in results_a) / len(results_a)
                        avg_cost_b = sum(r['cost'] for r in results_b) / len(results_b)
                        
                        # Display results
                        st.markdown("### Test Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Version A Performance**")
                            st.metric("Avg Tokens", f"{avg_tokens_a:.0f}")
                            st.metric("Avg Cost", f"${avg_cost_a:.4f}")
                            
                            # Show sample outputs
                            with st.expander("Sample Outputs"):
                                for i, result in enumerate(results_a[:3]):
                                    st.write(f"**Run {i+1}:**")
                                    st.write(result['result'][:200] + "...")
                                    st.markdown("---")
                        
                        with col2:
                            st.markdown("**Version B Performance**")
                            st.metric("Avg Tokens", f"{avg_tokens_b:.0f}")
                            st.metric("Avg Cost", f"${avg_cost_b:.4f}")
                            
                            with st.expander("Sample Outputs"):
                                for i, result in enumerate(results_b[:3]):
                                    st.write(f"**Run {i+1}:**")
                                    st.write(result['result'][:200] + "...")
                                    st.markdown("---")
                        
                        # Recommendation
                        if avg_cost_a < avg_cost_b:
                            st.success("✅ Version A is more cost-effective")
                        else:
                            st.success("✅ Version B is more cost-effective")
                        
                        # Save test results
                        test_id = str(uuid.uuid4())
                        if 'ab_test_results' not in st.session_state:
                            st.session_state.ab_test_results = {}
                        
                        st.session_state.ab_test_results[test_id] = {
                            'name': test_name,
                            'type': test_type,
                            'agent': selected_agent,
                            'prompt_a': prompt_a,
                            'prompt_b': prompt_b,
                            'results_a': results_a,
                            'results_b': results_b,
                            'avg_tokens_a': avg_tokens_a,
                            'avg_tokens_b': avg_tokens_b,
                            'avg_cost_a': avg_cost_a,
                            'avg_cost_b': avg_cost_b,
                            'timestamp': datetime.now().isoformat()
                        }

elif page == "👥 Team":
    st.title("👥 Team Management")
    
    if st.session_state.user_role != 'admin':
        st.warning("You need admin access to manage team settings.")
    else:
        tabs = st.tabs(["Team Members", "Roles & Permissions", "Activity Log"])
        
        with tabs[0]:
            st.markdown("### Team Members")
            
            # Mock team data
            team_members = [
                {"name": "You", "email": "admin@company.com", "role": "Admin", "status": "Active"},
                {"name": "Sarah Chen", "email": "sarah@company.com", "role": "Editor", "status": "Active"},
                {"name": "Mike Johnson", "email": "mike@company.com", "role": "Viewer", "status": "Active"},
            ]
            
            for member in team_members:
                col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
                with col1:
                    st.write(f"**{member['name']}**")
                    st.caption(member['email'])
                with col2:
                    st.write(member['role'])
                with col3:
                    st.write(member['status'])
                with col4:
                    if member['name'] != "You":
                        if st.button("Edit", key=f"edit_{member['email']}"):
                            st.info("Edit functionality coming soon!")
            
            st.markdown("### Invite Team Member")
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                invite_email = st.text_input("Email Address")
            with col2:
                invite_role = st.selectbox("Role", ["Viewer", "Editor", "Admin"])
            with col3:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Send Invite"):
                    st.success(f"✅ Invitation sent to {invite_email}")
        
        with tabs[1]:
            st.markdown("### Roles & Permissions")
            
            permissions = {
                "Admin": ["Create/Edit/Delete Workflows", "Manage Team", "View Analytics", "Manage Billing"],
                "Editor": ["Create/Edit Workflows", "Generate Content", "View Analytics"],
                "Viewer": ["View Workflows", "View Content", "View Analytics"]
            }
            
            for role, perms in permissions.items():
                with st.expander(f"{role} Permissions"):
                    for perm in perms:
                        st.write(f"✓ {perm}")
        
        with tabs[2]:
            st.markdown("### Recent Activity")
            
            activities = [
                {"user": "Sarah Chen", "action": "Created workflow", "target": "Email Campaign Generator", "time": "2 hours ago"},
                {"user": "Mike Johnson", "action": "Generated content", "target": "Blog Post", "time": "3 hours ago"},
                {"user": "You", "action": "Updated agent", "target": "Content Writer", "time": "5 hours ago"},
            ]
            
            for activity in activities:
                st.write(f"**{activity['user']}** {activity['action']} *{activity['target']}* - {activity['time']}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Momentic AI - Empowering Marketing Teams with AI-Powered Content Operations</p>
        <p style='font-size: 0.8em;'>Version 2.0 | Made with ❤️ for content creators</p>
    </div>
    """,
    unsafe_allow_html=True
)
