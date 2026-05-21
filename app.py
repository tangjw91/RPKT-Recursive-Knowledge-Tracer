import streamlit as st
from knowledge_tracer import KnowledgeTracer, Prerequisite
import json
from typing import List, Dict
import plotly.graph_objects as go
import networkx as nx

st.set_page_config(
    page_title="Recursive Knowledge Tracer",
    page_icon="🧠",
    layout="wide"
)

if 'tracer' not in st.session_state:
    st.session_state.tracer = KnowledgeTracer()
if 'current_tree' not in st.session_state:
    st.session_state.current_tree = None
if 'checking_depth' not in st.session_state:
    st.session_state.checking_depth = 0
if 'knowledge_status' not in st.session_state:
    st.session_state.knowledge_status = {}
if 'learning_path' not in st.session_state:
    st.session_state.learning_path = []
if 'show_explanation' not in st.session_state:
    st.session_state.show_explanation = False
if 'question_analysis' not in st.session_state:
    st.session_state.question_analysis = None
if 'expanded_unknowns' not in st.session_state:
    st.session_state.expanded_unknowns = set()

def display_prerequisite_card(prereq: Prerequisite, key_prefix: str, parent_unknown: bool = False):
    is_known = st.session_state.knowledge_status.get(prereq.name)
    is_expanded = prereq.name in st.session_state.expanded_unknowns
    
    card_style = """
        <style>
        .known-card {
            opacity: 0.6;
            background-color: #f0f8f0;
        }
        .unknown-card {
            background-color: #fff5f5;
            border-left: 3px solid #ff6b6b;
        }
        .expanded-card {
            background-color: #fffbf0;
            border: 2px solid #ffa94d;
        }
        </style>
    """
    st.markdown(card_style, unsafe_allow_html=True)
    
    container_class = "known-card" if is_known else ("expanded-card" if is_expanded else "unknown-card")
    
    with st.container():
        col1, col2 = st.columns([5, 1])
        
        with col1:
            level_indicator = f"[L{prereq.depth_level}]" if prereq.depth_level > 0 else ""
            st.markdown(f"### 📚 {prereq.name} {level_indicator}")
            st.markdown(f"**What:** {prereq.explanation}")
            st.markdown(f"**Why needed:** {prereq.relevance}")
            if prereq.example:
                st.info(f"💡 {prereq.example}")
        
        with col2:
            st.markdown("###  ")
            
            # Check if this concept was already evaluated elsewhere
            if prereq.name in st.session_state.knowledge_status:
                already_known = st.session_state.knowledge_status[prereq.name]
                if already_known:
                    st.success("✅ Already confirmed")
                else:
                    st.error("❌ Already confirmed")
                    # If marked unknown and not yet expanded here, expand it
                    if prereq.name not in st.session_state.expanded_unknowns:
                        st.session_state.expanded_unknowns.add(prereq.name)
                        st.session_state.tracer.expand_unknown_prerequisites(
                            [prereq], 
                            prereq.depth_level
                        )
            elif is_known is None:
                col_yes, col_no = st.columns(2)
                with col_yes:
                    if st.button("✅", key=f"{key_prefix}_{prereq.name}_know", 
                                help="I know this", use_container_width=True):
                        st.session_state.knowledge_status[prereq.name] = True
                        prereq.user_knows = True
                        st.rerun()
                
                with col_no:
                    if st.button("❌", key=f"{key_prefix}_{prereq.name}_dont", 
                                help="I don't know this", use_container_width=True):
                        st.session_state.knowledge_status[prereq.name] = False
                        prereq.user_knows = False
                        st.session_state.expanded_unknowns.add(prereq.name)
                        st.session_state.tracer.expand_unknown_prerequisites(
                            [prereq], 
                            prereq.depth_level
                        )
                        st.rerun()
            elif is_known:
                st.success("✅ Known")
            else:
                st.error("❌ Unknown")
    
    # Show expansion if marked as unknown (either here or elsewhere)
    should_expand = (is_expanded or 
                    (prereq.name in st.session_state.knowledge_status and 
                     not st.session_state.knowledge_status[prereq.name]))
    
    if should_expand and prereq.prerequisites:
        with st.container():
            st.markdown(f"#### 🔍 Prerequisites for {prereq.name}:")
            for i, sub_prereq in enumerate(prereq.prerequisites):
                display_prerequisite_card(sub_prereq, f"{key_prefix}_sub_{prereq.name}_{i}", True)

def display_prerequisites_level(prerequisites: List[Prerequisite], depth: int):
    if not prerequisites:
        return
    
    for i, prereq in enumerate(prerequisites):
        display_prerequisite_card(prereq, f"depth_{depth}_card_{i}")

def create_knowledge_graph(root: Prerequisite):
    G = nx.DiGraph()
    
    def add_nodes_edges(node: Prerequisite, parent_name: str = None):
        node_color = 'lightgreen' if node.user_knows else 'lightcoral' if node.user_knows == False else 'lightblue'
        G.add_node(node.name, color=node_color, level=node.depth_level)
        
        if parent_name:
            G.add_edge(parent_name, node.name)
        
        for prereq in node.prerequisites:
            add_nodes_edges(prereq, node.name)
    
    add_nodes_edges(root)
    
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    edge_trace = go.Scatter(
        x=[], y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)
    
    node_trace = go.Scatter(
        x=[], y=[],
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            size=[20 - 2*G.nodes[node]['level'] for node in G.nodes()],
            line_width=2
        ),
        text=[],
        textposition="top center",
        hovertext=[]
    )
    
    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)
        level = G.nodes[node]['level']
        node_trace['text'] += (f"{node}\n(L{level})",)
        node_trace['hovertext'] += (f"{node} - Level {level}",)
        
    colors = [G.nodes[node]['color'] for node in G.nodes()]
    node_trace['marker']['color'] = colors
    
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500
        )
    )
    
    return fig

def display_nested_learning_path(root: Prerequisite):
    def build_tree_structure(node: Prerequisite, prefix=""):
        tree_lines = []
        
        # Sync with session state
        if node.name in st.session_state.knowledge_status:
            node.user_knows = st.session_state.knowledge_status[node.name]
        
        if node.user_knows == False or node.user_knows is None:
            status = "❌" if node.user_knows == False else "❓"
            tree_lines.append(f"{prefix}{status} {node.name} (Level {node.depth_level})")
            
            for i, prereq in enumerate(node.prerequisites):
                is_last = i == len(node.prerequisites) - 1
                extension = "└── " if is_last else "├── "
                continuation = "    " if is_last else "│   "
                
                tree_lines.append(f"{prefix}{extension}{prereq.name}")
                subtree = build_tree_structure(prereq, prefix + continuation)
                tree_lines.extend(subtree)
        
        return tree_lines
    
    tree = build_tree_structure(root)
    return "\n".join(tree)

def main():
    st.title("🧠 Recursive Prerequisite Knowledge Tracer")
    st.markdown("*Learn efficiently by understanding what you don't know*")
    
    st.sidebar.header("How it works")
    st.sidebar.markdown("""
    1. **Ask a question** about any topic
    2. **Understand the topic** - we'll explain what you want to learn
    3. **Check prerequisites** - mark what you know (✅) or don't know (❌)
    4. **Trace deeper** - unknown concepts expand automatically
    5. **Learn efficiently** - get a personalized learning path
    """)
    
    st.sidebar.divider()
    
    education_level = st.sidebar.selectbox(
        "Your education level:",
        ["Elementary School", "Middle School", "High School", 
         "Undergraduate", "Graduate", "PhD/Research", "Professional", "Self-taught"],
        index=3  # Default to Undergraduate
    )
    
    max_level = st.sidebar.slider("Maximum recursion depth (Levels):", 1, 6, 3)
    st.session_state.tracer.max_depth = max_level + 1
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_question = st.text_area(
            "What would you like to learn about?",
            placeholder="E.g., How does backpropagation work in neural networks?",
            height=100
        )
    
    with col2:
        st.markdown("###  ")
        if st.button("🔍 Analyze Prerequisites", type="primary", use_container_width=True):
            if user_question:
                with st.spinner("Analyzing your question..."):
                    # First analyze the question
                    st.session_state.question_analysis = st.session_state.tracer.analyze_question(
                        user_question, education_level
                    )
                    
                    # Create the tree with key concepts as prerequisites
                    from knowledge_tracer import Prerequisite
                    st.session_state.current_tree = Prerequisite(
                        name=user_question,
                        explanation=st.session_state.question_analysis.get('explanation', ''),
                        relevance="This is what you want to learn",
                        depth_level=0
                    )
                    
                    # Convert key concepts to Prerequisite objects
                    key_concepts = st.session_state.question_analysis.get('key_concepts', [])
                    prerequisites = []
                    for concept in key_concepts:
                        if isinstance(concept, dict):
                            prereq = Prerequisite(
                                name=concept.get('name', ''),
                                explanation=concept.get('explanation', ''),
                                relevance=concept.get('relevance', ''),
                                example=concept.get('example'),
                                depth_level=1
                            )
                        else:
                            # Fallback for simple string concepts
                            prereq = Prerequisite(
                                name=str(concept),
                                explanation=f"Understanding {concept}",
                                relevance=f"Required for {user_question}",
                                depth_level=1
                            )
                        prerequisites.append(prereq)
                    
                    st.session_state.current_tree.prerequisites = prerequisites
                    st.session_state.checking_depth = 0
                    st.session_state.knowledge_status = {}
                    st.session_state.expanded_unknowns = set()
                    st.session_state.show_explanation = False
    
    if st.session_state.question_analysis:
        st.divider()
        
        with st.expander("📖 Understanding Your Question", expanded=True):
            st.markdown("### What you want to learn:")
            st.info(st.session_state.question_analysis.get('explanation', ''))
            
            st.markdown("### Why it's important:")
            st.success(st.session_state.question_analysis.get('importance', ''))
            
            key_concepts = st.session_state.question_analysis.get('key_concepts', [])
            if key_concepts:
                st.markdown("### Key concepts to check:")
                st.markdown("*These will be checked for your understanding below*")
                cols = st.columns(min(len(key_concepts), 3))
                for i, concept in enumerate(key_concepts):
                    with cols[i % 3]:
                        if isinstance(concept, dict):
                            st.markdown(f"• **{concept.get('name', concept)}**")
                        else:
                            st.markdown(f"• **{concept}**")
    
    if st.session_state.current_tree:
        st.divider()
        
        tab1, tab2, tab3 = st.tabs(["📊 Knowledge Check", "🗺️ Knowledge Map", "📚 Learning Path"])
        
        with tab1:
            st.markdown("### Check Your Prerequisites")
            st.markdown("Click ✅ if you know it, ❌ if you don't. Unknown concepts will expand automatically.")
            
            display_prerequisites_level(
                st.session_state.current_tree.prerequisites, 
                0
            )
            
            all_checked = all(
                prereq.user_knows is not None 
                for prereq in st.session_state.current_tree.prerequisites
            )
            
            if all_checked:
                st.success("✅ Prerequisite check complete!")
                
                if st.button("📖 Generate Personalized Explanation", type="primary"):
                    st.session_state.show_explanation = True
        
        with tab2:
            st.markdown("### Your Knowledge Graph")
            st.markdown("*Node size decreases with depth, labels show level*")
            
            if st.session_state.current_tree.prerequisites:
                fig = create_knowledge_graph(st.session_state.current_tree)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("**Legend:**")
                cols = st.columns(3)
                with cols[0]:
                    st.markdown("🟢 **Known** concepts")
                with cols[1]:
                    st.markdown("🔴 **Unknown** concepts")
                with cols[2]:
                    st.markdown("🔵 **Unchecked** concepts")
        
        with tab3:
            st.markdown("### Your Personalized Learning Path")
            
            learning_path = st.session_state.tracer.get_learning_path(st.session_state.current_tree)
            
            if learning_path:
                st.markdown("**Nested structure showing dependencies:**")
                
                tree_view = display_nested_learning_path(st.session_state.current_tree)
                st.code(tree_view, language="markdown")
            else:
                st.info("Complete the knowledge check to see your personalized learning path.")
        
        if st.session_state.show_explanation:
            st.divider()
            st.markdown("## 📖 Personalized Explanation")
            
            known_concepts = [
                name for name, status in st.session_state.knowledge_status.items() 
                if status
            ]
            unknown_concepts = [
                name for name, status in st.session_state.knowledge_status.items() 
                if not status
            ]
            
            with st.spinner("Generating explanation based on your knowledge gaps..."):
                explanation = st.session_state.tracer.generate_comprehensive_explanation(
                    user_question,
                    known_concepts,
                    unknown_concepts
                )
                st.markdown(explanation)

if __name__ == "__main__":
    main()