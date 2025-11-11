"""
Project Selection Page - Intermediate page for selecting/creating projects before entering a module
"""

import streamlit as st
from database import db
from datetime import datetime


def show_project_selection_page(module, module_display_name):
    """
    Show project selection page for a module
    
    Args:
        module: Module identifier ('eda', 'modeling', 'optimizer')
        module_display_name: Display name for the module ('EDA', 'Modeling', 'Optimizer')
    """
    
    # Custom CSS matching app theme
    st.markdown("""
    <style>
        /* Main header styling */
        .project-header {
            background: linear-gradient(135deg, #FFBD59 0%, #FF9A3D 100%);
            padding: 3rem 2rem;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 15px rgba(255, 189, 89, 0.3);
        }
        .project-header h1 {
            color: white;
            font-size: 2.5rem;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        .project-header p {
            color: rgba(255,255,255,0.95);
            font-size: 1.1rem;
            margin-top: 0.5rem;
        }
        
        /* Project cards */
        .project-card {
            background: white;
            border: 2px solid #f0f2f6;
            border-left: 5px solid #FFBD59;
            padding: 1.5rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        .project-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 20px rgba(255, 189, 89, 0.2);
            border-left-color: #41C185;
        }
        .project-name {
            font-size: 1.4rem;
            font-weight: 600;
            color: #2E86AB;
            margin-bottom: 0.5rem;
        }
        .project-desc {
            font-size: 0.95rem;
            color: #666;
            margin-bottom: 0.8rem;
            line-height: 1.5;
        }
        .project-meta {
            font-size: 0.85rem;
            color: #999;
            display: flex;
            gap: 1rem;
        }
        .project-meta span {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
        }
        
        /* Create project section */
        .create-section {
            background: linear-gradient(135deg, #41C185 0%, #2E9B6A 100%);
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 15px rgba(65, 193, 133, 0.2);
        }
        .create-section h3 {
            color: white;
            margin: 0 0 1rem 0;
        }
        
        /* Stats cards */
        .stat-card {
            background: white;
            border: 2px solid #f0f2f6;
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        .stat-value {
            font-size: 2rem;
            font-weight: bold;
            color: #FFBD59;
        }
        .stat-label {
            font-size: 0.9rem;
            color: #666;
            margin-top: 0.3rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Get user's projects first
    if 'username' not in st.session_state:
        st.error("❌ Please login first")
        return
    
    username = st.session_state['username']
    projects = db.get_projects(username, module)
    
    # Simpler header
    st.title(f"📁 {module_display_name} Projects")
    st.markdown(f"**Welcome back, {username}!** Select a project or create a new one to get started.")
    st.markdown("---")
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 🏠 Navigation")
        if st.button("⬅️ Back to Home", use_container_width=True):
            st.session_state['current_page'] = 'home'
            st.rerun()
        st.markdown("---")
        st.markdown(f"**Logged in as:** {username}")
        if st.button("🚪 Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Two column layout: Projects on left, Create on right
    main_col, create_col = st.columns([2, 1])
    
    with create_col:
        st.markdown("### ➕ Create New Project")
        
        new_project_name = st.text_input(
            "Project Name",
            placeholder="e.g., Q4 2024 Campaign",
            key=f"{module}_new_project_name"
        )
        
        new_project_desc = st.text_area(
            "Description",
            placeholder="What is this project about?",
            key=f"{module}_new_project_desc",
            height=100
        )
        
        create_btn = st.button("🚀 Create & Open", key=f"{module}_create_btn", type="primary", use_container_width=True)
        
        if create_btn:
            if new_project_name:
                if db.create_project(username, module, new_project_name, new_project_desc):
                    st.session_state[f'{module}_current_project'] = new_project_name
                    st.session_state['current_page'] = module
                    st.success(f"✅ Created!")
                    st.rerun()
                else:
                    st.error("❌ Already exists!")
            else:
                st.warning("⚠️ Enter name")
        
        # Stats in create column
        if projects:
            st.markdown("---")
            st.markdown("**📊 Your Stats**")
            st.metric("Total Projects", len(projects))
            latest = max(projects, key=lambda x: x['created_at'])
            st.caption(f"Latest: {latest['name'][:20]}")
    
    with main_col:
    
        # Existing Projects Section
        if projects:
            st.markdown(f"### 📂 Your Projects ({len(projects)})")
            
            # Search and Sort controls
            col_search, col_sort, col_order = st.columns([3, 2, 1])
            
            with col_search:
                search = st.text_input("🔍 Search projects", placeholder="Type to filter...", key=f"{module}_search", label_visibility="collapsed")
            
            with col_sort:
                sort_by = st.selectbox(
                    "Sort by",
                    ["Last Accessed", "Name (A-Z)", "Name (Z-A)", "Created (Newest)", "Created (Oldest)"],
                    key=f"{module}_sort",
                    label_visibility="collapsed"
                )
            
            with col_order:
                view_mode = st.selectbox(
                    "View",
                    ["📋 List", "🎴 Grid"],
                    key=f"{module}_view",
                    label_visibility="collapsed"
                )
            
            st.divider()
            
            # Filter projects by search
            filtered_projects = projects
            if search:
                filtered_projects = [p for p in projects if search.lower() in p['name'].lower() or search.lower() in p.get('description', '').lower()]
            
            # Sort projects
            if sort_by == "Last Accessed":
                filtered_projects = sorted(filtered_projects, key=lambda x: x['last_accessed'], reverse=True)
            elif sort_by == "Name (A-Z)":
                filtered_projects = sorted(filtered_projects, key=lambda x: x['name'].lower())
            elif sort_by == "Name (Z-A)":
                filtered_projects = sorted(filtered_projects, key=lambda x: x['name'].lower(), reverse=True)
            elif sort_by == "Created (Newest)":
                filtered_projects = sorted(filtered_projects, key=lambda x: x['created_at'], reverse=True)
            elif sort_by == "Created (Oldest)":
                filtered_projects = sorted(filtered_projects, key=lambda x: x['created_at'])
            
            if not filtered_projects:
                st.info("🔍 No projects found matching your search")
            else:
                # Display based on view mode
                if view_mode == "📋 List":
                    # List view - single column
                    for i, project in enumerate(filtered_projects):
                        desc_text = project.get('description', 'No description')
                        if len(desc_text) > 100:
                            desc_text = desc_text[:100] + "..."
                        
                        # Project card
                        st.markdown(f"""
                            <div class="project-card">
                                <div class="project-name">📁 {project['name']}</div>
                                <div class="project-desc">{desc_text}</div>
                                <div class="project-meta">
                                    <span>📅 Created {project['created_at'][:10]}</span>
                                    <span>🕒 Accessed {project['last_accessed'][:10]}</span>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Action buttons
                        btn_col1, btn_col2 = st.columns([3, 1])
                        with btn_col1:
                            if st.button(f"📂 Open Project", key=f"open_{module}_{i}", use_container_width=True, type="primary"):
                                st.session_state[f'{module}_current_project'] = project['name']
                                db.update_project_access(username, module, project['name'])
                                st.session_state['current_page'] = module
                                st.rerun()
                        
                        with btn_col2:
                            if st.button("🗑️", key=f"del_{module}_{i}", help="Delete project"):
                                if db.delete_project(username, module, project['name']):
                                    st.success(f"✅ Deleted!")
                                    st.rerun()
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                
                else:
                    # Grid view - two columns
                    cols_per_row = 2
                    for i in range(0, len(filtered_projects), cols_per_row):
                        cols = st.columns(cols_per_row)
                        
                        for j, col in enumerate(cols):
                            if i + j < len(filtered_projects):
                                project = filtered_projects[i + j]
                                
                                with col:
                                    desc_text = project.get('description', 'No description')
                                    if len(desc_text) > 80:
                                        desc_text = desc_text[:80] + "..."
                                    
                                    # Project card
                                    st.markdown(f"""
                                        <div class="project-card">
                                            <div class="project-name">📁 {project['name']}</div>
                                            <div class="project-desc">{desc_text}</div>
                                            <div class="project-meta">
                                                📅 {project['created_at'][:10]} | 
                                                🕒 {project['last_accessed'][:10]}
                                            </div>
                                        </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Action buttons
                                    btn_col1, btn_col2 = st.columns([2, 1])
                                    with btn_col1:
                                        if st.button(f"📂 Open", key=f"open_grid_{module}_{i}_{j}", use_container_width=True, type="primary"):
                                            st.session_state[f'{module}_current_project'] = project['name']
                                            db.update_project_access(username, module, project['name'])
                                            st.session_state['current_page'] = module
                                            st.rerun()
                                    
                                    with btn_col2:
                                        if st.button("🗑️", key=f"del_grid_{module}_{i}_{j}", help="Delete"):
                                            if db.delete_project(username, module, project['name']):
                                                st.success("✅ Deleted!")
                                                st.rerun()
                                    
                                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    with btn_col3:
                        if st.button("🗑️ Delete", key=f"del_{module}_{i}", use_container_width=True):
                            if db.delete_project(username, module, project['name']):
                                st.success("✅ Deleted")
                                st.rerun()
                    
                    st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.info("👈 Create your first project to get started!")
